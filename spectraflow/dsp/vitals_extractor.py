"""Vital-sign extraction: respiration rate (RPM) and heart rate (BPM).

Pipeline, per analysis step, over a sliding window of CSI frames:

1. **Sensing signal.** The window's per-subcarrier sanitized phase is scored by
   temporal variance and the strongest ``dominant_subcarriers`` are averaged.
   Selecting per window (rather than fixing a subcarrier once) adapts to fading,
   while averaging several of them suppresses the per-subcarrier noise that is
   otherwise the dominant error term.

   The averaging is done over *individually* sanitized phases, never over the
   phase vector as a whole: sanitization removes the cross-subcarrier mean, so
   averaging first would annihilate exactly the signal we want.

2. **Uniform resampling.** Frames arrive at jittery intervals, and an FFT
   requires a uniform grid, so the window is interpolated onto one at the
   nominal rate before spectral analysis.

3. **Band-pass filtering.** A 2nd-order Butterworth band-pass isolates each
   band: 0.1-0.5 Hz (6-30 RPM) for respiration and 0.8-2.5 Hz (48-150 BPM) for
   the heartbeat.

4. **Spectral peak + confidence.** The Hann-windowed FFT of each filtered band
   is searched for its dominant peak, refined to sub-bin accuracy, and scored
   against the out-of-band median noise floor. The resulting SNR maps through a
   logistic to a confidence in ``[0, 1]``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from spectraflow.config import SpectraflowConfig, default_config
from spectraflow.dsp import _backend
from spectraflow.dsp._numpy_dsp import Biquad
from spectraflow.dsp.phase_sanitizer import PhaseSanitizer
from spectraflow.ingestion.udp_receiver import CsiFrame

__all__ = ["VitalSignsExtractor", "VitalsResult"]


@dataclass(slots=True)
class VitalsResult:
    """Outcome of one analysis step.

    ``bpm``/``rpm`` are ``None`` until the estimator has enough data and a peak
    that clears ``min_snr_db``; the UI renders those as ``--`` rather than
    inventing a number.
    """

    bpm: float | None = None
    rpm: float | None = None
    confidence: float = 0.0
    bpm_confidence: float = 0.0
    rpm_confidence: float = 0.0
    bpm_snr_db: float = 0.0
    rpm_snr_db: float = 0.0
    motion: float = 0.0
    presence: bool = False
    ready: bool = False
    frames: int = 0
    window_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready mapping matching the frozen WebSocket contract."""
        return {
            "bpm": None if self.bpm is None else round(float(self.bpm), 2),
            "rpm": None if self.rpm is None else round(float(self.rpm), 2),
            "confidence": round(float(self.confidence), 4),
            "bpm_confidence": round(float(self.bpm_confidence), 4),
            "rpm_confidence": round(float(self.rpm_confidence), 4),
            "bpm_snr": round(float(self.bpm_snr_db), 2),
            "rpm_snr": round(float(self.rpm_snr_db), 2),
        }


#: Relative frame-to-frame amplitude change that maps to a motion score of
#: ~0.63, where "relative" means normalised by the mean channel amplitude. A
#: relative measure is what makes the score comparable across link qualities;
#: an absolute one would read "high motion" on a strong link and "no motion" on
#: a weak one for the same physical movement.
_MOTION_RELATIVE_SCALE = 0.05
#: EMA weight for amplitude smoothing. Raw per-frame amplitude is dominated by
#: receiver noise; smoothing before differencing suppresses it by ~sqrt(1/alpha)
#: while leaving real (slower) motion intact.
_MOTION_SMOOTHING = 0.3
#: Motion score above which a person is presumed present regardless of vitals.
_MOTION_PRESENCE_THRESHOLD = 0.18


class VitalSignsExtractor:
    """Stateful, streaming vital-sign estimator.

    Feed it CSI frames in arrival order and read the latest :class:`VitalsResult`
    from each :meth:`update` call. Analysis runs at most every
    ``vitals_update_seconds`` because a sliding 5 s window does not change
    meaningfully faster than that, and re-running the FFT per frame would burn
    the thermal budget for no information gain.

    Example::

        extractor = VitalSignsExtractor()
        for frame in stream:
            result = extractor.update(frame)
    """

    __slots__ = (
        "config",
        "_sanitizer",
        "_times",
        "_phases",
        "_smooth_amplitude",
        "_prev_amplitude",
        "_motion_ema",
        "_resp_history",
        "_heart_history",
        "_resp_misses",
        "_heart_misses",
        "_last_analysis_t",
        "_result",
        "_analysis_count",
    )

    def __init__(self, config: SpectraflowConfig | None = None) -> None:
        self.config = config or default_config()
        self.config.validate()
        self._sanitizer = PhaseSanitizer(unwrap=True)

        capacity = int(np.ceil(self.config.window_seconds * self.config.sample_rate_hz)) + 8
        self._times: deque[float] = deque(maxlen=capacity)
        self._phases: deque[npt.NDArray[np.float32]] = deque(maxlen=capacity)

        self._smooth_amplitude: npt.NDArray[np.float32] | None = None
        self._prev_amplitude: npt.NDArray[np.float32] | None = None
        self._motion_ema = 0.0

        # Rolling history of recent *valid* peak frequencies per band, used to
        # gate on cross-window stability rather than trusting a single frame's
        # spectrum.
        self._resp_history: deque[float] = deque(maxlen=8)
        self._heart_history: deque[float] = deque(maxlen=8)
        self._resp_misses = 0
        self._heart_misses = 0

        self._last_analysis_t: float | None = None
        self._result = VitalsResult()
        self._analysis_count = 0

    # -- public API --------------------------------------------------------
    def update(self, frame: CsiFrame) -> VitalsResult:
        """Ingest one CSI frame and return the current estimate."""
        t = frame.timestamp_seconds
        self._times.append(t)
        self._phases.append(self._sanitizer.process(frame.csi))

        self._update_motion(frame.csi)

        due = (
            self._last_analysis_t is None
            or (t - self._last_analysis_t) >= self.config.vitals_update_seconds
            or self._analysis_count == 0
        )
        if due and self._has_enough_data():
            self._analyze()
            self._last_analysis_t = t
            self._analysis_count += 1

        return self._result

    def process_matrix(
        self,
        csi: npt.ArrayLike,
        *,
        timestamps: npt.ArrayLike | None = None,
        fs: float | None = None,
    ) -> VitalsResult:
        """Analyse a batch of frames at once (offline analysis and tests).

        Frames are replayed through :meth:`update` rather than analysed in one
        shot. That is not just code reuse: the estimator gates each reported
        rate on cross-window stability, so a single analysis of the whole batch
        would always abstain. Replaying keeps batch and streaming behaviour
        identical by construction.

        Args:
            csi: ``(n_frames, n_subcarriers)`` complex CSI, oldest first.
            timestamps: Optional per-frame capture times in seconds; a uniform
                grid at ``fs`` is assumed otherwise.
            fs: Sample rate; defaults to the configured rate.

        Returns:
            The estimator's state after the whole batch has been consumed.
        """
        matrix = np.asarray(csi)
        if matrix.ndim != 2:
            raise ValueError("csi must be a 2-D (frames, subcarriers) array")
        if matrix.shape[0] == 0:
            return self._result

        rate = float(fs if fs is not None else self.config.sample_rate_hz)
        if rate <= 0:
            raise ValueError("fs must be > 0")

        if timestamps is None:
            times = np.arange(matrix.shape[0], dtype=np.float64) / rate
        else:
            times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if times.size != matrix.shape[0]:
                raise ValueError("timestamps length must match the frame count")

        self.reset()
        for index, (t, row) in enumerate(zip(times, matrix)):
            self.update(
                CsiFrame(
                    node_id=0,
                    sequence=index,
                    timestamp_us=int(round(float(t) * 1e6)),
                    csi=np.ascontiguousarray(row, dtype=np.complex64),
                )
            )
        return self._result

    @property
    def result(self) -> VitalsResult:
        return self._result

    @property
    def frames_seen(self) -> int:
        return len(self._times)

    def reset(self) -> None:
        """Drop all buffered state."""
        self._sanitizer.reset()
        self._times.clear()
        self._phases.clear()
        self._smooth_amplitude = None
        self._prev_amplitude = None
        self._motion_ema = 0.0
        self._resp_history.clear()
        self._heart_history.clear()
        self._resp_misses = 0
        self._heart_misses = 0
        self._last_analysis_t = None
        self._result = VitalsResult()
        self._analysis_count = 0

    # -- internals ---------------------------------------------------------
    def _update_motion(self, csi: npt.ArrayLike) -> None:
        """Track gross motion from the relative frame-to-frame amplitude change.

        Motion and vital signs live in disjoint frequency bands, so the
        discriminator is how fast the channel amplitude changes: breathing
        perturbs it by a fraction of a percent per frame, a limb movement by
        several percent.

        Two details make this usable. The amplitude is smoothed with an EMA
        first, because raw per-frame amplitude is dominated by receiver noise
        and would otherwise make an empty room look like a moving one. The
        change is then expressed *relative* to the mean amplitude, so the score
        means the same thing on a strong and a weak link.
        """
        amplitude = np.abs(np.asarray(csi, dtype=np.complex64)).astype(np.float32)

        if self._smooth_amplitude is None or self._smooth_amplitude.shape != amplitude.shape:
            # First frame, or the subcarrier count changed: seed and skip.
            self._smooth_amplitude = amplitude.copy()
            self._prev_amplitude = None
            return

        self._smooth_amplitude = (
            (1.0 - _MOTION_SMOOTHING) * self._smooth_amplitude
            + _MOTION_SMOOTHING * amplitude
        )

        if self._prev_amplitude is not None:
            scale = float(np.mean(np.abs(self._smooth_amplitude))) + 1e-6
            delta = float(np.mean(np.abs(self._smooth_amplitude - self._prev_amplitude)))
            relative = delta / scale
            instant = 1.0 - float(np.exp(-relative / _MOTION_RELATIVE_SCALE))
            self._motion_ema = 0.85 * self._motion_ema + 0.15 * instant

        self._prev_amplitude = self._smooth_amplitude.copy()

    def _has_enough_data(self) -> bool:
        """True once the buffer spans enough time and holds enough samples."""
        if len(self._times) < 8:
            return False
        span = self._times[-1] - self._times[0]
        return span >= self.config.window_seconds * 0.6

    def _analysis_rate(self) -> float:
        """Sample rate actually observed, falling back to the configured one."""
        if len(self._times) >= 2:
            span = self._times[-1] - self._times[0]
            if span > 0:
                return (len(self._times) - 1) / span
        return self.config.sample_rate_hz

    def _sensing_signal(self, rate_hz: float) -> npt.NDArray[np.float64]:
        """Build the uniform, clutter-free scalar sensing signal.

        The subject's motion is *common mode* across subcarriers -- every
        subcarrier sees the same chest displacement -- whereas receiver noise is
        independent per subcarrier. That difference is the single best SNR lever
        available, so rather than averaging a few hand-picked subcarriers this
        projects the window onto its first principal component, which is the
        maximum-likelihood common-mode waveform.

        The projection is taken *after* band-limiting to the union of the vital
        bands, so the component captures subject motion rather than slow
        environmental drift -- which is also common mode, and would otherwise
        dominate the first component.

        Returns an empty array when the window is too short or degenerate.
        """
        phases = np.stack(list(self._phases))  # (frames, subcarriers)
        if phases.shape[0] < 8:
            return np.empty(0, dtype=np.float64)

        times = np.asarray(self._times, dtype=np.float64)
        span = float(times[-1] - times[0])
        if span <= 0:
            return np.empty(0, dtype=np.float64)

        step = 1.0 / float(rate_hz)
        n_uniform = int(np.floor(span / step)) + 1
        if n_uniform < 8:
            return np.empty(0, dtype=np.float64)
        grid = times[0] + np.arange(n_uniform) * step

        # The FFT requires a uniform grid, and CSI frames arrive on a jittery
        # one. This is a vectorised linear interpolation across all subcarriers
        # at once -- a per-column np.interp loop is measurably too slow at
        # 64 subcarriers x several analyses per second.
        uniform = self._resample_uniform(times, phases, grid)

        # Remove DC and any linear trend per subcarrier before band-limiting.
        index = np.arange(n_uniform, dtype=np.float64)
        index_centred = index - index.mean()
        denom = float(np.sum(index_centred * index_centred))
        uniform -= uniform.mean(axis=0, keepdims=True)
        if denom > 0:
            slopes = (index_centred @ uniform) / denom
            uniform -= np.outer(index_centred, slopes)

        # Band-limit to the union of the vital bands, dropping both the drift
        # below 0.1 Hz and the out-of-band noise above 2.5 Hz.
        #
        # Applied as a spectral mask rather than a time-domain IIR: the mask is
        # applied to every subcarrier in a single vectorised operation, whereas
        # an IIR recursion is inherently sequential and would need one Python
        # loop per subcarrier. The per-band Butterworth filtering that the
        # estimator's accuracy depends on still happens below, on the single
        # 1-D sensing signal where its exact response matters.
        nyquist = rate_hz / 2.0
        f_lo = min(self.config.respiration_band_hz[0], self.config.heart_band_hz[0])
        f_hi = min(
            max(self.config.respiration_band_hz[1], self.config.heart_band_hz[1]),
            nyquist * 0.98,
        )
        if f_lo < f_hi:
            uniform = self._spectral_band_limit(uniform, rate_hz, f_lo, f_hi)

        # Retain the strongest subcarriers to bound cost and discard dead ones.
        k = min(int(self.config.dominant_subcarriers), uniform.shape[1])
        if k < uniform.shape[1]:
            energy = np.einsum("ij,ij->j", uniform, uniform)
            top = np.argpartition(energy, -k)[-k:]
            uniform = uniform[:, top]

        try:
            # First principal component: left singular vector scaled by its
            # singular value, i.e. the common-mode waveform itself.
            u, s, _ = np.linalg.svd(uniform, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return uniform.mean(axis=1)
        if s.size == 0 or s[0] <= 0:
            return np.empty(0, dtype=np.float64)
        return (u[:, 0] * s[0]).astype(np.float64)

    def _analyze(self, rate_hz: float | None = None) -> None:
        """Run one spectral analysis pass and refresh :attr:`_result`."""
        cfg = self.config
        rate = float(rate_hz if rate_hz is not None else self._analysis_rate())
        if rate <= 0:
            return

        signal = self._sensing_signal(rate)
        if signal.size < 8:
            return

        # Remove DC *and the linear trend*. A 5 s window of a slow random walk
        # (thermal drift, a door settling) looks almost like a ramp, and a ramp
        # leaks heavily into the respiration band. The band-pass alone does not
        # remove it: the filter's zero is at DC, not along a slope.
        n = signal.size
        index = np.arange(n, dtype=np.float64)
        index_centred = index - index.mean()
        centred = signal - signal.mean()
        denom = float(np.sum(index_centred * index_centred))
        slope = float(np.sum(index_centred * centred) / denom) if denom > 0 else 0.0
        signal = centred - slope * index_centred

        # Broadband noise reference, measured once on the unfiltered signal and
        # shared by both bands. Scoring against a reference taken from a
        # *band-passed* spectrum would be meaningless (the filter has already
        # removed the very bins the reference is drawn from), and that is what
        # made an empty room report a confident heart rate.
        noise_floor = self._broadband_noise_floor(signal, rate)

        resp = self._band_peak(
            signal,
            rate,
            cfg.respiration_band_hz,
            noise_floor=noise_floor,
            min_snr_db=cfg.min_snr_db,
        )

        # Search the heart band with the respiration harmonics suppressed. The
        # respiration signal is 10-20x larger than the cardiac one, so its 2nd
        # and 3rd harmonics fall inside 0.8-2.5 Hz and would otherwise be
        # reported as a heart rate (a 16 RPM breath puts its 3rd harmonic at
        # 0.81 Hz == 48 BPM, right at the band edge).
        resp_reference = resp.frequency_hz if resp.valid else None
        heart = self._band_peak(
            signal,
            rate,
            cfg.heart_band_hz,
            suppress_harmonics_of=resp_reference,
            noise_floor=noise_floor,
            min_snr_db=cfg.heart_min_snr_db,
        )
        resp_rpm, self._resp_misses = self._gate(
            self._resp_history, resp, self._resp_misses
        )
        heart_bpm, self._heart_misses = self._gate(
            self._heart_history, heart, self._heart_misses
        )

        confidences = [c for c, v in ((resp.confidence, resp.valid), (heart.confidence, heart.valid)) if v]
        confidence = float(np.mean(confidences)) if confidences else 0.0

        motion = self._motion_ema
        presence = bool(
            resp_rpm is not None
            or heart_bpm is not None
            or motion >= _MOTION_PRESENCE_THRESHOLD
        )

        self._result = VitalsResult(
            bpm=heart_bpm,
            rpm=resp_rpm,
            confidence=confidence,
            bpm_confidence=float(heart.confidence),
            rpm_confidence=float(resp.confidence),
            bpm_snr_db=float(heart.snr_db),
            rpm_snr_db=float(resp.snr_db),
            motion=motion,
            presence=presence,
            ready=True,
            frames=len(self._times),
            window_seconds=float(self._times[-1] - self._times[0]),
        )

    @staticmethod
    def _resample_uniform(
        times: npt.NDArray[np.float64],
        matrix: npt.NDArray[np.float64],
        grid: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Linearly interpolate all columns of ``matrix`` onto ``grid``.

        Vectorised across subcarriers: each column shares the same time base, so
        the search indices and weights are computed once for the whole matrix.
        """
        count = times.size
        idx = np.searchsorted(times, grid, side="right") - 1
        idx = np.clip(idx, 0, count - 2)
        t0 = times[idx]
        t1 = times[idx + 1]
        span = np.maximum(t1 - t0, 1e-12)
        weight = np.clip((grid - t0) / span, 0.0, 1.0)[:, None]
        return matrix[idx] * (1.0 - weight) + matrix[idx + 1] * weight

    @staticmethod
    def _spectral_band_limit(
        matrix: npt.NDArray[np.float64],
        rate_hz: float,
        f_lo: float,
        f_hi: float,
    ) -> npt.NDArray[np.float64]:
        """Zero every Fourier component outside ``[f_lo, f_hi]``, per column.

        The mask edges are tapered over a few bins so the abrupt truncation does
        not ring in the time domain and leak energy back into the passband.
        """
        n = matrix.shape[0]
        if n < 2:
            return np.zeros_like(matrix)

        # Transform at the exact length. Zero-padding to a power of two and then
        # truncating the inverse transform back to n would be a circular
        # operation *plus* a rectangular truncation -- and that truncation leaks
        # broadband energy back across the whole spectrum, which raises the
        # out-of-band floor the SNR is measured against. NumPy's FFT accepts any
        # length, so there is no reason to pad here.
        spectrum = np.fft.rfft(matrix, axis=0)
        freqs = np.fft.rfftfreq(n, d=1.0 / float(rate_hz))
        if freqs.size < 2:
            return np.zeros_like(matrix)

        # Raised-cosine transitions two bins wide on each side, so the mask is
        # continuous and does not ring in the time domain.
        transition = 2.0 * float(freqs[1] - freqs[0])
        rise = 0.5 * (
            1.0
            - np.cos(np.pi * np.clip((freqs - (f_lo - transition)) / transition, 0.0, 1.0))
        )
        fall = 0.5 * (
            1.0
            + np.cos(np.pi * np.clip((freqs - f_hi) / transition, 0.0, 1.0))
        )
        weights = np.minimum(rise, fall)
        if not np.any(weights > 0.0):
            return np.zeros_like(matrix)

        return np.fft.irfft(spectrum * weights[:, None], n=n, axis=0)

    def _broadband_noise_floor(
        self, signal: npt.NDArray[np.float64], rate: float
    ) -> float:
        """Noise-floor magnitude measured outside the vital bands.

        Uses the *unfiltered* detrended signal, so the estimate is drawn from a
        wide, signal-free region (0.5-0.8 Hz and above 2.5 Hz) and is therefore
        a stable reference. This is the quantity that actually separates a real
        vital sign (~40 dB above it) from an empty room (~15 dB), where the
        in-band peak-to-local-median ratio separates them not at all.

        Returns 0.0 when the spectrum is too small to yield a reference, which
        makes :func:`find_band_peak` fall back to its own internal estimate.
        """
        n = int(signal.size)
        if n < 8:
            return 0.0
        nfft = max(int(self.config.nfft), 1 << max(0, (n - 1).bit_length()))
        spectrum = _backend.magnitude_spectrum(signal, nfft)
        bin_hz = _backend.bin_width_hz(nfft, rate)
        if bin_hz <= 0.0:
            return 0.0

        power = np.asarray(spectrum, dtype=np.float64) ** 2
        f_lo = min(self.config.respiration_band_hz[0], self.config.heart_band_hz[0])
        f_hi = min(
            max(self.config.respiration_band_hz[1], self.config.heart_band_hz[1]),
            rate / 2.0,
        )
        mask = np.ones(power.size, dtype=bool)
        mask[: max(1, int(np.ceil(f_lo / bin_hz)))] = False
        mask[int(np.floor(f_hi / bin_hz)) :] = False
        if int(mask.sum()) < 4:
            return 0.0
        return float(np.sqrt(np.median(power[mask])))

    def _band_peak(
        self,
        signal: npt.NDArray[np.float64],
        rate: float,
        band: tuple[float, float],
        suppress_harmonics_of: float | None = None,
        noise_floor: float = 0.0,
        min_snr_db: float | None = None,
    ) -> _backend.SpectralPeak:
        """Band-pass, window and locate the dominant peak in one band."""
        f_lo, f_hi = band
        nyquist = rate / 2.0
        if f_hi >= nyquist:
            # The requested band exceeds what this rate can represent; clip
            # rather than throw so a slow link degrades instead of failing.
            f_hi = nyquist * 0.98
        if f_lo >= f_hi:
            return _backend.SpectralPeak()

        coeffs = _backend.design_bandpass(f_lo, f_hi, rate)
        filtered = Biquad(*coeffs).process_block(signal)

        length = filtered.size
        nfft = max(int(self.config.nfft), 1 << max(0, (length - 1).bit_length()))
        spectrum = _backend.magnitude_spectrum(filtered, nfft)
        bin_hz = _backend.bin_width_hz(nfft, rate)

        if suppress_harmonics_of is not None and suppress_harmonics_of > 0.0:
            spectrum = self._suppress_harmonics(
                spectrum, bin_hz, suppress_harmonics_of, f_lo, f_hi
            )

        return _backend.find_band_peak(
            spectrum,
            bin_hz,
            f_lo,
            f_hi,
            snr_reference_db=self.config.snr_reference_db,
            min_snr_db=(
                self.config.min_snr_db if min_snr_db is None else min_snr_db
            ),
            noise_floor=noise_floor,
        )

    def _suppress_harmonics(
        self,
        spectrum: npt.NDArray[np.float64],
        bin_hz: float,
        f_fundamental: float,
        f_lo: float,
        f_hi: float,
    ) -> npt.NDArray[np.float64]:
        """Attenuate integer harmonics of ``f_fundamental`` inside a band.

        Respiration is far stronger than the cardiac signal, so its harmonics
        contaminate the heart band. Notching them out (rather than needing a
        longer window to resolve them) is the standard mitigation for short
        analysis windows.
        """
        if bin_hz <= 0.0:
            return spectrum
        out = spectrum.copy()
        gain = float(self.config.harmonic_suppression)
        max_harmonic = int(np.floor(f_hi / f_fundamental)) if f_fundamental > 0 else 0
        for k in range(2, max_harmonic + 1):
            frequency = k * f_fundamental
            if frequency < f_lo or frequency > f_hi:
                continue
            centre = int(round(frequency / bin_hz))
            lo = max(0, centre - 1)
            hi = min(out.size - 1, centre + 1)
            if lo <= hi:
                out[lo : hi + 1] *= gain
        return out

    def _gate(
        self,
        history: deque[float],
        peak: _backend.SpectralPeak,
        misses: int,
    ) -> tuple[float | None, int]:
        """Convert a single-window peak into a rate, or ``None``.

        A spectral peak alone is not evidence of a vital sign: slow
        environmental drift also produces a strong low-frequency peak, and a
        single 5 s window cannot tell them apart. What *does* separate them is
        persistence -- a physiological rhythm holds its frequency while a drift
        peak wanders. So the reported value is the median of the recent valid
        estimates, and it is withheld until enough windows agree within
        ``stability_tolerance``.

        Returns:
            ``(rate_or_None, miss_count)``.
        """
        cfg = self.config
        if not peak.valid:
            misses += 1
            if misses > cfg.stability_max_misses:
                # Sustained loss of signal: drop the history so a stale rate is
                # never reported after the subject has left.
                history.clear()
            return None, misses

        misses = 0
        history.append(float(peak.frequency_hz))
        if len(history) < cfg.stability_min_samples:
            return None, misses

        values = np.asarray(history, dtype=np.float64)
        median = float(np.median(values))
        if median <= 0.0:
            return None, misses

        q1, q3 = np.percentile(values, [25.0, 75.0])
        spread = float((q3 - q1) / median)
        if spread > cfg.stability_tolerance:
            # Frequency is wandering: not a physiological rhythm.
            return None, misses

        return median * 60.0, misses

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<VitalSignsExtractor frames={len(self._times)} "
            f"ready={self._result.ready} backend={_backend.BACKEND}>"
        )


def result_fields() -> list[str]:
    """Field names of :class:`VitalsResult` (used by tests)."""
    return list(asdict(VitalsResult()).keys())

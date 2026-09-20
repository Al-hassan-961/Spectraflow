"""Pure-NumPy DSP primitives.

These are the reference implementations that back the pipeline whenever the
native ``spectraflow_core`` extension is unavailable. They are written to match
``core/src/dsp_filters.cpp`` numerically, so switching backends never changes
results -- only speed. ``tests/test_dsp.py`` asserts that equivalence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "Biquad",
    "SpectralPeak",
    "bin_width_hz",
    "design_butterworth_bandpass",
    "find_band_peak",
    "hann_window",
    "magnitude_spectrum",
    "next_power_of_two",
    "sanitize_phase",
    "snr_to_confidence",
    "unwrap_phase",
]

_EPSILON = 1e-20
_MIN_DB = -300.0


# ---------------------------------------------------------------------------
# Phase sanitization
# ---------------------------------------------------------------------------


def unwrap_phase(phase: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Unwrap a phase sequence so consecutive steps stay within +/- pi."""
    arr = np.asarray(phase, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float32)
    return np.unwrap(arr).astype(np.float32)


def sanitize_phase(phase: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Linear phase unwrapping removing the CFO/SFO ramp and the mean offset.

        phase_sanitized[i] = phase[i]
                             - ((phase[N-1] - phase[0]) / (N - 1)) * i
                             - mean(phase)

    The slope is estimated from the endpoints, which is what makes the
    transform insensitive to the *absolute* phase (unavailable on commodity
    hardware) while still cancelling the linear ramp that Carrier and Sampling
    Frequency Offset impose across subcarriers.

    Note that the mean subtracted is that of the *input* phase, not of the
    de-ramped signal, so the result is not zero-mean in general. The defining
    property is that the linear trend is gone: the endpoint-to-endpoint slope of
    the output is zero.
    """
    arr = np.asarray(phase, dtype=np.float64)
    n = arr.size
    if n == 0:
        return arr.astype(np.float32)
    if n == 1:
        return np.zeros(1, dtype=np.float32)

    slope = (arr[-1] - arr[0]) / (n - 1)
    mean = arr.mean()
    idx = np.arange(n, dtype=np.float64)
    return (arr - slope * idx - mean).astype(np.float32)


# ---------------------------------------------------------------------------
# IIR filtering
# ---------------------------------------------------------------------------


def design_butterworth_bandpass(
    f_lo: float, f_hi: float, fs: float
) -> tuple[float, float, float, float, float]:
    """Design a 2nd-order Butterworth band-pass biquad.

    Bilinear transform with frequency pre-warping applied to the
    low-pass -> band-pass substitution ``s -> (s^2 + w0^2) / (s * BW)`` on a
    first-order Butterworth prototype. The result has unity gain at the band
    centre ``sqrt(f_lo * f_hi)``.

    Returns:
        ``(b0, b1, b2, a1, a2)`` with ``a0`` normalised to 1.

    Raises:
        ValueError: if the band is unsatisfiable for the given rate.
    """
    if not fs > 0:
        raise ValueError("fs must be > 0")
    if not f_lo > 0:
        raise ValueError("f_lo must be > 0")
    if not f_hi > f_lo:
        raise ValueError("require f_hi > f_lo")
    if f_hi >= fs / 2.0:
        raise ValueError("f_hi must be below the Nyquist frequency")

    wa_lo = math.tan(math.pi * f_lo / fs)
    wa_hi = math.tan(math.pi * f_hi / fs)
    bw = wa_hi - wa_lo
    w0sq = wa_lo * wa_hi

    a0 = 1.0 + bw + w0sq
    a1 = -2.0 + 2.0 * w0sq
    a2 = 1.0 - bw + w0sq

    return (bw / a0, 0.0, -bw / a0, a1 / a0, a2 / a0)


class Biquad:
    """Single-section IIR filter with persistent state (transposed direct form II).

    Holding state across frames is what lets the filter run continuously on a
    stream instead of restarting -- and ringing -- on every analysis window.
    """

    __slots__ = ("b0", "b1", "b2", "a1", "a2", "_z1", "_z2", "initialised")

    def __init__(self, b0: float, b1: float, b2: float, a1: float, a2: float) -> None:
        self.b0, self.b1, self.b2, self.a1, self.a2 = (
            float(b0),
            float(b1),
            float(b2),
            float(a1),
            float(a2),
        )
        self._z1 = 0.0
        self._z2 = 0.0
        self.initialised = False

    @classmethod
    def bandpass(cls, f_lo: float, f_hi: float, fs: float) -> "Biquad":
        return cls(*design_butterworth_bandpass(f_lo, f_hi, fs))

    def process(self, x: float) -> float:
        y = self.b0 * x + self._z1
        self._z1 = self.b1 * x - self.a1 * y + self._z2
        self._z2 = self.b2 * x - self.a2 * y
        self.initialised = True
        return y

    def process_block(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        out = np.empty(arr.size, dtype=np.float64)
        for i, value in enumerate(arr):
            out[i] = self.process(float(value))
        return out

    def reset(self) -> None:
        self._z1 = 0.0
        self._z2 = 0.0
        self.initialised = False


def lfilter_block(
    coeffs: tuple[float, float, float, float, float], x: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Zero-state filtering of a whole block (convenience for offline analysis)."""
    return Biquad(*coeffs).process_block(x)


# ---------------------------------------------------------------------------
# Windows and spectra
# ---------------------------------------------------------------------------


def next_power_of_two(n: int) -> int:
    """Smallest power of two >= n (with a floor of 1)."""
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def hann_window(n: int) -> npt.NDArray[np.float64]:
    """Periodic Hann window (the correct choice for FFT-based analysis)."""
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))


def bin_width_hz(nfft: int, fs: float) -> float:
    """Frequency spacing of an ``nfft``-point spectrum at rate ``fs``."""
    return 0.0 if nfft <= 0 else float(fs) / float(nfft)


def magnitude_spectrum(
    signal: npt.ArrayLike, nfft: int
) -> npt.NDArray[np.float64]:
    """One-sided Hann-windowed amplitude spectrum, zero-padded to ``nfft``.

    Magnitudes are normalised by the window's coherent gain and doubled to
    account for the discarded negative frequencies (except DC and Nyquist),
    so a unit-amplitude sinusoid reads back as ~1.0.
    """
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = next_power_of_two(max(int(nfft), 2))
    half = n // 2 + 1
    if arr.size == 0:
        return np.zeros(half, dtype=np.float64)

    win = hann_window(arr.size)
    m = min(arr.size, n)
    buf = np.zeros(n, dtype=np.float64)
    buf[:m] = arr[:m] * win[:m]
    coherent_gain = float(win[:m].sum())
    if coherent_gain <= _EPSILON:
        return np.zeros(half, dtype=np.float64)

    spec = np.abs(np.fft.rfft(buf, n=n)) * (2.0 / coherent_gain)
    spec[0] *= 0.5
    if n // 2 < half:
        spec[n // 2] *= 0.5
    return spec.astype(np.float64)


# ---------------------------------------------------------------------------
# Spectral peak scoring
# ---------------------------------------------------------------------------


def snr_to_confidence(snr_db: float, reference_db: float = 30.0) -> float:
    """Logistic map from SNR in dB to a confidence in ``[0, 1]``.

    Centred on ``reference_db`` with a 5 dB scale, spanning the ~15-45 dB range a
    broadband-referenced peak occupies so the bar stays responsive rather than
    saturating at either end.
    """
    x = (float(snr_db) - float(reference_db)) / 5.0
    if x > 60.0:
        return 1.0
    if x < -60.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-x)))


@dataclass(slots=True)
class SpectralPeak:
    """Outcome of a band-limited spectral peak search."""

    frequency_hz: float = 0.0
    magnitude: float = 0.0
    peak_db: float = _MIN_DB
    snr_db: float = 0.0
    snr_linear: float = 0.0
    confidence: float = 0.0
    valid: bool = False

    def __bool__(self) -> bool:
        return self.valid


def find_band_peak(
    magnitude: npt.ArrayLike,
    bin_hz: float,
    f_lo: float,
    f_hi: float,
    snr_reference_db: float = 30.0,
    min_snr_db: float = 20.0,
    noise_floor: float = 0.0,
) -> SpectralPeak:
    """Locate the dominant in-band peak and score it against a noise floor.

    Three details make this usable for vital signs rather than merely correct:

    * the peak bin is refined by fitting a parabola through the log magnitudes
      of its neighbours, giving sub-bin frequency accuracy -- without it a 5 s
      window quantises the estimate to 0.2 Hz (12 breaths/min);
    * confidence is a smooth logistic in SNR rather than a hard threshold, so
      the UI can show a meaningful bar during warm-up;
    * the noise floor is supplied by the caller, measured on a *broadband*
      spectrum of the same signal. Callers band-pass before taking the spectrum
      they pass in here, and the filtered spectrum's out-of-band bins are
      already attenuated -- deriving the floor from them inflates the SNR for
      every input, which is how an empty room ends up reporting a heart rate.
      Passing ``noise_floor <= 0`` falls back to a floor derived from ``mag``.
    """
    mag = np.asarray(magnitude, dtype=np.float64).reshape(-1)
    peak = SpectralPeak()
    if mag.size == 0 or bin_hz <= 0.0 or not f_hi > f_lo:
        return peak

    lo_bin = max(0, int(math.ceil(f_lo / bin_hz)))
    hi_bin = min(mag.size - 1, int(math.floor(f_hi / bin_hz)))
    if lo_bin > hi_bin or lo_bin >= mag.size:
        return peak

    peak_bin = lo_bin + int(np.argmax(mag[lo_bin : hi_bin + 1]))

    if noise_floor > 0.0:
        floor = float(noise_floor)
    else:
        guard_lo = max(peak_bin - 1, 0)
        guard_hi = peak_bin + 1
        in_band = np.arange(lo_bin, hi_bin + 1)
        reference = mag[in_band[(in_band < guard_lo) | (in_band > guard_hi)]]
        if reference.size < 3:
            mask = np.ones(mag.size, dtype=bool)
            mask[0] = False
            mask[lo_bin : hi_bin + 1] = False
            reference = mag[mask]
        floor = float(np.median(reference)) if reference.size else 0.0

    peak_mag = float(mag[peak_bin])
    if peak_mag <= _EPSILON:
        # A flat/empty spectrum has no peak: reporting the first in-band bin as
        # the peak location would be misleading, so abstain entirely.
        return peak
    noise_pow = max(floor * floor, _EPSILON)
    peak_pow = peak_mag * peak_mag
    snr_db = 10.0 * math.log10(max(peak_pow, _EPSILON) / noise_pow)

    refined_bin = float(peak_bin)
    if 0 < peak_bin < mag.size - 1:
        a = math.log(max(float(mag[peak_bin - 1]), _EPSILON))
        b = math.log(max(float(mag[peak_bin]), _EPSILON))
        c = math.log(max(float(mag[peak_bin + 1]), _EPSILON))
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            delta = 0.5 * (a - c) / denom
            if -0.5 < delta < 0.5:
                refined_bin += delta

    peak.frequency_hz = refined_bin * bin_hz
    peak.magnitude = peak_mag
    peak.peak_db = 20.0 * math.log10(peak_mag) if peak_mag > _EPSILON else _MIN_DB
    peak.snr_db = snr_db
    peak.snr_linear = float(10.0 ** (snr_db / 10.0))
    peak.confidence = snr_to_confidence(snr_db, snr_reference_db)
    peak.valid = snr_db >= min_snr_db
    return peak

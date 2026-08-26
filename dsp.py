"""
Spectraflow DSP engine.

A real-time, ARM-friendly signal-processing pipeline that turns a noisy
Android RSSI stream into micro-movement intelligence.  Because Android
exposes RSSI, not CSI, we recover what physics allows: a denoised,
low-frequency "breathing envelope" from the 0.2–0.5 Hz band, with an honest
confidence score instead of a fake guarantee.

Pipeline per AP
---------------
raw RSSI ──► Adaptive Kalman filter (level + trend states)
          ──► ring buffer, resampled to a uniform grid
          ──► detrend + Hann window ──► rFFT
          ──► band-energy ratio (0.2–0.5 Hz), dominant frequency,
              parabolic peak interpolation, frequency-stability
Multi-AP fusion
    per-AP band power is weighted by 1/innovation² (inverse-noise
    weighting), giving N weakly-correlated "channels" a real SNR gain.
Classifier
    hysteresis state machine: CLEAR → PRESENCE → BREATHING / MOTION,
    with dwell times and a 0..1 confidence for the respiration estimate.

Design rules for a mobile ARM CPU
---------------------------------
* Kalman steps use scalar float algebra (no numpy.linalg per sample).
* The FFT runs at most every FFT_STRIDE samples and on 64-point windows
  (microsecond-scale cost; the scanner subprocess dwarfs it).
* SciPy is imported lazily and is strictly optional.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

import config


# ---------------------------------------------------------------------------
# 1. Adaptive Kalman filter (RSSI denoising)
# ---------------------------------------------------------------------------
class AdaptiveKalmanRSSI:
    """
    Two-state Kalman filter for a streaming RSSI value.

    State vector  x = [level, trend]      (dBm, dBm/s)
    Model         F = [[1, dt], [0, 1]]   constant-trend motion model
    Measurement   z = level + noise       H = [1, 0]

    The measurement-noise covariance R is adapted online from the squared
    innovation residual (Mehra-style), clamped to [R_MIN, R_MAX].  This is
    essential for RSSI: jitter changes with link quality, people walking
    between you and the AP, and phone orientation.  A fixed R either leaves
    noise in the estimate or lags real signal.

    The 2x2 update is expanded into scalar algebra — numpy.linalg would
    allocate arrays for every sample and measurably heat an ARM core.

    Role in the pipeline: an optimal *level* estimator is inherently
    low-pass, so feeding its output to the FFT would attenuate the very
    0.2–0.5 Hz ripple we seek.  It therefore feeds the time-domain display,
    the adaptive fusion weights (innovation), and link-quality diagnostics —
    while the FFT operates on the raw stream (see StreamingSpectrum).
    """

    __slots__ = ("level", "trend", "p00", "p01", "p11", "r", "last_t", "n", "innov_smooth")

    def __init__(self, init_level: float | None = None):
        k = config.KALMAN
        self.level = init_level if init_level is not None else k["INIT_LEVEL_DB"]
        self.trend = 0.0
        self.p00 = 1.0            # variance of level estimate (dBm^2)
        self.p01 = 0.0            # covariance level/trend
        self.p11 = 0.05           # variance of trend estimate
        self.r = k["R_INIT"]
        self.last_t: float | None = None
        self.n = 0
        self.innov_smooth = 0.0   # smoothed innovation magnitude (dB)

    def reset(self, level: float) -> None:
        """Re-seed the filter (e.g. after a long gap or AP switch)."""
        self.level = level
        self.trend = 0.0
        self.p00 = 1.0
        self.p01 = 0.0
        self.p11 = 0.05
        self.r = config.KALMAN["R_INIT"]
        self.n = 0
        self.innov_smooth = 0.0

    def update(self, z: float, t: float) -> float:
        """Feed one measurement, return the filtered level (dBm)."""
        k = config.KALMAN
        if self.last_t is not None:
            dt = max(1e-4, t - self.last_t)
        else:
            dt = 0.5
        self.last_t = t
        self.n += 1

        # ---- predict (scalar-expanded F P F^T + Q) ----
        dt2 = dt * dt
        self.p00 += 2.0 * dt * self.p01 + dt2 * self.p11 + k["Q_LEVEL"] * dt
        self.p01 += dt * self.p11
        self.p11 += k["Q_TREND"] * dt

        # ---- update ----
        innovation = z - self.level
        s = self.p00 + self.r
        if s <= 1e-12:
            s = 1e-12
        k0 = self.p00 / s
        k1 = self.p01 / s

        self.level += k0 * innovation
        self.trend += k1 * innovation

        # Joseph-form-lite covariance update (keep symmetric, stay PSD)
        p01_old = self.p01
        self.p00 = (1.0 - k0) * self.p00
        self.p01 = (1.0 - k0) * self.p01
        self.p11 -= k1 * p01_old

        # ---- adaptive R from innovation variance ----
        y2 = innovation * innovation
        self.r = (1.0 - k["R_ALPHA"]) * self.r + k["R_ALPHA"] * y2
        self.r = min(k["R_MAX"], max(k["R_MIN"], self.r))

        # smoothed |innovation| for fusion weights and diagnostics
        if self.n == 1:
            self.innov_smooth = abs(innovation)
        else:
            self.innov_smooth = 0.9 * self.innov_smooth + 0.1 * abs(innovation)
        return self.level


# ---------------------------------------------------------------------------
# 2. Streaming FFT / respiration-band spectrum
# ---------------------------------------------------------------------------
class StreamingSpectrum:
    """
    Sliding-window spectral analysis on a jittery, irregularly-timed stream.

    Every `stride` new samples the last WINDOW_SAMPLES points are resampled
    onto a uniform grid (np.interp), detrended, Hann-windowed and rFFT'd.

    A single window of noisy RSSI has enormous spectral variance, so the
    power spectrum is also averaged across windows (EWMA).  Classification
    uses the averaged spectrum; raw spectra are kept for the browser chart.

    Reported metrics:
      * ber      — band-energy ratio (0.2–0.5 Hz vs all AC power),
                   LONG-EWMA averaged (stable classification metric)
      * band_snr — mean band power per bin / median noise-band power per bin,
                   LONG-EWMA averaged
      * dom_freq — dominant frequency inside the band (parabolic peak
                   interpolation on the SHORT-EWMA spectrum), Hz
      * motion_rms — RMS of the 0.5–2 Hz bandpassed window (gross motion)
      * mags/freqs — latest raw spectrum (display)
      * breath_wave — FFT-mask bandpass of the window (0.15–0.6 Hz),
                   approximating the chest-motion envelope — pure NumPy.
    """

    def __init__(self, window_samples: int | None = None, stride: int | None = None):
        sp = config.SPECTRUM
        self.window = window_samples or sp["WINDOW_SAMPLES"]
        self.min_window = sp["MIN_WINDOW_SAMPLES"]
        self.stride = stride or sp["FFT_STRIDE"]
        self.values = deque(maxlen=self.window)
        self.times = deque(maxlen=self.window)
        self.since_fft = 0

        self.ber = 0.0
        self.dom_freq = 0.0
        self.dom_mag = 0.0
        self.band_snr = 0.0
        self.motion_rms = 0.0
        self.freqs: np.ndarray = np.zeros(1)
        self.mags: np.ndarray = np.zeros(1)
        self.breath_wave: np.ndarray = np.zeros(1)
        self.last_spectrum_t = 0.0
        self.ready = False
        self.stable = False            # long-EWMA spectrum available
        self._pw_avg: np.ndarray | None = None

    def add(self, value: float, t: float) -> None:
        self.values.append(float(value))
        self.times.append(float(t))
        self.since_fft += 1
        # progressive lock-in: spectral resolution improves as the window
        # fills, but the first spectrum appears after only min_window samples
        if self.since_fft >= self.stride and len(self.values) >= self.min_window:
            self.since_fft = 0
            self._compute(t)

    # -- internals ----------------------------------------------------------
    def _compute(self, t: float) -> None:
        sp = config.SPECTRUM
        vals = np.asarray(self.values, dtype=np.float64)
        ts = np.asarray(self.times, dtype=np.float64)
        n = len(vals)                       # grows to self.window over warm-up

        t0, t1 = ts[0], ts[-1]
        span = max(t1 - t0, 1e-6)
        # nominal grid spacing from the actual observation span
        grid = np.linspace(t0, t1, n)
        uniform = np.interp(grid, ts, vals)

        # polynomial detrend (degree from config): removes DC, ramp and slow
        # room-level curvature, flattening the low-frequency noise floor
        # that would otherwise bias the band-SNR statistic.  A 0.2–0.5 Hz
        # sinusoid spans several periods per window, so the fit absorbs
        # <1% of its power.
        uniform = uniform - np.polyval(
            np.polyfit(grid, uniform, sp["DETREND_POLY"]), grid)

        # Hann window + rFFT
        win = np.hanning(n)
        xw = uniform * win
        spec = np.fft.rfft(xw)
        mag = np.abs(spec)
        pw = mag * mag
        dt_grid = span / (n - 1)
        freqs = np.fft.rfftfreq(n, d=dt_grid)

        # EWMA averaging of the power spectrum once the window is full
        # (long average → stable classification metrics + rate estimate)
        if n == self.window:
            if self._pw_avg is None:
                self._pw_avg = pw.copy()
            else:
                a = sp["EWMA_ALPHA"]
                self._pw_avg = a * pw + (1.0 - a) * self._pw_avg
            self.stable = True
            pw_cls = self._pw_avg
        else:
            pw_cls = pw

        # band indices (skip DC bin 0)
        lo, hi = sp["RESP_LOW_HZ"], sp["RESP_HIGH_HZ"]
        band = np.where((freqs >= lo) & (freqs <= hi))[0]
        band = band[band > 0]
        ac = np.arange(1, len(mag))                     # all AC bins

        if len(band) > 0 and len(ac) > 0:
            total = pw_cls[ac].sum()
            bpower = pw_cls[band].sum()
            self.ber = float(bpower / total) if total > 1e-12 else 0.0

            # band SNR vs a noise-reference band above the breathing band
            # (0.55–1.0 Hz).  Mean-based: RSSI jitter is 1/f-colored, and
            # the reference band is systematically quieter than 0.2–0.5 Hz,
            # so the mean of the reference gives a clean noise floor that
            # keeps silence near ~1.0 while breathing reads ~2.
            nb_lo, nb_hi = sp["NOISE_BAND_HZ"]
            nb = np.where((freqs >= nb_lo) & (freqs <= nb_hi))[0]
            nb = nb[nb > 0]
            noise_per_bin = float(pw_cls[nb].mean()) if len(nb) else 1e-12
            self.band_snr = float(pw_cls[band].mean()) / (noise_per_bin + 1e-12)

            # dominant peak inside band + parabolic interpolation.
            # Uses the LONG-EWMA spectrum: the short-EWMA peak wanders across
            # noise bins and biases the rate low; the averaged spectrum locks
            # onto the true breathing bin (rate response is ~30 s, standard
            # for respiration monitors).
            k = int(np.argmax(pw_cls[band]))
            bin_idx = band[k]
            y0 = pw_cls[bin_idx - 1] if bin_idx - 1 >= 0 else pw_cls[bin_idx]
            y2 = pw_cls[bin_idx + 1] if bin_idx + 1 < len(pw_cls) else pw_cls[bin_idx]
            y1 = pw_cls[bin_idx]
            denom = (y0 - 2.0 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            delta = min(0.5, max(-0.5, delta))
            df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            self.dom_freq = float(freqs[bin_idx] + delta * df)
            self.dom_mag = float(mag[bin_idx])
        else:
            self.ber = 0.0
            self.dom_freq = 0.0
            self.dom_mag = 0.0
            self.band_snr = 0.0

        self.freqs = freqs
        self.mags = mag
        self.ready = True
        self.last_spectrum_t = t

        # breathing waveform: FFT-mask bandpass (0.15–0.6 Hz), pure NumPy
        bp = spec.copy()
        bp[freqs < 0.15] = 0.0
        bp[freqs > 0.6] = 0.0
        self.breath_wave = np.fft.irfft(bp, n=n)

        # gross-motion RMS: energy in the 0.5–2 Hz band (separates body
        # motion from respiration; noise alone contributes ~0.8 dB)
        m_lo, m_hi = sp["MOTION_BAND_HZ"]
        bm = spec.copy()
        bm[freqs < m_lo] = 0.0
        bm[freqs > m_hi] = 0.0
        motion_wave = np.fft.irfft(bm, n=n)
        self.motion_rms = float(np.sqrt(np.mean(motion_wave * motion_wave)))


# ---------------------------------------------------------------------------
# 3. Micro-motion detector (per-AP Kalman + spectrum, fused, state machine)
# ---------------------------------------------------------------------------
class MicroMotionDetector:
    """
    Owns the per-AP DSP chains and fuses them into a single classification.

    State machine (hysteresis, dwell-gated):

        CLEAR ──► PRESENCE ──► BREATHING   (elevated band energy, stable freq)
          ▲            │
          └────────────┴──► MOTION         (broadband RMS excursion)

    Public attributes (read by the app layer):
        state, confidence, breath_rate_bpm, ber_fused, motion_rms,
        freq_std, ap_innovation (per-AP smoothed |innovation|)
    """

    STATE_CLEAR = "CLEAR"
    STATE_PRESENCE = "PRESENCE"
    STATE_BREATHING = "BREATHING"
    STATE_MOTION = "MOTION"

    def __init__(self):
        d = config.DETECTOR
        self.chains: dict[str, dict] = {}          # bssid -> {"kalman","spec"}
        self.state = self.STATE_CLEAR
        self.state_since = 0.0                     # uses signal time, not wall time
        self.dwell = d["DWELL_CLEAR"]

        self.confidence = 0.0
        self.breath_rate_bpm = 0.0
        self.ber_fused = 0.0
        self.band_snr_fused = 0.0
        self.motion_rms = 0.0
        self.freq_std = 0.0
        self.dom_freq_fused = 0.0

        self._freq_hist = deque(maxlen=d["FREQ_STABILITY_WINDOWS"])
        self._motion_hist = deque(maxlen=d["MOTION_WINDOW_SAMPLES"])
        self._last_classify = 0.0
        self.last_update_t = 0.0
        self.samples_seen = 0

    # -- public API ----------------------------------------------------------
    def update(self, ap_readings: dict[str, float], t: float) -> None:
        """Feed one acquisition cycle: {bssid: raw_rssi_dBm}."""
        self.last_update_t = t
        self.samples_seen += 1

        # per-AP chain: Kalman for display/weights, raw stream for the FFT
        for bssid, rssi in ap_readings.items():
            chain = self.chains.get(bssid)
            if chain is None:
                chain = {
                    "kalman": AdaptiveKalmanRSSI(),
                    "spec": StreamingSpectrum(),
                    "display": deque(maxlen=64),
                }
                self.chains[bssid] = chain
            kal = chain["kalman"]
            filtered = kal.update(float(rssi), t)
            chain["display"].append(filtered)
            chain["spec"].add(float(rssi), t)   # raw → spectral path

        self._classify(t)

    def fused_ber(self) -> float:
        """Innovation-weighted average band-energy ratio across APs."""
        f = config.FUSION
        if not f["ENABLED"]:
            return self._best_ber()
        weights, bers = [], []
        for bssid, chain in self.chains.items():
            spec = chain["spec"]
            if not spec.ready:
                continue
            kal = chain["kalman"]
            if kal.n < f["MIN_SAMPLES_PER_AP"]:
                continue
            w = 1.0 / (kal.innov_smooth ** 2 + 1e-6) if f["WEIGHT_BY_INNOVATION"] else 1.0
            weights.append(w)
            bers.append(spec.ber)
        if not bers:
            return 0.0
        wsum = sum(weights)
        return sum(w * b for w, b in zip(weights, bers)) / wsum if wsum > 0 else sum(bers) / len(bers)

    def _best_ber(self) -> float:
        bers = [c["spec"].ber for c in self.chains.values() if c["spec"].ready]
        return max(bers) if bers else 0.0

    def _fused_snr(self) -> float:
        """Innovation-weighted band-SNR across APs (same weighting as BER)."""
        f = config.FUSION
        weights, snrs = [], []
        for bssid, chain in self.chains.items():
            spec = chain["spec"]
            if not spec.ready:
                continue
            kal = chain["kalman"]
            if kal.n < f["MIN_SAMPLES_PER_AP"]:
                continue
            w = 1.0 / (kal.innov_smooth ** 2 + 1e-6) if f["WEIGHT_BY_INNOVATION"] else 1.0
            weights.append(w)
            snrs.append(spec.band_snr)
        if not snrs:
            return 0.0
        wsum = sum(weights)
        return sum(w * s for w, s in zip(weights, snrs)) / wsum if wsum > 0 else sum(snrs) / len(snrs)

    def _spec_stable(self) -> bool:
        """True once at least one AP's averaged spectrum is available."""
        return any(c["spec"].stable for c in self.chains.values())

    def ap_innovation(self, bssid: str) -> float:
        chain = self.chains.get(bssid)
        return chain["kalman"].innov_smooth if chain else 0.0

    def filtered_series(self, bssid: str, maxlen: int = 64):
        """(times, Kalman-filtered levels) for the frontend chart."""
        chain = self.chains.get(bssid)
        if chain is None:
            return [], []
        times = list(chain["spec"].times)
        vals = list(chain["display"])[-len(times):] if times else []
        return times, vals

    def spectrum(self, bssid: str | None = None):
        """(freqs, mags, breath_wave) of the best AP for display."""
        best = self._best_ap()
        chain = self.chains.get(best or bssid or "")
        if chain is None or not chain["spec"].ready:
            return np.zeros(1), np.zeros(1), np.zeros(1)
        spec = chain["spec"]
        return spec.freqs, spec.mags, spec.breath_wave

    def _best_ap(self) -> str | None:
        best, best_w = None, -1.0
        for bssid, chain in self.chains.items():
            spec = chain["spec"]
            if not spec.ready:
                continue
            w = 1.0 / (chain["kalman"].innov_smooth ** 2 + 1e-6)
            if w > best_w:
                best_w, best = w, bssid
        return best

    # -- internals -----------------------------------------------------------
    def _classify(self, t: float) -> None:
        d = config.DETECTOR
        # fused metrics (averaged spectra once stable, raw during warm-up)
        self.ber_fused = self.fused_ber()
        self.band_snr_fused = self._fused_snr()

        dom_freqs = [c["spec"].dom_freq for c in self.chains.values()
                     if c["spec"].ready and c["spec"].dom_freq > 0]
        if dom_freqs:
            dom_freqs.sort()
            self.dom_freq_fused = dom_freqs[len(dom_freqs) // 2]   # median, robust
            self._freq_hist.append(self.dom_freq_fused)
        self.freq_std = float(np.std(self._freq_hist)) if len(self._freq_hist) >= 2 else 1.0

        # gross motion RMS from the best AP's 0.5–2 Hz band energy
        best = self._best_ap()
        if best is not None:
            self.motion_rms = self.chains[best]["spec"].motion_rms
        self._motion_hist.append(self.motion_rms)

        # confidence: band energy + band SNR + frequency stability
        conf_ber = min(1.0, max(0.0, (self.ber_fused - 0.38) / 0.25))
        conf_snr = min(1.0, max(0.0, (self.band_snr_fused - 1.35) / 1.0))
        conf_stab = min(1.0, max(0.0, 1.0 - self.freq_std / 0.10))
        self.confidence = 0.5 * conf_ber + 0.3 * conf_snr + 0.2 * conf_stab
        self.breath_rate_bpm = self.dom_freq_fused * 60.0 if self.confidence > 0.3 else 0.0

        # hysteresis state machine with dwell gating (signal time).
        # Gross motion is live (works during spectral warm-up); respiration
        # states require the averaged spectrum to be available.
        if t - self.state_since < self.dwell:
            return
        rms = self.motion_rms

        # asymmetric hysteresis: once BREATHING is declared, hold it while
        # the averaged metrics stay above a lower exit threshold, so a
        # transient disturbance (e.g. a motion burst and its ~30 s EWMA
        # tail) doesn't flap the state.  Exit requires a genuine absence.
        if self.state == self.STATE_BREATHING:
            if self.ber_fused >= 0.40 and self.band_snr_fused >= 1.20:
                self.state_since = t
                return

        if rms > d["MOTION_RMS_DB"]:
            self._set_state(self.STATE_MOTION, t)
        elif self._spec_stable():
            if (self.ber_fused >= d["BREATHING_PRESENT_BER"]
                    and self.band_snr_fused >= d["BAND_SNR_BREATHING"]
                    and self.freq_std < d["FREQ_STABILITY_HZ"]):
                self._set_state(self.STATE_BREATHING, t)
            elif (self.ber_fused >= d["BREATHING_CONFIDENCE_BER"]
                  or self.band_snr_fused >= d["BAND_SNR_PRESENCE"]):
                self._set_state(self.STATE_PRESENCE, t)
            else:
                self._set_state(self.STATE_CLEAR, t)
        else:
            self._set_state(self.STATE_CLEAR, t)

    def _set_state(self, state: str, t: float) -> None:
        if state == self.state:
            return
        self.state = state
        self.state_since = t
        self.dwell = config.DETECTOR["DWELL_" + state]


# ---------------------------------------------------------------------------
# 4. Synthetic signal generator (simulation mode + tests)
# ---------------------------------------------------------------------------
_SIM_RNG = np.random.default_rng(20260)   # deterministic across runs


def generate_synthetic_sample(t: float, phase: float = 0.0) -> float:
    """
    Deterministic, physically-plausible RSSI stream for a person breathing
    near the device, for SIMULATION=1 mode and unit tests.

    rssi(t) = base + drift + breath*sin(2π f_b t + φ) + motion_burst + noise
    """
    s = config.SIMULATION
    f_b = s["BREATH_RATE_BPM"] / 60.0

    drift = 0.4 * math.sin(2 * math.pi * 0.008 * t)          # room-level drift
    breath = s["BREATH_AMPLITUDE_DB"] * math.sin(2 * math.pi * f_b * t + phase)
    motion = 0.0
    cycle = s["MOTION_EVERY_S"]
    if cycle > 0:
        m = (t % cycle)
        if m < 4.0:                                            # 4 s burst
            envelope = math.sin(math.pi * m / 4.0)             # smooth on/off
            motion = s["MOTION_AMPLITUDE_DB"] * envelope * math.sin(2 * math.pi * 0.9 * t)
    noise = _SIM_RNG.normal(0.0, s["NOISE_DB"])
    return s["BASE_RSSI"] + drift + breath + motion + noise

"""
Spectraflow — central configuration.

All DSP and power-management knobs live here so the pipeline can be retuned
on-device without touching the algorithm code.

Units & conventions
-------------------
* RSSI is in dBm (more-negative = weaker).
* Frequencies are in Hz.  The respiration band is 0.20–0.50 Hz,
  i.e. 12–30 breaths per minute (bpm = Hz * 60).
* All sample rates are *nominal*; the pipeline adapts to the real,
  jittery sampling cadence Android actually delivers.
"""

# ---------------------------------------------------------------------------
# Sampling & acquisition
# ---------------------------------------------------------------------------
# How often we poll `termux-wifi-connectioninfo` at most, per power tier.
# Breathing (0.2–0.5 Hz) needs fs >= 1 Hz; 2 Hz gives headroom and lets the
# anti-aliasing low-pass sit comfortably below Nyquist.
SAMPLE_RATE_TIERS = {
    "ECO":          1.0,   # 1 Hz  — battery saver, coarse presence only
    "NORMAL":       2.0,   # 2 Hz  — default, respiration-grade
    "PERFORMANCE":  4.0,   # 4 Hz  — best SNR, hotter
}
ACTIVE_TIER = "NORMAL"

# Full spectrum sweeps (termux-wifi-scaninfo) are expensive; we only run them
# for the radar view and for multi-AP fusion, and never faster than this.
RADAR_SCAN_INTERVAL = 3.0

# ---------------------------------------------------------------------------
# Kalman filter (RSSI denoising)
# ---------------------------------------------------------------------------
KALMAN = {
    # Process noise for the level state (dBm^2 per second).  Tuned so the
    # filter passes 0.2–0.5 Hz content (breathing) while still cutting the
    # high-frequency jitter that would otherwise raise the FFT noise floor.
    # Too small => the filter eats the breathing ripple; too large => it
    # stops denoising.  0.03 is the empirical sweet spot for ~1.3 dB jitter.
    "Q_LEVEL": 0.03,
    # Process noise for the trend state.  Breathing appears as a slow,
    # smooth drift of the RSSI, so the trend state must be allowed to move.
    "Q_TREND": 0.0015,
    # Initial measurement noise (dBm^2).  Android RSSI jitter is ~1–3 dB.
    "R_INIT": 2.25,
    # Adaptive noise bounds.  R adapts from the innovation (Kalman "learning"
    # from the residual): clamps prevent runaway trust in a broken link.
    "R_MIN": 0.25,          # 0.5 dB std — floor, don't over-trust
    "R_MAX": 16.0,          # 4.0 dB std — ceiling, link is very noisy
    "R_ALPHA": 0.05,        # innovation-variance smoothing factor
    "INIT_LEVEL_DB": -70.0,
}

# ---------------------------------------------------------------------------
# Spectrum analysis (respiration band)
# ---------------------------------------------------------------------------
SPECTRUM = {
    "RESP_LOW_HZ": 0.20,
    "RESP_HIGH_HZ": 0.50,
    # FFT window length in samples @ nominal 2 Hz == 24 s of history
    # (df = 1/24 ≈ 0.042 Hz ≈ 2.5 bpm precision; the 0.2–0.5 Hz band spans
    # ~8 bins).  48-point rFFT is trivially cheap on ARM.
    "WINDOW_SAMPLES": 48,
    # Compute on a shorter window during warm-up so the first spectrum
    # appears after ~MIN_WINDOW/rate seconds, then lock into full windows.
    "MIN_WINDOW_SAMPLES": 24,
    # Temporal averaging of the power spectrum (EWMA across FFT windows).
    # A single 24 s window of noisy RSSI has enormous spectral variance;
    # a stable breathing sinusoid survives averaging while white noise
    # averages out.  This is the single biggest robustness win for
    # RSSI-only sensing.  Effective averaging ~ (2-α)/α windows ≈ 30 s.
    "EWMA_ALPHA": 0.12,
    # Motion band (Hz): gross body motion vs the respiration band.  RSSI
    # energy in this band, measured as the RMS of the bandpassed window,
    # separates motion from breathing without the Kalman filter.
    "MOTION_BAND_HZ": (0.5, 2.0),
    # Noise-reference band (Hz): power here estimates the per-bin noise
    # floor used for the band-SNR statistic (must sit above breathing
    # leakage, ~±0.09 Hz around the 0.5 Hz band edge).
    "NOISE_BAND_HZ": (0.55, 1.0),
    # Recompute the spectrum at most every N samples (big CPU saver;
    # the FFT is ~2 orders of magnitude cheaper than the scanner subprocess).
    "FFT_STRIDE": 4,
    # Detrend polynomial degree.  2 removes DC + ramp + slow room-level
    # curvature, flattening the 1/f noise floor that would otherwise bias
    # the band-SNR statistic.  A 0.2–0.5 Hz sinusoid spans several periods
    # per window, so the fit absorbs <1% of its power.
    "DETREND_POLY": 2,
}

# ---------------------------------------------------------------------------
# Micro-motion classification
# ---------------------------------------------------------------------------
DETECTOR = {
    # Averaged band-energy ratio (0..1): fraction of total AC spectral power
    # inside the respiration band, after EWMA spectral averaging.
    # White RSSI noise alone lands near (band bins)/(AC bins) ≈ 0.35,
    # so thresholds must sit above that floor.
    "BREATHING_PRESENT_BER": 0.45,
    "BREATHING_CONFIDENCE_BER": 0.40,
    # Band SNR: mean power per band bin / mean power per noise-band bin
    # (0.55–1.0 Hz).  RSSI jitter is 1/f-colored, so silence converges near
    # ~1.3; a real narrowband breathing signature reads ~2+.  Calibrated
    # against the sweep in tests/test_dsp.py.
    "BAND_SNR_PRESENCE": 1.35,
    "BAND_SNR_BREATHING": 1.50,
    # Dominant-frequency stability: std of the dominant bin freq over the
    # last N windows must stay below this for a confident "breathing" label.
    "FREQ_STABILITY_HZ": 0.08,
    "FREQ_STABILITY_WINDOWS": 4,
    # Gross-motion detector: RMS of the 0.5–2 Hz bandpassed window (dB).
    # Noise alone lands near ~0.8; a real body-motion burst pushes past 1.2.
    "MOTION_WINDOW_SAMPLES": 16,
    "MOTION_RMS_DB": 1.20,
    # Hysteresis dwell times (s) to prevent state flapping.
    "DWELL_CLEAR": 3.0,
    "DWELL_PRESENCE": 4.0,
    "DWELL_BREATHING": 5.0,
    "DWELL_MOTION": 2.0,
}

# ---------------------------------------------------------------------------
# Multi-AP fusion
# ---------------------------------------------------------------------------
# RSSI-only sensing is weak per-AP; fusing the top-N strongest APs gives
# N weakly-correlated "channels" and a real SNR boost for band-power sums.
FUSION = {
    "ENABLED": True,
    "MAX_APS": 3,
    "MIN_SAMPLES_PER_AP": 16,   # ignore APs with too little history
    # Fusion weight: stronger APs have cleaner variance, so weight by
    # inverse-noise (computed from each AP's Kalman innovation).
    "WEIGHT_BY_INNOVATION": True,
}

# ---------------------------------------------------------------------------
# Power / thermal management
# ---------------------------------------------------------------------------
THERMAL = {
    # Target loop budget per sample (s).  The scheduler compares real cost
    # and downshifts the sample tier when we overshoot.
    "LOOP_BUDGET_S": 0.45,
    # Thermal zones to probe (Termux-typical paths).  Silently ignored when
    # unreadable (most devices restrict access without root).
    "THERMAL_ZONES": [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
    ],
    "TEMP_DIVISOR": 1000.0,          # most zones report millidegrees
    "THROTTLE_CELSIUS": 52.0,        # downshift above this
    "THROTTLE_RECOVER_CELSIUS": 47.0,
    "DOWNGRADE_COOLDOWN_S": 20.0,    # don't flip tiers faster than this
}

# ---------------------------------------------------------------------------
# SSE / persistence
# ---------------------------------------------------------------------------
OUTPUT = {
    "SSE_MAX_RATE_HZ": 2.0,          # never push to browsers faster than this
    "DB_FLUSH_INTERVAL_S": 5.0,      # batch inserts, one transaction
    "DB_MAX_ROWS": 20000,            # prune old telemetry
}

# ---------------------------------------------------------------------------
# Simulation mode (no hardware needed)
# ---------------------------------------------------------------------------
SIMULATION = {
    "ENABLED": False,                # or run: SIMULATION=1 python app.py
    "BREATH_RATE_BPM": 16.0,         # ~0.267 Hz
    # Operating envelope (validated by the test suite): reliable respiration
    # detection needs chest-motion modulation >= ~1 dB with link jitter
    # <= ~1 dB — i.e., a person 0.5–1.5 m from the phone on a strong AP.
    # Weaker links degrade gracefully toward presence-only detection.
    "BREATH_AMPLITUDE_DB": 1.0,
    "NOISE_DB": 1.0,                 # Android-like RSSI jitter (good link)
    "MOTION_EVERY_S": 45.0,          # inject a gross-motion burst
    "MOTION_AMPLITUDE_DB": 3.0,
    "BASE_RSSI": -62.0,
}

# Fusion target BSSIDs (uppercase, colon-separated) — leave empty for auto.
TARGET_WATCHLIST = []

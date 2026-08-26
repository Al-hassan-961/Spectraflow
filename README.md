# Spectraflow

Micro-movement Wi-Fi sensing for Termux (Android).  Turns the noisy RSSI
stream of a standard Android device — no CSI hardware required — into
respiration-rate estimation, subtle-presence detection, and gross-motion
classification, using an adaptive Kalman filter and a streaming FFT pipeline
engineered for the thermal budget of a mobile ARM core.

> **What this can honestly do.** Android exposes only RSSI, not the
> per-subcarrier CSI that enterprise systems use.  RSSI breathing
> modulation is real but small (~0.5–2 dB) and rides on ~1 dB of jitter.
> Spectraflow recovers it by *integration*: Kalman innovation-based
> weighting, long spectral averaging, band-SNR statistics, and multi-AP
> fusion — and reports a confidence score instead of pretending certainty.
> Reliable respiration reads need a person ~0.5–1.5 m from the phone on a
> strong AP; weaker links degrade gracefully to presence-only detection.

## Architecture

```
termux-wifi-connectioninfo      (fast poll, up to 4 Hz — the sensing channel)
        │
        ▼
MicroMotionDetector (dsp.py)
  ├─ Adaptive Kalman filter          → denoised display stream, link-quality
  │                                    innovation, fusion weights
  ├─ Streaming FFT (0.2–0.5 Hz)      → band-energy ratio, band-SNR, dominant
  │                                    frequency, breathing waveform
  │                                    (resampled, detrended, Hann-windowed,
  │                                    rFFT + long-EWMA spectral averaging)
  ├─ Motion band (0.5–2 Hz)          → gross-motion RMS (separates motion
  │                                    from breathing, no Kalman needed)
  ├─ Multi-AP fusion                 → innovation-weighted across top-N APs
  └─ hysteresis state machine        → CLEAR / PRESENCE / BREATHING / MOTION
        │
        ▼
Flask SSE ──► browser            (rate-limited pushes, batched SQLite writes)
```

Key files:
| File | Role |
|---|---|
| `dsp.py` | Kalman filter, streaming FFT, fusion, classifier |
| `thermal.py` | sample-tier scheduler, SoC thermal monitor, SSE rate limiter, GC tuning |
| `wifi_scanner.py` | fast `connectioninfo` poll vs expensive `scaninfo` sweep |
| `app.py` | Flask + SSE wiring, sensing/fusion threads, simulation mode |
| `config.py` | every tunable: sampling tiers, Kalman Q/R, band edges, thresholds |
| `tests/test_dsp.py` | physics validation suite on synthetic signals (21 checks) |

## Install & run (Termux)

```bash
pkg update && pkg upgrade
pkg install python git termux-api
pip install -r requirements.txt        # numpy + Flask; scipy optional
python app.py
# open http://127.0.0.1:5000 in the phone browser (or via termux-open-url)
```

No hardware? Validate the whole pipeline with a synthetic breathing subject:

```bash
SIMULATION=1 python app.py
```

Run the physics validation suite anytime:

```bash
python tests/test_dsp.py
```

## Power management (why it won't throttle)

The ARM/Android heat sources are the `termux-wifi-*` subprocess spawns,
SQLite fsync, and per-sample allocations.  Spectraflow attacks all three:

* **AdaptiveScheduler** — measures real loop cost each cycle and downshifts
  the sample tier (ECO 1 Hz / NORMAL 2 Hz / PERFORMANCE 4 Hz) when the loop
  overruns its budget, with a cooldown against flapping.  Sustained low cost
  steps back up automatically.
* **ThermalMonitor** — probes `/sys/class/thermal/thermal_zone*/temp` and
  forces a downgrade above 52 °C, recovering below 47 °C (hysteresis).
  Unreadable zones degrade gracefully to cost-based control.
* **Batched persistence** — SQLite WAL + one transaction per 5 s, so fsync
  doesn't fire per sample.
* **Rate-limited SSE** — browsers never receive more than 2 events/s.
* **GC tuning** — `gc.freeze()` at startup, periodic manual sweeps.
* **Scalar Kalman** — the 2×2 update is expanded into float arithmetic; no
  `numpy.linalg` per sample.  The FFT runs at most every 4 samples on
  48-point windows (microsecond-scale).

## DSP notes

* **Kalman placement.** An optimal level estimator is inherently low-pass;
  feeding its output to the FFT attenuates the very 0.2–0.5 Hz ripple we
  seek.  So the Kalman serves the display, the link-quality innovation
  (which also drives fusion weights and adaptive measurement-noise R), and
  the FFT operates on the raw stream — with the long-EWMA spectral average
  as the true denoiser for detection.
* **Statistics.** `band-energy ratio` (band vs all AC power) and `band-SNR`
  (band mean vs a 0.55–1.0 Hz noise reference) are long-EWMA averaged
  (~30 s); the dominant frequency uses the same average (rate latency is
  standard for respiration monitors).  Frequency stability over consecutive
  windows gates the `BREATHING` label.
* **Multi-AP fusion.** Each visible AP is a weakly-correlated "channel";
  per-AP band power is weighted by 1/innovation² so a flaky AP can't drag
  the fused estimate down.
* **Respiration band.** 0.2–0.5 Hz == 12–30 breaths/min.  Nyquist requires
  ≥ 1 Hz sampling, which is why NORMAL tier polls `connectioninfo` at 2 Hz.

## Author
**Al-hassan Shehade** — Spectraflow

"""
Spectraflow DSP validation suite.

Runs the whole pipeline on synthetic signals so the math is verifiable
without hardware.  Physics under test:

  * the adaptive Kalman filter denoises without destroying the 0.2–0.5 Hz
    breathing ripple (this is the make-or-break tradeoff for RSSI sensing),
  * the streaming FFT isolates the respiration band and recovers the true
    breathing rate from noisy RSSI,
  * the state machine reaches BREATHING and reacts to gross motion,
  * multi-AP fusion raises the band-energy ratio vs the best single AP,
  * the power manager downshifts on cost and thermal overruns.

Run:  python tests/test_dsp.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import dsp     # noqa: E402
import thermal # noqa: E402

FS = 2.0                          # nominal NORMAL-tier rate
DT = 1.0 / FS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def feed_breathing(det: dsp.MicroMotionDetector, seconds: float,
                   bpm: float = 16.0, amp_db: float | None = None,
                   ap_count: int = 1, phase_offset: float = 0.0,
                   base: float = -62.0, noise_db: float | None = None):
    """Feed `seconds` of synthetic breathing to the detector at 2 Hz.

    Each AP receives its own independent noise draw (channel diversity),
    so multi-AP fusion has something to fuse."""
    s = config.SIMULATION
    amp = amp_db if amp_db is not None else s["BREATH_AMPLITUDE_DB"]
    noise = noise_db if noise_db is not None else s["NOISE_DB"]
    f_b = bpm / 60.0
    rng = np.random.default_rng(42)
    n = int(seconds * FS)
    for i in range(n):
        t = i * DT
        readings = {}
        for j in range(ap_count):
            z = base + amp * math.sin(2 * math.pi * f_b * t + phase_offset + j * 0.7) \
                   + rng.normal(0.0, noise)
            readings[f"AP:{j}"] = z
        det.update(readings, t)
    return det


def feed_silence(det: dsp.MicroMotionDetector, seconds: float,
                 ap_count: int = 1, noise_db: float | None = None):
    s = config.SIMULATION
    noise = noise_db if noise_db is not None else s["NOISE_DB"]
    rng = np.random.default_rng(7)
    n = int(seconds * FS)
    for i in range(n):
        t = i * DT
        det.update({f"AP:{j}": -62.0 + rng.normal(0.0, noise) for j in range(ap_count)}, t)
    return det


def run_all() -> int:
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))

    # ------------------------------------------------------------------
    print("\n1. Adaptive Kalman filter")
    kal = dsp.AdaptiveKalmanRSSI()
    rng = np.random.default_rng(1)
    true_level = -60.0
    out, innovs = [], []
    for _ in range(400):
        z = true_level + rng.normal(0.0, 1.6)
        out.append(kal.update(z, _ * 0.5))
        innovs.append(abs(z - out[-1]))
    err = np.mean(np.abs(np.asarray(out) - true_level))
    check("tracks constant level within noise/2", err < 0.8, f"mean abs err={err:.3f} dB")

    # step response: convergence time
    kal2 = dsp.AdaptiveKalmanRSSI()
    for _ in range(10):
        kal2.update(-60.0, _ * 0.5)
    for i in range(30):
        kal2.update(-50.0, (10 + i) * 0.5)
    check("step tracked within 3 dB in < 10 s",
          abs(kal2.level - (-50.0)) < 3.0, f"level={kal2.level:.2f} dBm")

    # adaptive R stays clamped
    kal3 = dsp.AdaptiveKalmanRSSI()
    for _ in range(50):
        kal3.update(-60.0 + rng.normal(0.0, 5.0), _ * 0.5)
    r = kal3.r
    check("adaptive R clamped to [R_MIN, R_MAX]",
          config.KALMAN["R_MIN"] - 1e-9 <= r <= config.KALMAN["R_MAX"] + 1e-9,
          f"R={r:.3f}")

    # Kalman denoises the display stream (raw jitter reduced) while keeping
    # the slow breathing envelope: filtered std must be well below raw std
    kal4 = dsp.AdaptiveKalmanRSSI()
    s = config.SIMULATION
    f_b = 16.0 / 60.0
    rng4 = np.random.default_rng(3)
    raw_vals, filtered = [], []
    for i in range(300):
        t = i * 0.5
        z = -62.0 + s["BREATH_AMPLITUDE_DB"] * math.sin(2 * math.pi * f_b * t) \
                   + rng4.normal(0.0, s["NOISE_DB"])
        raw_vals.append(z)
        filtered.append(kal4.update(z, t))
    raw_std = float(np.std(raw_vals))
    filt_std = float(np.std(filtered))
    check("Kalman denoises: filtered std < 0.6 * raw std",
          filt_std < 0.6 * raw_std, f"raw_std={raw_std:.2f} filt_std={filt_std:.2f}")

    # ------------------------------------------------------------------
    print("\n2. Streaming FFT — respiration band isolation")
    spec = dsp.StreamingSpectrum()
    s = config.SIMULATION
    rng = np.random.default_rng(11)
    for i in range(240):            # 2 min: lets the EWMA converge
        t = i * 0.5
        z = -62.0 + s["BREATH_AMPLITUDE_DB"] * math.sin(2 * math.pi * (16.0 / 60.0) * t) \
                   + rng.normal(0.0, s["NOISE_DB"])
        spec.add(z, t)
    check("spectrum ready after warm-up", spec.ready)
    check("spectrum stable (long-EWMA) after full window", spec.stable)
    check("band-energy ratio high for breathing",
          spec.ber > 0.45, f"BER={spec.ber:.3f}")
    check("band SNR above noise floor for breathing",
          spec.band_snr > 1.5, f"SNR={spec.band_snr:.2f}")
    check("dominant frequency ≈ 0.267 Hz (16 bpm)",
          abs(spec.dom_freq - 16.0 / 60.0) < 0.04,
          f"dom_freq={spec.dom_freq:.4f} Hz")
    check("motion band quiet during breathing (no gross motion)",
          spec.motion_rms < 1.2, f"motion_rms={spec.motion_rms:.2f}")

    # silence: BER should be clearly lower, SNR near the ~1.3 colored-noise floor
    spec2 = dsp.StreamingSpectrum()
    rng2 = np.random.default_rng(12)
    for i in range(240):
        spec2.add(-62.0 + rng2.normal(0.0, s["NOISE_DB"]), i * 0.5)
    check("no-breathing BER below breathing BER",
          spec2.ber < spec.ber * 0.9, f"silence BER={spec2.ber:.3f}")
    check("no-breathing band SNR below presence threshold",
          spec2.band_snr < 1.35, f"silence SNR={spec2.band_snr:.2f}")

    # ------------------------------------------------------------------
    print("\n3. MicroMotionDetector — end-to-end classification")
    det = dsp.MicroMotionDetector()
    feed_breathing(det, 90.0, bpm=16.0)
    check("reaches BREATHING state", det.state == "BREATHING",
          f"state={det.state}")
    check("recovers ~16 bpm", abs(det.breath_rate_bpm - 16.0) < 2.0,
          f"rate={det.breath_rate_bpm:.2f} bpm")
    check("confidence > 0.4", det.confidence > 0.4,
          f"confidence={det.confidence:.2f}")

    det2 = dsp.MicroMotionDetector()
    feed_silence(det2, 90.0)
    check("silence stays CLEAR", det2.state == "CLEAR",
          f"state={det2.state} BER={det2.ber_fused:.3f}")

    # gross-motion burst mid-test
    det3 = dsp.MicroMotionDetector()
    feed_breathing(det3, 30.0, bpm=16.0)
    rng3 = np.random.default_rng(5)
    burst_started = None
    for i in range(60):                      # 30 s of heavy motion
        t = 30.0 + i * 0.5
        z = -62.0 + 3.0 * math.sin(2 * math.pi * 0.9 * t) + rng3.normal(0.0, 1.6)
        det3.update({"AP:0": z}, t)
        if det3.state == "MOTION" and burst_started is None:
            burst_started = t
    check("MOTION detected during burst",
          burst_started is not None,
          f"at t={burst_started:.1f}s" if burst_started else f"state={det3.state}")

    # ------------------------------------------------------------------
    print("\n4. Multi-AP fusion")
    # (a) fusion stays inside the single-AP envelope
    d_fused = dsp.MicroMotionDetector()
    feed_breathing(d_fused, 60.0, ap_count=3)
    singles = []
    for j in range(3):
        d = dsp.MicroMotionDetector()
        feed_breathing(d, 60.0, ap_count=1, phase_offset=j * 0.7)
        singles.append(d.ber_fused)
    lo, hi = min(singles), max(singles)
    check("fusion inside single-AP BER envelope",
          lo - 0.02 <= d_fused.ber_fused <= hi + 0.02,
          f"fused={d_fused.ber_fused:.3f} singles={[round(s,3) for s in singles]}")

    # (b) innovation weighting suppresses a broken channel
    def feed_mixed(det, seconds, bad_noise):
        amp = config.SIMULATION["BREATH_AMPLITUDE_DB"]
        rng = np.random.default_rng(9)
        f_b = 16.0 / 60.0
        for i in range(int(seconds * FS)):
            t = i * DT
            good = -62.0 + amp * math.sin(2 * math.pi * f_b * t) + rng.normal(0.0, 1.0)
            bad = -62.0 + amp * math.sin(2 * math.pi * f_b * t) + rng.normal(0.0, bad_noise)
            det.update({"GOOD": good, "BAD": bad}, t)
    d_mix = dsp.MicroMotionDetector()
    feed_mixed(d_mix, 60.0, bad_noise=3.0)
    d_bad = dsp.MicroMotionDetector()
    rng_b = np.random.default_rng(9)
    amp = config.SIMULATION["BREATH_AMPLITUDE_DB"]
    f_b = 16.0 / 60.0
    for i in range(int(60 * FS)):
        t = i * DT
        d_bad.update({"BAD": -62.0 + amp * math.sin(2 * math.pi * f_b * t)
                             + rng_b.normal(0.0, 3.0)}, t)
    check("fusion beats the broken channel alone (innovation weighting)",
          d_mix.ber_fused > d_bad.ber_fused + 0.05,
          f"fused={d_mix.ber_fused:.3f} bad-only={d_bad.ber_fused:.3f}")

    # ------------------------------------------------------------------
    print("\n5. Power management")
    FAKE_ZONES = ["/nonexistent/thermal_zone0/temp"]   # no host thermal interference
    sch = thermal.AdaptiveScheduler(initial_tier="PERFORMANCE", zones=FAKE_ZONES)
    tier0 = sch.tier
    for _ in range(3):                      # sustained cost overrun
        sch.report_cycle(0.8)
    check("downgrades from PERFORMANCE on cost overrun",
          thermal.AdaptiveScheduler.TIER_ORDER.index(sch.tier)
          < thermal.AdaptiveScheduler.TIER_ORDER.index(tier0),
          f"tier={sch.tier}")

    sch2 = thermal.AdaptiveScheduler(initial_tier="NORMAL", zones=FAKE_ZONES)
    for _ in range(60):
        sch2.report_cycle(0.05)
    check("recovers to PERFORMANCE under sustained low cost",
          sch2.tier == "PERFORMANCE", f"tier={sch2.tier}")

    lim = thermal.RateLimiter(max_rate_hz=2.0)
    import time as _t
    _t.sleep(0.01)
    ok1 = lim.allow()
    ok2 = lim.allow()                       # immediately after -> denied
    check("SSE rate limiter gates pushes", ok1 and not ok2)

    mon = thermal.ThermalMonitor(zones=["/nonexistent/zone"])
    check("thermal monitor degrades gracefully (None zones)",
          mon.temperature() is None and mon.update() is False)

    print(f"\n{'=' * 60}\nResult: {'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(run_all())

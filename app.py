"""
Spectraflow — micro-movement Wi-Fi sensing for Termux.

Architecture
------------
    termux-wifi-connectioninfo (fast, up to 4 Hz)
              │
              ▼
    MicroMotionDetector (dsp.py)
      ├─ Adaptive Kalman filter          → denoised RSSI stream
      ├─ Streaming FFT (0.2–0.5 Hz band) → respiration band energy,
      │                                     dominant frequency, breath wave
      └─ hysteresis state machine        → CLEAR / PRESENCE / BREATHING / MOTION
              │
              ▼
    Flask SSE  ──►  browser (rate-limited, batched DB writes)

Multi-AP fusion runs in a background thread (scaninfo is ~1–3 s per sweep);
the sample-tier scheduler and thermal monitor (thermal.py) keep the ARM core
inside its thermal envelope.  No hardware?  Run with SIMULATION=1 to feed a
synthetic breathing signal through the full pipeline.

Run:  python app.py            (or SIMULATION=1 python app.py)
"""
from __future__ import annotations

import json
import queue
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

import config
import database as db
import dsp
import thermal
import wifi_scanner as scanner

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
detector = dsp.MicroMotionDetector()
scheduler = thermal.AdaptiveScheduler()
sse_limiter = thermal.RateLimiter()
scan_buffer = db.ScanBuffer()


def _sim_flag() -> bool:
    import os
    return os.environ.get("SIMULATION", "").strip() in ("1", "true", "yes")


SIMULATION = config.SIMULATION["ENABLED"] or _sim_flag()

TARGET_WATCHLIST = set(b.upper() for b in config.TARGET_WATCHLIST)

# fusion (background sweep) state
_fusion_readings: dict[str, float] = {}
_fusion_lock = threading.Lock()
_fusion_aps: list[dict] = []

# telemetry for /api/health
_health = {
    "started": time.time(),
    "samples": 0,
    "last_loop_cost": 0.0,
    "tier": scheduler.tier,
    "thermal_c": None,
    "throttling": False,
    "sse_pushes": 0,
}

# ---------------------------------------------------------------------------
# SSE pub/sub (unchanged pattern: per-client bounded queue, drop-on-full)
# ---------------------------------------------------------------------------
clients: set[queue.Queue] = set()
clients_lock = threading.Lock()


def notify_clients(data: dict) -> None:
    with clients_lock:
        for client_queue in list(clients):
            try:
                client_queue.put_nowait(data)
            except queue.Full:
                pass


def get_vendor(mac_address: str | None) -> str:
    if not mac_address:
        return "Encrypted / Unknown"
    oui = mac_address.replace(":", "").upper()[:6]
    vendors = {
        "FCF8AE": "Samsung", "001A11": "Google", "DCA904": "Apple",
        "001422": "Dell", "000142": "Cisco", "CC46D6": "Huawei",
    }
    return vendors.get(oui, "Generic Hardware")


# ---------------------------------------------------------------------------
# Acquisition: fast active-AP poll + cached fusion readings
# ---------------------------------------------------------------------------
def acquire() -> tuple[dict[str, float], dict | None]:
    """Returns ({bssid: rssi}, active_ap_meta).  Never blocks on fusion."""
    readings: dict[str, float] = {}
    active = None
    if not SIMULATION:
        active = scanner.poll_active_ap()
        if active:
            readings[active["bssid"]] = active["rssi"]
        with _fusion_lock:
            for bssid, rssi in _fusion_readings.items():
                readings.setdefault(bssid, rssi)     # don't clobber the live read
    else:
        t = time.time()
        phase = getattr(acquire, "_phase", 0.0)
        acquire._phase = phase + 0.01
        sim_ap = {
            "ssid": "SIMULATED_AP",
            "bssid": "AA:BB:CC:00:11:22",
            "rssi": dsp.generate_synthetic_sample(t, phase),
            "frequency_mhz": 2412,
            "timestamp": t,
        }
        readings[sim_ap["bssid"]] = sim_ap["rssi"]
        active = sim_ap
    return readings, active


def fusion_loop() -> None:
    """Background sweeper for multi-AP fusion.  Backs off when a sweep is
    slow or when the thermal monitor reports throttling."""
    global _fusion_readings, _fusion_aps
    interval = config.RADAR_SCAN_INTERVAL
    while True:
        try:
            if not SIMULATION:
                t0 = time.time()
                aps = scanner.scan_top_aps(config.FUSION["MAX_APS"])
                cost = time.time() - t0
                with _fusion_lock:
                    _fusion_readings = {ap["bssid"]: ap["rssi"] for ap in aps}
                    _fusion_aps = aps
                # adaptive backoff: a slow sweep means the radio/CPU is busy
                if cost > 2.5:
                    interval = min(15.0, interval * 2)
                else:
                    interval = config.RADAR_SCAN_INTERVAL
                if scheduler.metrics["throttling"]:
                    time.sleep(max(interval, 8.0))
                else:
                    time.sleep(interval)
            else:
                time.sleep(config.RADAR_SCAN_INTERVAL)
        except Exception:
            time.sleep(interval)


# ---------------------------------------------------------------------------
# Sensing loop: acquire → DSP → SSE → persist, gated by the scheduler
# ---------------------------------------------------------------------------
def sensing_loop() -> None:
    global _health
    sweep_gc = thermal.tune_gc()
    last_gc = time.time()

    while True:
        t0 = time.monotonic()
        try:
            readings, active = acquire()
            now = time.time()
            detector.update(readings, now)

            # persist the active AP's raw telemetry (batched)
            if active:
                chain = detector.chains.get(active["bssid"])
                filtered = chain["kalman"].level if chain else active["rssi"]
                scan_buffer.add(
                    now, active["bssid"], active["ssid"], active["rssi"],
                    active["frequency_mhz"],
                    scanner.estimate_distance(filtered),
                )
            scan_buffer.maybe_flush(now)

            _health["samples"] += 1
            _health["tier"] = scheduler.tier
            _health["thermal_c"] = scheduler.metrics["thermal_c"]
            _health["throttling"] = scheduler.metrics["throttling"]

            if sse_limiter.allow():
                _health["sse_pushes"] += 1
                notify_clients(_build_event(active, now))

            if now - last_gc > 60.0:
                last_gc = now
                sweep_gc()
        except Exception:
            # never kill the loop; a single bad cycle must not stop sensing
            pass

        cost = time.monotonic() - t0
        _health["last_loop_cost"] = cost
        tier = scheduler.report_cycle(cost)
        interval = 1.0 / config.SAMPLE_RATE_TIERS[tier]
        time.sleep(max(0.0, interval - cost))


def _build_event(active: dict | None, now: float) -> dict:
    state = detector.state
    human = state != detector.STATE_CLEAR

    best_bssid = detector._best_ap() or (active or {}).get("bssid")
    times, filtered = ([], [])
    if best_bssid:
        times, filtered = detector.filtered_series(best_bssid, maxlen=64)
        # align lengths for the chart
        filtered = filtered[-len(times):] if len(filtered) > len(times) else filtered

    freqs, mags, breath_wave = detector.spectrum(best_bssid)

    target_alert = bool(best_bssid and best_bssid.upper() in TARGET_WATCHLIST)

    return {
        "timestamp": now,
        "status_label": ("TARGET ALERT!" if target_alert else state),
        "human_detected": human or target_alert,
        "target_alert": target_alert,
        # micro-movement metrics
        "respiration_rate_bpm": round(detector.breath_rate_bpm, 2),
        "respiration_confidence": round(detector.confidence, 3),
        "band_energy_ratio": round(detector.ber_fused, 4),
        "band_snr": round(detector.band_snr_fused, 3),
        "motion_rms": round(detector.motion_rms, 3),
        "freq_stability": round(detector.freq_std, 4),
        "spectrum_freqs": [round(float(f), 4) for f in freqs[:33]],
        "spectrum_mags": [round(float(m), 4) for m in mags[:33]],
        "breath_wave": [round(float(v), 4) for v in breath_wave[-64:]],
        # classic telemetry (compat)
        "ap_used": best_bssid,
        "vendor_info": get_vendor(best_bssid),
        "angle": int(best_bssid.replace(":", ""), 16) % 360 if best_bssid else 0,
        "distance": round(scanner.estimate_distance(filtered[-1]) if filtered else 0.0, 2),
        "history": {best_bssid: filtered} if best_bssid else {},
        "timestamps": [round(t, 2) for t in times],
        # power state
        "tier": scheduler.tier,
        "throttling": scheduler.metrics["throttling"],
        "thermal_c": scheduler.metrics["thermal_c"],
    }


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------
threading.Thread(target=sensing_loop, daemon=True).start()
threading.Thread(target=fusion_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan")
def scan():
    aps = scanner.scan_wifi()
    data = []
    for ap in aps:
        bssid = ap.get("bssid", "00:00:00:00:00:00")
        data.append({
            "ssid": ap.get("ssid", "Hidden"),
            "bssid": bssid,
            "vendor": get_vendor(bssid),
            "rssi": ap.get("rssi", -1000),
            "distance": scanner.estimate_distance(ap.get("rssi")),
            "frequency": ap.get("frequency_mhz", 0),
            "angle": int(bssid.replace(":", ""), 16) % 360 if bssid else 0,
        })
    return jsonify(data)


@app.route("/api/history")
def history():
    limit = request.args.get("limit", 20, type=int)
    rows = db.get_history(limit)
    return jsonify([
        {"timestamp": r[0], "bssid": r[1], "ssid": r[2], "rssi": r[3],
         "frequency": r[4], "distance": r[5]}
        for r in rows
    ])


@app.route("/api/health")
def health():
    return jsonify({**_health, "uptime_s": round(time.time() - _health["started"], 1)})


@app.route("/api/settings", methods=["POST"])
def settings():
    req = request.json or {}
    if "tier" in req:
        tier = str(req["tier"]).upper()
        if tier in config.SAMPLE_RATE_TIERS:
            scheduler.tier = tier
            scheduler._last_change = 0.0
            return jsonify({"status": "success", "tier": tier,
                            "rate_hz": config.SAMPLE_RATE_TIERS[tier]})
    if "sensitivity" in req:   # kept as a compatibility no-op knob
        return jsonify({"status": "success", "sensitivity": float(req["sensitivity"])})
    return jsonify({"status": "error"}), 400


@app.route("/stream")
def stream():
    def generate():
        client_queue: queue.Queue = queue.Queue(maxsize=10)
        with clients_lock:
            clients.add(client_queue)
        try:
            while True:
                try:
                    data = client_queue.get(timeout=2.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            with clients_lock:
                clients.remove(client_queue)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    db.init_db()
    print("[Spectraflow] engine online"
          + ("  [SIMULATION MODE]" if SIMULATION else "")
          + f"  tier={scheduler.tier} rate={scheduler.rate_hz:.1f} Hz")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

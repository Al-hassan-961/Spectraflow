"""
Spectraflow — Termux Wi-Fi acquisition layer.

Android gives us two APIs, with very different costs:

  * termux-wifi-connectioninfo — instantaneous, ~5–30 ms per call.
    Returns the *connected* AP's live RSSI.  This is the sensing channel:
    it can be polled at 2–4 Hz, which is what the respiration band needs
    (Nyquist >= 1 Hz for 0.2–0.5 Hz content).

  * termux-wifi-scaninfo — a full spectrum sweep, ~1–3 s per call.
    Expensive.  Used only for the radar view and for periodic multi-AP
    fusion refreshes (bounded by RADAR_SCAN_INTERVAL).

Every call spawns a subprocess; that process spawn + JSON parse dominates
CPU on ARM, so the DSP layer (thermal.py) gates how often we call these.
"""
from __future__ import annotations

import json
import subprocess
import time

import config


def _run_json(argv: list[str], timeout: float = 2.0) -> dict | list | None:
    """Run a termux API command and parse JSON.  Never raises."""
    try:
        res = subprocess.run(argv, capture_output=True, text=True,
                             timeout=timeout, check=True)
        if not res.stdout.strip():
            return None
        return json.loads(res.stdout)
    except Exception:
        return None


def poll_active_ap() -> dict | None:
    """
    Fast live read of the connected AP (the sensing channel).

    Returns {ssid, bssid, rssi, frequency_mhz, timestamp} or None when not
    connected.  Designed to be called at up to PERFORMANCE-tier rates.
    """
    data = _run_json(["termux-wifi-connectioninfo"])
    if not isinstance(data, dict) or not data.get("bssid"):
        return None
    return {
        "ssid": data.get("ssid") or "Secured_AP",
        "bssid": data["bssid"],
        "rssi": float(data.get("rssi", -100.0)),
        "frequency_mhz": int(data.get("frequency", 0) or 0),
        "timestamp": time.time(),
    }


def scan_wifi() -> list[dict]:
    """Full spectrum sweep (expensive).  Returns list of AP dicts."""
    networks = []
    conn = poll_active_ap()
    if conn:
        networks.append(conn)

    data = _run_json(["termux-wifi-scaninfo"], timeout=4.0)
    if isinstance(data, list):
        known = {net["bssid"] for net in networks}
        now = time.time()
        for ap in data:
            bssid = ap.get("bssid")
            if bssid and bssid not in known:
                networks.append({
                    "ssid": ap.get("ssid", "Hidden") or "Hidden",
                    "bssid": bssid,
                    "rssi": float(ap.get("rssi", -100.0)),
                    "frequency_mhz": int(ap.get("frequency", 0) or 0),
                    "timestamp": now,
                })
                known.add(bssid)
    return networks


def scan_top_aps(n: int = 3) -> list[dict]:
    """Full sweep, returning the n strongest APs (for multi-AP fusion).
    Stronger APs have cleaner variance and dominate the fused estimate."""
    aps = scan_wifi()
    aps.sort(key=lambda a: a.get("rssi", -1000.0), reverse=True)
    return aps[:n]


def estimate_distance(rssi: float | None) -> float:
    """Log-distance path-loss model with closed-room coefficients."""
    if rssi is None or rssi == 0:
        return 0.0
    if rssi >= -35:
        return 0.5
    A = -45.0   # RSSI at 1 m in this environment
    n = 2.75    # path-loss exponent (indoor, NLOS-ish)
    try:
        dist = 10 ** ((abs(rssi) - abs(A)) / (10 * n))
        return round(min(dist, 20.0), 2)
    except Exception:
        return 0.0

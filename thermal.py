"""
Spectraflow power & thermal management.

On an Android ARM SoC (especially passive-cooled phones) the biggest heat
sources are: the termux-wifi-* subprocess spawns, SQLite fsync, and any
per-sample numpy allocation.  This module adds the three levers that keep
the loop inside its thermal envelope:

1. AdaptiveScheduler — measures the real cost of each acquisition cycle and
   downshifts the sample-rate tier (ECO / NORMAL / PERFORMANCE) whenever the
   loop overruns its budget, with a cooldown so tiers don't flap.
2. ThermalMonitor — probes /sys/class/thermal/thermal_zone*/temp and forces
   a downgrade above a threshold, recovering below a lower one (hysteresis).
   Unreadable zones (non-rooted devices) degrade gracefully to cost-based
   control only.
3. RateLimiter — caps how often the SSE layer may push to browsers, so a
   fast producer can never fan out faster than the UI can consume.
"""
from __future__ import annotations

import gc
import time

import config


# ---------------------------------------------------------------------------
# Thermal zone probing
# ---------------------------------------------------------------------------
class ThermalMonitor:
    """Reads SoC temperature from common Android sysfs paths (rootless
    Termux often cannot read these; None means 'unknown, skip control')."""

    def __init__(self, zones=None, throttle_c=None, recover_c=None):
        th = config.THERMAL
        self.zones = zones or th["THERMAL_ZONES"]
        self.throttle_c = throttle_c if throttle_c is not None else th["THROTTLE_CELSIUS"]
        self.recover_c = recover_c if recover_c is not None else th["THROTTLE_RECOVER_CELSIUS"]
        self.throttled = False

    def temperature(self) -> float | None:
        for path in self.zones:
            try:
                with open(path, "r") as fh:
                    raw = fh.read().strip()
                val = float(raw) / config.THERMAL["TEMP_DIVISOR"]
                if 0 < val < 130:          # sanity range
                    return val
            except Exception:
                continue
        return None

    def update(self) -> bool:
        """Returns True when the system is currently throttling."""
        t = self.temperature()
        if t is None:
            return self.throttled
        if t >= self.throttle_c:
            self.throttled = True
        elif t <= self.recover_c:
            self.throttled = False
        return self.throttled


# ---------------------------------------------------------------------------
# Cost-driven sample-rate control
# ---------------------------------------------------------------------------
class AdaptiveScheduler:
    """
    Chooses the sample-rate tier from measured loop cost + thermal state.

    Every cycle the caller reports how long the acquisition+processing took
    (cost_s).  If cost exceeds the budget, we drop one tier (with a cooldown
    so a single slow cycle can't demote us); if we are far under budget for a
    sustained stretch, we may step back up.  Thermal throttling overrides.
    """

    TIER_ORDER = ["ECO", "NORMAL", "PERFORMANCE"]

    def __init__(self, initial_tier: str | None = None, zones: list[str] | None = None):
        th = config.THERMAL
        self.initial_tier = (initial_tier or config.ACTIVE_TIER).upper()
        if self.initial_tier not in self.TIER_ORDER:
            self.initial_tier = "NORMAL"
        self.tier = self.initial_tier
        self.thermal = ThermalMonitor(zones=zones)   # zones=None → config defaults
        self._last_change = 0.0
        self._fast_cycles = 0
        self.cooldown = th["DOWNGRADE_COOLDOWN_S"]
        self.budget = th["LOOP_BUDGET_S"]
        self.metrics = {"loop_cost_s": 0.0, "thermal_c": None, "throttling": False}

    @property
    def rate_hz(self) -> float:
        return config.SAMPLE_RATE_TIERS[self.tier]

    def report_cycle(self, cost_s: float) -> str:
        """Feed the measured cost of the last cycle; returns active tier."""
        self.metrics["loop_cost_s"] = cost_s
        self.metrics["thermal_c"] = self.thermal.temperature()
        self.metrics["throttling"] = self.thermal.update()

        now = time.time()
        if now - self._last_change < self.cooldown:
            return self.tier

        downgrade = False
        if self.metrics["throttling"]:
            downgrade = True                                    # thermal override
        elif cost_s > self.budget and self.tier != "ECO":
            downgrade = True                                    # cost override

        if downgrade:
            idx = self.TIER_ORDER.index(self.tier)
            if idx > 0:
                self.tier = self.TIER_ORDER[idx - 1]
                self._last_change = now
                self._fast_cycles = 0
        else:
            # step back up only after a sustained run well under budget
            if cost_s < self.budget * 0.55:
                self._fast_cycles += 1
                if (self._fast_cycles >= 30 and self.tier != "PERFORMANCE"
                        and not self.metrics["throttling"]):
                    idx = self.TIER_ORDER.index(self.tier)
                    self.tier = self.TIER_ORDER[idx + 1]
                    self._last_change = now
                    self._fast_cycles = 0
            else:
                self._fast_cycles = 0
        return self.tier


# ---------------------------------------------------------------------------
# SSE push limiter
# ---------------------------------------------------------------------------
class RateLimiter:
    """Simple interval gate: allow an event only if dt >= min_interval."""

    def __init__(self, max_rate_hz: float | None = None):
        self.min_interval = 1.0 / (max_rate_hz or config.OUTPUT["SSE_MAX_RATE_HZ"])
        self._last = 0.0

    def allow(self) -> bool:
        now = time.time()
        if now - self._last >= self.min_interval:
            self._last = now
            return True
        return False

    def reset(self) -> None:
        self._last = 0.0


# ---------------------------------------------------------------------------
# GC tuning for the hot loop
# ---------------------------------------------------------------------------
def tune_gc() -> None:
    """
    Freeze objects allocated at import time so the generational collector
    never scans them during the sensing loop, and disable collection in the
    hot path.  Our hot structures (deques, small ndarrays) are acyclic, so
    periodic manual collection keeps memory flat without GC pauses.
    """
    try:
        gc.freeze()
    except Exception:
        pass
    gc.disable()

    def _sweep() -> None:
        gc.collect()
        # re-arm after collect; disable() stays off between sweeps
        gc.disable()

    return _sweep

"""Phase sanitization: linear unwrapping and CFO/SFO removal.

Commodity Wi-Fi receivers do not preserve the absolute phase of the channel.
Two effects corrupt the per-subcarrier phase response linearly:

* **Carrier Frequency Offset (CFO)** -- the transmitter and receiver local
  oscillators differ slightly, adding a constant phase rotation that drifts
  over time;
* **Sampling Frequency Offset (SFO)** -- the sampling clocks differ, adding a
  phase term that grows *linearly with subcarrier index*.

Both are removed by fitting and subtracting the linear component:

    phase_sanitized[i] = phase[i]
                         - ((phase[N-1] - phase[0]) / (N - 1)) * i
                         - mean(phase)

Because the residual is what carries the physical displacement information,
this step is what makes phase-based sensing possible without a calibrated
receiver. It is also why a *linear* (not just modular) unwrap is required: a
wrapped phase track cannot be linearly detrended.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from spectraflow._native import HAVE_NATIVE, NATIVE
from spectraflow.dsp import _numpy_dsp

__all__ = ["PhaseSanitizer", "sanitize_phase", "unwrap_phase"]


def _as_phase_vector(values: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Coerce input to a float32 phase vector.

    Accepts either an already-real phase vector or complex channel estimates, in
    which case the phase is extracted with ``np.angle``. Complex input MUST be
    handled here rather than passed through a dtype cast: casting complex to
    float silently discards the imaginary part, yielding the real component of
    the channel instead of its phase -- a wrong answer that produces no error.
    """
    arr = np.asarray(values)
    if np.iscomplexobj(arr):
        return np.angle(arr).astype(np.float32).reshape(-1)
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)


def unwrap_phase(phase: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Unwrap phase so consecutive samples never jump by more than +/- pi."""
    arr = _as_phase_vector(phase)
    if arr.size == 0:
        return arr
    if HAVE_NATIVE:
        return np.asarray(NATIVE.unwrap_phase(arr), dtype=np.float32)
    return _numpy_dsp.unwrap_phase(arr)


def sanitize_phase(phase: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Remove the linear CFO/SFO ramp and the mean from a phase response.

    Accepts a real phase vector or complex CSI (whose phase is extracted first).

    See the module docstring for the exact formula. The defining property of
    the output is that its endpoint-to-endpoint slope is zero.
    """
    arr = _as_phase_vector(phase)
    if arr.size == 0:
        return arr
    if HAVE_NATIVE:
        return np.asarray(NATIVE.sanitize_phase(arr), dtype=np.float32)
    return _numpy_dsp.sanitize_phase(arr)


class PhaseSanitizer:
    """Stateless-per-frame phase extractor and sanitizer.

    Optionally unwraps across subcarriers before the linear detrend, which
    matters when the raw phase has wrapped (a chest displacement of more than
    half a wavelength produces a genuine wrap, and an unwrapped track is needed
    to keep the recovered displacement continuous).

    Example::

        sanitizer = PhaseSanitizer()
        phase = sanitizer.process(csi_frame.csi)
    """

    __slots__ = ("unwrap", "last_phase", "frames_processed")

    def __init__(self, unwrap: bool = True) -> None:
        self.unwrap = bool(unwrap)
        self.last_phase: npt.NDArray[np.float32] | None = None
        self.frames_processed = 0

    def process(self, csi: npt.ArrayLike) -> npt.NDArray[np.float32]:
        """Sanitized phase for one complex CSI observation.

        Args:
            csi: Complex channel estimates, shape ``(n_subcarriers,)``, or an
                already-real phase vector.

        Returns:
            Sanitized phase, shape ``(n_subcarriers,)``, ``float32``.
        """
        arr = np.asarray(csi)
        if np.iscomplexobj(arr):
            phase = np.angle(arr).astype(np.float32)
        else:
            phase = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)

        if phase.size == 0:
            self.last_phase = phase
            return phase

        if self.unwrap:
            phase = unwrap_phase(phase)
        clean = sanitize_phase(phase)

        self.last_phase = clean
        self.frames_processed += 1
        return clean

    __call__ = process

    def reset(self) -> None:
        self.last_phase = None
        self.frames_processed = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<PhaseSanitizer unwrap={self.unwrap} frames={self.frames_processed}>"


def backend() -> Any:
    """Name of the active backend, for diagnostics and tests."""
    return "native" if HAVE_NATIVE else "numpy"

"""Clutter rejection by rolling dynamic background cancellation.

The static multipath environment -- walls, furniture, the floor -- dominates the
raw CSI by one to two orders of magnitude, and it is *stationary*. The signal we
want, a chest moving by millimetres, is tiny but it *changes*. Subtracting a
trailing mean therefore removes almost all of the energy while leaving the
motion-induced perturbation intact:

    H_cleaned(f, t) = H(f, t) - mean(H(f, t - m))     for m in 1..M

Choosing M is the central trade-off: it must be long compared to one breath
(~4 s) so the breathing signal is not itself averaged away, yet short enough to
track environmental drift (a door moving, someone walking past). The default
M = 100 frames is 5 s at 20 Hz.

.. warning::

   This is the right tool for **amplitude**-domain work, and the wrong one to
   place in front of a **phase**-based estimator. The phase of a noisy complex
   sample carries an error proportional to ``1/|H|``; subtraction shrinks ``|H|``
   by an order of magnitude while leaving the absolute noise floor untouched, so
   ``arg(H)`` comes out far noisier than it went in -- measured at 56x worse for
   a 19x magnitude reduction. Applied before phase extraction it drops subject
   respiration SNR from ~18 dB to ~1 dB.

   :class:`~spectraflow.dsp.vitals_extractor.VitalSignsExtractor` therefore
   removes static clutter by detrending *within* its analysis window instead,
   which cancels the background without shrinking the magnitude used to form the
   phase. ``tests/test_dsp.py`` pins this.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt

from spectraflow._native import HAVE_NATIVE, NATIVE

__all__ = ["ClutterRemover"]


class ClutterRemover:
    """Static-clutter cancellation over a configurable rolling window.

    Uses the native implementation when available (``spectraflow_core``), which
    keeps a running sum and an explicit ring of frames; otherwise an equivalent
    NumPy implementation with the same semantics is used.

    The remover is *stateful and streaming*: call :meth:`process` once per frame
    in arrival order. The background estimate is built from the RAW frames --
    feeding back the cleaned output would drive the estimate to zero and destroy
    the cancellation.

    Example::

        remover = ClutterRemover(window=100)
        for frame in stream:
            cleaned = remover.process(frame.csi)
    """

    __slots__ = ("window", "_native", "_history", "_sum", "_count", "_write_idx")

    def __init__(self, window: int = 100, *, use_native: bool | None = None) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = int(window)
        self._native = (
            NATIVE.ClutterRemover(self.window)
            if (HAVE_NATIVE and (use_native is None or use_native))
            else None
        )
        self._history: deque[npt.NDArray[np.complex128]] = deque(maxlen=self.window)
        self._sum: npt.NDArray[np.complex128] | None = None
        self._count = 0
        self._write_idx = 0

    @property
    def count(self) -> int:
        """Number of frames currently in the averaging window."""
        if self._native is not None:
            return int(self._native.count)
        return self._count

    @property
    def warm(self) -> bool:
        """True once the window has filled and the estimate is meaningful."""
        if self._native is not None:
            return bool(self._native.warm)
        return self._count >= self.window

    def process(self, csi: npt.ArrayLike) -> npt.NDArray[np.complex64]:
        """Subtract the trailing mean from one CSI observation.

        Args:
            csi: Complex channel estimates, shape ``(n_subcarriers,)``.

        Returns:
            The clutter-cancelled CSI, same shape, ``complex64``.
        """
        arr = np.ascontiguousarray(csi, dtype=np.complex64).reshape(-1)
        if arr.size == 0:
            return arr

        if self._native is not None:
            return np.asarray(self._native.process(arr), dtype=np.complex64)

        if self._sum is None or self._sum.size != arr.size:
            # Subcarrier count changed (channel retune): old history is not
            # comparable, so restart rather than mix incompatible vectors.
            self._history.clear()
            self._sum = np.zeros(arr.size, dtype=np.complex128)
            self._count = 0

        original = arr.astype(np.complex128)
        if self._count > 0:
            mean = self._sum / self._count
            cleaned = (original - mean).astype(np.complex64)
        else:
            cleaned = arr.copy()

        if self._count == self.window:
            self._sum -= self._history[0]
        self._history.append(original)
        self._sum += original
        if self._count < self.window:
            self._count += 1

        return cleaned

    __call__ = process

    def reset(self) -> None:
        """Forget all history."""
        if self._native is not None:
            self._native.reset()
        self._history.clear()
        self._sum = None
        self._count = 0
        self._write_idx = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        backend = "native" if self._native is not None else "numpy"
        return (
            f"<ClutterRemover window={self.window} count={self.count} "
            f"backend={backend}>"
        )

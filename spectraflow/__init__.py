"""Spectraflow — Wi-Fi CSI 3D sensing, human pose estimation and contactless
vital-sign extraction.

The package is organised as a straight pipeline:

    ingestion -> dsp (sanitize, de-clutter, vitals) -> inference (pose)
                                                        |
                                              server (WebSocket) -> static/ WebGL UI

Every stage runs with nothing but NumPy installed. The optional native
``spectraflow_core`` extension (built from ``core/``) transparently accelerates
the hot DSP primitives, and ``onnxruntime`` enables real neural pose inference;
both are detected at import time and degrade to equivalent implementations
rather than failing.
"""

from __future__ import annotations

from spectraflow._native import HAVE_NATIVE, NATIVE
from spectraflow.config import SpectraflowConfig, default_config

__all__ = [
    "SpectraflowConfig",
    "default_config",
    "HAVE_NATIVE",
    "NATIVE",
    "__version__",
]

__version__ = "1.0.0"

"""Signal processing: phase sanitization, clutter rejection, vital signs.

The public surface is intentionally small -- three classes and the supporting
spectral helpers. Every primitive dispatches to the native C++ core when it is
built and to an equivalent NumPy implementation otherwise.
"""

from __future__ import annotations

from spectraflow.dsp._backend import BACKEND, SpectralPeak
from spectraflow.dsp._numpy_dsp import (
    Biquad,
    bin_width_hz,
    design_butterworth_bandpass,
    find_band_peak,
    hann_window,
    magnitude_spectrum,
    snr_to_confidence,
)
from spectraflow.dsp.clutter_removal import ClutterRemover
from spectraflow.dsp.phase_sanitizer import PhaseSanitizer, sanitize_phase, unwrap_phase
from spectraflow.dsp.vitals_extractor import VitalSignsExtractor, VitalsResult

__all__ = [
    "BACKEND",
    "Biquad",
    "ClutterRemover",
    "PhaseSanitizer",
    "SpectralPeak",
    "VitalSignsExtractor",
    "VitalsResult",
    "bin_width_hz",
    "design_butterworth_bandpass",
    "find_band_peak",
    "hann_window",
    "magnitude_spectrum",
    "sanitize_phase",
    "snr_to_confidence",
    "unwrap_phase",
]

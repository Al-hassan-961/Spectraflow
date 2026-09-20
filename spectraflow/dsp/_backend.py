"""Backend dispatch for the DSP primitives.

Chooses the native ``spectraflow_core`` implementation when it is importable and
the NumPy reference otherwise, normalising both to the same Python types. Callers
never branch on the backend themselves.

The native code is only ever an accelerator: every function here has a NumPy
path with the same semantics, and ``tests/test_dsp.py`` asserts that the two
backends agree numerically.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spectraflow._native import HAVE_NATIVE, NATIVE
from spectraflow.dsp import _numpy_dsp
from spectraflow.dsp._numpy_dsp import SpectralPeak

__all__ = [
    "BACKEND",
    "SpectralPeak",
    "bin_width_hz",
    "design_bandpass",
    "find_band_peak",
    "magnitude_spectrum",
    "snr_to_confidence",
]

BACKEND = "native" if HAVE_NATIVE else "numpy"


def design_bandpass(
    f_lo: float, f_hi: float, fs: float
) -> tuple[float, float, float, float, float]:
    """Butterworth band-pass coefficients as ``(b0, b1, b2, a1, a2)``."""
    if HAVE_NATIVE:
        c = NATIVE.design_butterworth_bandpass(float(f_lo), float(f_hi), float(fs))
        return (c.b0, c.b1, c.b2, c.a1, c.a2)
    return _numpy_dsp.design_butterworth_bandpass(f_lo, f_hi, fs)


def bin_width_hz(nfft: int, fs: float) -> float:
    """Frequency spacing of an ``nfft``-point spectrum at rate ``fs``."""
    if HAVE_NATIVE:
        return float(NATIVE.bin_width_hz(int(nfft), float(fs)))
    return _numpy_dsp.bin_width_hz(nfft, fs)


def magnitude_spectrum(
    signal: npt.ArrayLike, nfft: int
) -> npt.NDArray[np.float64]:
    """One-sided Hann-windowed amplitude spectrum, zero-padded to ``nfft``."""
    arr = np.ascontiguousarray(signal, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(max(int(nfft), 2) // 2 + 1, dtype=np.float64)
    if HAVE_NATIVE:
        return np.asarray(NATIVE.magnitude_spectrum(arr, int(nfft)), dtype=np.float64)
    return _numpy_dsp.magnitude_spectrum(arr, nfft)


def find_band_peak(
    magnitude: npt.ArrayLike,
    bin_hz: float,
    f_lo: float,
    f_hi: float,
    snr_reference_db: float = 30.0,
    min_snr_db: float = 20.0,
    noise_floor: float = 0.0,
) -> SpectralPeak:
    """Dominant in-band peak with sub-bin interpolation and SNR scoring.

    ``noise_floor`` is a magnitude measured on a broadband spectrum of the same
    signal; pass 0 to derive one from ``magnitude`` instead (see
    :mod:`spectraflow.dsp._numpy_dsp` for why the explicit reference matters).
    """
    arr = np.ascontiguousarray(magnitude, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return SpectralPeak()
    if HAVE_NATIVE:
        native_peak = NATIVE.find_band_peak(
            arr, float(bin_hz), float(f_lo), float(f_hi),
            float(snr_reference_db), float(min_snr_db), float(noise_floor),
        )
        return SpectralPeak(
            frequency_hz=float(native_peak.frequency_hz),
            magnitude=float(native_peak.magnitude),
            peak_db=float(native_peak.peak_db),
            snr_db=float(native_peak.snr_db),
            snr_linear=float(native_peak.snr_linear),
            confidence=float(native_peak.confidence),
            valid=bool(native_peak.valid),
        )
    return _numpy_dsp.find_band_peak(
        arr, bin_hz, f_lo, f_hi, snr_reference_db, min_snr_db, noise_floor
    )


def snr_to_confidence(snr_db: float, reference_db: float = 30.0) -> float:
    """Logistic SNR(dB) -> confidence in ``[0, 1]``."""
    if HAVE_NATIVE:
        return float(NATIVE.snr_to_confidence(float(snr_db), float(reference_db)))
    return _numpy_dsp.snr_to_confidence(snr_db, reference_db)

"""Dummy CSI frame generator for the test-suite.

Every test in this directory runs with no ESP32 attached. Two generators are
provided:

* :func:`synthetic_source` wraps the production physics-based simulator
  (:class:`spectraflow.ingestion.SyntheticCsiSource`), so the DSP tests exercise
  the same signal model the deployment uses;
* :func:`random_csi` / :func:`noise_frames` produce simple, reproducible arrays
  for codec and shape tests where the physics is irrelevant.

Keeping the generator here (rather than only in the package) means the suite
documents exactly what stimulus each assertion was written against.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spectraflow.ingestion import (
    CsiFrame,
    Subject,
    SyntheticCsiSource,
    pack_csi_datagram,
)

__all__ = [
    "DEFAULT_SUBCARRIERS",
    "noise_frames",
    "random_csi",
    "round_trip_datagram",
    "synthetic_source",
]

#: Subcarriers used by the fixtures; matches a 20 MHz Wi-Fi channel's usable
#: subcarrier count closely enough for the DSP to behave realistically.
DEFAULT_SUBCARRIERS = 64


def synthetic_source(
    *,
    respiration_hz: float = 0.27,
    heart_hz: float = 1.2,
    heart_amplitude_m: float = 0.0006,
    n_subcarriers: int = DEFAULT_SUBCARRIERS,
    rate_hz: float = 20.0,
    seed: int = 12345,
    still: bool = False,
) -> SyntheticCsiSource:
    """Build a physics-based CSI source.

    Args:
        respiration_hz: Breathing rate in Hz (0.27 Hz == 16.2 RPM).
        heart_hz: Heart rate in Hz (1.2 Hz == 72 BPM).
        heart_amplitude_m: Cardiac chest excursion. The default is physically
            realistic (~0.6 mm) and is *deliberately* below the reliably
            detectable floor; pass a larger value to test the high-SNR path.
        still: When True, simulate an empty room (no chest displacement).
    """
    if still:
        subject = Subject(respiration_amplitude_m=0.0, heart_amplitude_m=0.0)
    else:
        subject = Subject(
            respiration_hz=respiration_hz,
            heart_hz=heart_hz,
            heart_amplitude_m=heart_amplitude_m,
        )
    return SyntheticCsiSource(
        n_subcarriers=n_subcarriers,
        rate_hz=rate_hz,
        subject=subject,
        seed=seed,
    )


def random_csi(
    n_frames: int = 32,
    n_subcarriers: int = DEFAULT_SUBCARRIERS,
    *,
    seed: int = 0,
) -> npt.NDArray[np.complex64]:
    """Reproducible pseudo-random complex CSI, ``(n_frames, n_subcarriers)``."""
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(n_frames, n_subcarriers))
        + 1j * rng.normal(size=(n_frames, n_subcarriers))
    ).astype(np.complex64)


def noise_frames(
    n_frames: int = 20,
    n_subcarriers: int = DEFAULT_SUBCARRIERS,
    *,
    seed: int = 7,
) -> list[CsiFrame]:
    """A list of :class:`CsiFrame` objects carrying pure noise."""
    matrix = random_csi(n_frames, n_subcarriers, seed=seed)
    return [
        CsiFrame(
            node_id=1,
            sequence=i,
            timestamp_us=int(i * 50_000),
            csi=row,
            channel=6,
        )
        for i, row in enumerate(matrix)
    ]


def round_trip_datagram(csi: npt.ArrayLike, **kwargs: object) -> bytes:
    """Serialize ``csi`` to a wire datagram using the production encoder."""
    return pack_csi_datagram(csi, **kwargs)  # type: ignore[arg-type]

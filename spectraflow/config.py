"""Central configuration for the Spectraflow pipeline.

Every tunable lives here so the DSP, inference and server layers never hard-code
a magic number. :class:`SpectraflowConfig` is a plain dataclass, which makes it
trivially overridable from a test, a CLI flag or an environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Final

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Nominal CSI frame rate produced by the ESP32 transmitter node, in Hz. The
#: Nyquist limit implied by this is 10 Hz, comfortably above the 2.5 Hz top of
#: the heart-rate band.
DEFAULT_SAMPLE_RATE_HZ: Final[float] = 20.0

#: Respiration band in Hz == 6..30 breaths per minute.
RESPIRATION_BAND_HZ: Final[tuple[float, float]] = (0.1, 0.5)

#: Heart-rate band in Hz == 48..150 beats per minute.
HEART_BAND_HZ: Final[tuple[float, float]] = (0.8, 2.5)

#: STFT analysis window length in seconds.
DEFAULT_WINDOW_SECONDS: Final[float] = 5.0


@dataclass(slots=True)
class SpectraflowConfig:
    """Runtime configuration for ingestion, DSP, inference and the server."""

    # -- Ingestion ---------------------------------------------------------
    udp_host: str = "0.0.0.0"
    udp_port: int = 5500
    #: Datagrams larger than this are rejected before parsing.
    max_datagram_bytes: int = 1472
    #: Frames older than this (seconds) are dropped as out-of-order stragglers.
    stale_frame_seconds: float = 1.0
    #: Lock-free ring buffer depth between the socket and the DSP worker.
    ring_capacity: int = 512

    # -- DSP ---------------------------------------------------------------
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    #: Clutter-rejection averaging window M, in frames. At 20 Hz, 100 frames
    #: is 5 s: long compared to a breath, short compared to environmental drift.
    clutter_window: int = 100
    respiration_band_hz: tuple[float, float] = RESPIRATION_BAND_HZ
    heart_band_hz: tuple[float, float] = HEART_BAND_HZ
    #: Sliding STFT / spectral-analysis window, in seconds.
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    #: FFT length. 256 bins at 20 Hz gives 0.078 Hz resolution (~4.7 BPM).
    nfft: int = 256
    #: Number of highest-variance subcarriers averaged into the sensing signal.
    dominant_subcarriers: int = 8
    #: SNR (dB) at which the confidence logistic reaches 0.5. Scored against a
    #: BROADBAND noise floor, a genuine vital-sign peak sits near 15-24 dB while
    #: an empty room stays below ~8 dB.
    snr_reference_db: float = 15.0
    #: Respiration validity threshold, in dB. Set inside the measured
    #: subject/empty-room gap (subject >= 8.2 dB, empty room <= 5.0 dB at the
    #: default 5 s window).
    min_snr_db: float = 7.0
    #: Heart-rate validity threshold, in dB. Deliberately much higher than the
    #: respiration one: the cardiac chest excursion is roughly 8x smaller than
    #: the respiratory one, so the heart band is far easier to fill with noise.
    #: A higher bar means the estimator reports "--" instead of inventing a
    #: plausible-looking but wrong heart rate.
    heart_min_snr_db: float = 12.0
    #: Recompute the spectrum at most this often, in seconds.
    vitals_update_seconds: float = 0.25
    #: A vital sign is only reported once this many consecutive analysis windows
    #: agree. Gating on persistence is what separates a real physiological rhythm
    #: (whose frequency is stable) from a slow environmental drift, whose
    #: spectral peak wanders from window to window.
    stability_min_samples: int = 4
    #: Maximum tolerated relative spread (interquartile range / median) of the
    #: recent peak estimates. 0.15 rejects a wandering drift peak while easily
    #: admitting a true rhythm.
    stability_tolerance: float = 0.15
    #: Consecutive invalid analyses after which the stability history is
    #: discarded, so a stale rate is never reported after the subject leaves.
    stability_max_misses: int = 4
    #: Residual gain applied to respiration harmonics inside the heart band.
    #: Respiration is 10-20x larger than the cardiac signal, so its 2nd and 3rd
    #: harmonics land squarely in the 0.8-2.5 Hz heart band and would otherwise
    #: be reported as a heart rate.
    harmonic_suppression: float = 0.05

    # -- Inference ---------------------------------------------------------
    #: Path to a 3D pose ONNX model. When absent or unloadable, the built-in
    #: data-driven synthetic keypoint generator is used.
    onnx_model_path: str | None = None
    #: Number of keypoints emitted (COCO-17).
    num_keypoints: int = 17

    # -- Server ------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    #: Maximum frames/s pushed to a single WebSocket client.
    stream_rate_hz: float = 20.0
    #: Per-client outbound queue depth. Beyond this the OLDEST frame is
    #: dropped, so a slow client never blocks the sensing pipeline.
    client_queue_depth: int = 8
    #: Run the synthetic CSI generator instead of binding the UDP socket.
    simulation: bool = field(
        default_factory=lambda: os.environ.get("SPECTRAFLOW_SIMULATION", "") not in ("", "0", "false")
    )
    #: Master switch for the built-in synthetic frame source.
    simulation_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ

    def validate(self) -> None:
        """Fail fast on a configuration that cannot produce valid physics."""
        nyquist = self.sample_rate_hz / 2.0
        for name, band in (
            ("respiration_band_hz", self.respiration_band_hz),
            ("heart_band_hz", self.heart_band_hz),
        ):
            lo, hi = band
            if not 0.0 < lo < hi:
                raise ValueError(f"{name} must satisfy 0 < lo < hi (got {band})")
            if hi >= nyquist:
                raise ValueError(
                    f"{name} top edge {hi} Hz is at or above the Nyquist limit "
                    f"{nyquist} Hz implied by sample_rate_hz={self.sample_rate_hz}"
                )
        if self.clutter_window < 2:
            raise ValueError("clutter_window must be >= 2")
        if self.nfft < 16:
            raise ValueError("nfft must be >= 16")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if not 0 < self.dominant_subcarriers:
            raise ValueError("dominant_subcarriers must be >= 1")

    def replace(self, **changes: object) -> "SpectraflowConfig":
        """Return a validated copy with ``changes`` applied."""
        updated = replace(self, **changes)  # type: ignore[arg-type]
        updated.validate()
        return updated


def default_config() -> SpectraflowConfig:
    """Build and validate the default configuration."""
    cfg = SpectraflowConfig()
    cfg.validate()
    return cfg

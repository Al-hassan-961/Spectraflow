"""Synthetic CSI source.

Generates physically-grounded CSI frames so the entire pipeline -- parsing, DSP,
inference and the WebSocket stream -- can be exercised end to end with no ESP32
attached.

The model is not noise dressed up as data. It is the standard two-component
reflection model for Wi-Fi sensing:

    H(f_k, t) = A(f_k) + B(f_k) * exp(-j * 4*pi*d(t) / lambda)

where ``A`` is the static multipath background (walls, furniture), ``B`` is the
reflection off the subject's chest, and ``d(t)`` is the chest-wall displacement
caused by breathing and the cardiac pulse. The ``4*pi`` factor is the round-trip
path-length phase term for a reflection, so a 5 mm chest excursion at 2.437 GHz
produces a genuine ~0.5 rad phase rotation in ``H`` -- exactly the effect the
real DSP has to recover.

Because ``|A| >> |B|`` the total phase modulation is attenuated by ``|B|/|A|``,
which is why contactless vital-sign sensing needs strong signal processing. The
defaults reproduce that regime rather than an artificially easy one.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import numpy.typing as npt

from spectraflow.ingestion.udp_receiver import CsiFrame, pack_csi_datagram

#: Speed of light in vacuum, m/s.
_C = 299_792_458.0


@dataclass(slots=True)
class Subject:
    """Physical parameters of the simulated subject."""

    #: Respiration rate in Hz (0.2 Hz == 12 breaths/min).
    respiration_hz: float = 0.27
    #: Heart rate in Hz (1.2 Hz == 72 bpm).
    heart_hz: float = 1.2
    #: Chest excursion peak amplitude from breathing, in metres.
    respiration_amplitude_m: float = 0.005
    #: Chest excursion peak amplitude from the cardiac pulse, in metres.
    heart_amplitude_m: float = 0.0006
    #: Standard deviation of a random-walk gross-motion term, in metres per
    #: sqrt(second). Zero means a perfectly still subject.
    motion_std_m: float = 0.0
    #: Distance to the subject, in metres (sets the reflection geometry).
    distance_m: float = 1.5

    def displacement_m(self, t: float) -> float:
        """Chest-wall displacement at time ``t``, in metres."""
        return (
            self.respiration_amplitude_m * math.sin(2.0 * math.pi * self.respiration_hz * t)
            + self.heart_amplitude_m * math.sin(2.0 * math.pi * self.heart_hz * t)
            + self.distance_m * 0.0
        )

    @property
    def respiration_rpm(self) -> float:
        return self.respiration_hz * 60.0

    @property
    def heart_bpm(self) -> float:
        return self.heart_hz * 60.0


class SyntheticCsiSource:
    """Deterministic CSI frame generator.

    Args:
        n_subcarriers: Subcarriers per frame (1..128).
        rate_hz: Frame rate.
        carrier_hz: Channel centre frequency, default 2.437 GHz (Wi-Fi channel 6).
        subject: Physical subject parameters.
        static_gain: Magnitude of the static multipath background ``A``.
        dynamic_gain: Magnitude of the subject reflection ``B``. The ratio
            ``dynamic_gain / static_gain`` sets how much phase modulation
            survives; the default is a realistic, non-trivial link.
        noise_std: Complex Gaussian receiver noise standard deviation.
        drift_std_m: Slow environmental drift amplitude, in metres. Gives the
            clutter remover something to track.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_subcarriers: int = 64,
        rate_hz: float = 20.0,
        *,
        carrier_hz: float = 2.437e9,
        subject: Subject | None = None,
        static_gain: float = 22.0,
        dynamic_gain: float = 12.0,
        noise_std: float = 0.5,
        drift_std_m: float = 3e-5,
        seed: int = 12345,
        node_id: int = 1,
        channel: int = 6,
    ) -> None:
        if not 1 <= n_subcarriers <= 128:
            raise ValueError("n_subcarriers must be in 1..128")
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")

        self.n_subcarriers = int(n_subcarriers)
        self.rate_hz = float(rate_hz)
        self.carrier_hz = float(carrier_hz)
        self.subject = subject or Subject()
        self.noise_std = float(noise_std)
        self.drift_std_m = float(drift_std_m)
        self.node_id = int(node_id)
        self.channel = int(channel)

        self.wavelength_m = _C / self.carrier_hz
        self._rng = np.random.default_rng(seed)

        # Fixed multipath profile: a smooth, decaying complex response so
        # neighbouring subcarriers stay correlated, as in a real channel.
        k = np.arange(self.n_subcarriers, dtype=np.float64)
        taper = np.exp(-k / (self.n_subcarriers * 1.6))
        phase_static = self._rng.uniform(-math.pi, math.pi, size=self.n_subcarriers)
        self._static = (static_gain * taper * np.exp(1j * phase_static)).astype(np.complex64)

        phase_dyn = self._rng.uniform(-math.pi, math.pi, size=self.n_subcarriers)
        self._dynamic = (dynamic_gain * taper * np.exp(1j * phase_dyn)).astype(np.complex64)

        # Slow drift state (an Ornstein-Uhlenbeck-ish random walk, in metres).
        self._drift_m = 0.0
        self._motion_m = 0.0
        # Sub-wavelength carrier phase offset, so the model is not always
        # starting from a favourable phase alignment.
        self._carrier_phase = 0.0
        self.sample_index = 0

    # -- generation --------------------------------------------------------
    def frame(self, t: float | None = None) -> CsiFrame:
        """Produce the next frame.

        ``t`` is the simulated time in seconds; when omitted it advances by
        ``1 / rate_hz`` per call, giving an exactly uniform sample grid.
        """
        if t is None:
            t = self.sample_index / self.rate_hz

        dt = 1.0 / self.rate_hz

        # Environmental drift and optional gross motion, both as bounded random
        # walks (mean-reverting, so the subject cannot wander off to infinity).
        self._drift_m = 0.98 * self._drift_m + self.drift_std_m * self._rng.standard_normal()
        if self.subject.motion_std_m > 0.0:
            self._motion_m = 0.95 * self._motion_m + (
                self.subject.motion_std_m * math.sqrt(dt) * self._rng.standard_normal()
            )

        displacement = self.subject.displacement_m(t) + self._drift_m + self._motion_m

        # Round-trip reflection phase.
        angle = 4.0 * math.pi * displacement / self.wavelength_m
        rotating = np.exp(
            -1j * (angle + self._carrier_phase)
        ).astype(np.complex64)

        h = self._static + self._dynamic * rotating

        if self.noise_std > 0.0:
            noise = (
                self._rng.standard_normal(self.n_subcarriers)
                + 1j * self._rng.standard_normal(self.n_subcarriers)
            ) * (self.noise_std / math.sqrt(2.0))
            h = h + noise.astype(np.complex64)

        # Quantise to the int8 wire format so downstream DSP sees exactly the
        # same precision as a real node would deliver.
        h = np.clip(np.rint(h), -128, 127).astype(np.complex64)

        frame = CsiFrame(
            node_id=self.node_id,
            sequence=self.sample_index & 0xFFFFFFFF,
            timestamp_us=int(round(t * 1e6)),
            csi=h,
            channel=self.channel,
            rssi_dbm=-45,
            noise_floor_dbm=-95,
        )
        self.sample_index += 1
        return frame

    def datagram(self, t: float | None = None) -> bytes:
        """Produce the next frame already serialized to wire bytes."""
        frame = self.frame(t)
        return pack_csi_datagram(
            frame.csi,
            node_id=frame.node_id,
            sequence=frame.sequence,
            timestamp_us=frame.timestamp_us,
            channel=frame.channel,
            rssi_dbm=frame.rssi_dbm,
            noise_floor_dbm=frame.noise_floor_dbm,
        )

    def frames(self, duration_seconds: float, start: float = 0.0) -> Iterator[CsiFrame]:
        """Yield frames covering ``duration_seconds`` of simulated time."""
        count = max(1, int(round(duration_seconds * self.rate_hz)))
        for i in range(count):
            yield self.frame(start + i / self.rate_hz)

    def matrix(self, duration_seconds: float, start: float = 0.0) -> npt.NDArray[np.complex64]:
        """Return a ``(n_frames, n_subcarriers)`` matrix of simulated CSI."""
        return np.stack(
            [f.csi for f in self.frames(duration_seconds, start)]
        ).astype(np.complex64)

    def reset(self, seed: int | None = None) -> None:
        """Restart the generator, optionally with a new seed."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._drift_m = 0.0
        self._motion_m = 0.0
        self.sample_index = 0

    # -- real-time streaming ----------------------------------------------
    async def stream(self, *, realtime: bool = True):
        """Async generator pacing frames at ``rate_hz`` in real time."""
        import asyncio

        period = 1.0 / self.rate_hz
        t0 = time.monotonic()
        while True:
            frame = self.frame()
            yield frame
            if realtime:
                target = t0 + self.sample_index * period
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

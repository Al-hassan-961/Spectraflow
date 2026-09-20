"""Raw CSI frame parsing and UDP ingestion.

Wire format (see ``core/include/csi_packet.hpp`` for the authoritative table):
a 24-byte little-endian header followed by ``n_subcarriers * 2`` bytes of
interleaved ``int8`` (imaginary, real) pairs.

Parsing prefers the native ``spectraflow_core`` codec and falls back to an
equivalent :mod:`struct` implementation, so the parser behaves identically
whether or not the extension is built.
"""

from __future__ import annotations

import asyncio
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Iterable

import numpy as np
import numpy.typing as npt

from spectraflow._native import HAVE_NATIVE, NATIVE

# ---------------------------------------------------------------------------
# Protocol constants (must match core/include/csi_packet.hpp)
# ---------------------------------------------------------------------------

CSI_PACKET_MAGIC = 0x5F43
CSI_PROTOCOL_VERSION = 1
CSI_HEADER_BYTES = 24
CSI_MAX_SUBCARRIERS = 128
CSI_MAX_DATAGRAM_BYTES = CSI_HEADER_BYTES + CSI_MAX_SUBCARRIERS * 2

CSI_FLAG_FIRST_WORD_INVALID = 1 << 0
CSI_FLAG_LAST_WORD_INVALID = 1 << 1

#: Little-endian, no padding:
#: magic(H) version(B) flags(B) node_id(H) payload_bytes(H) sequence(I)
#: timestamp_us(Q) channel(B) rssi(b) noise_floor(b) n_subcarriers(B)
_HEADER = struct.Struct("<HBBHHIQBbbB")

assert _HEADER.size == CSI_HEADER_BYTES, "header struct must be exactly 24 bytes"


class CsiParseError(ValueError):
    """Raised when a datagram is not a well-formed CSI packet."""


# ---------------------------------------------------------------------------
# Frame model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CsiFrame:
    """A single decoded CSI observation.

    ``csi`` holds one complex channel estimate per subcarrier in subcarrier
    order, so ``csi[k]`` is ``H(f_k)``.
    """

    node_id: int
    sequence: int
    timestamp_us: int
    csi: npt.NDArray[np.complex64]
    channel: int = 0
    rssi_dbm: int = 0
    noise_floor_dbm: int = 0
    flags: int = 0
    invalid_edges: int = 0
    #: Local arrival time (``time.time()``), used for freshness checks and for
    #: pacing the STFT when the sender clock is unavailable.
    received_at: float = field(default_factory=time.time)

    @property
    def n_subcarriers(self) -> int:
        return int(self.csi.size)

    @property
    def timestamp_seconds(self) -> float:
        """Sender capture time in seconds."""
        return self.timestamp_us * 1e-6

    def amplitude_db(self) -> npt.NDArray[np.float32]:
        """Per-subcarrier amplitude in dB, floored to avoid ``log10(0)``."""
        return (20.0 * np.log10(np.maximum(np.abs(self.csi), 1e-8))).astype(np.float32)

    def phase_radians(self) -> npt.NDArray[np.float32]:
        """Raw (wrapped) per-subcarrier phase in radians."""
        return np.angle(self.csi).astype(np.float32)

    def power_linear(self) -> npt.NDArray[np.float32]:
        """Per-subcarrier power ``|H|^2``."""
        return (np.abs(self.csi) ** 2).astype(np.float32)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<CsiFrame node={self.node_id} seq={self.sequence} "
            f"n={self.n_subcarriers} t={self.timestamp_seconds:.3f}s>"
        )


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def _parse_python(data: bytes) -> CsiFrame:
    """Pure-Python/NumPy datagram parser.

    Kept byte-for-byte compatible with the native codec so tests and
    deployments agree regardless of which backend is active.
    """
    if len(data) < CSI_HEADER_BYTES:
        raise CsiParseError(
            f"truncated header ({len(data)} < {CSI_HEADER_BYTES} bytes)"
        )

    (
        magic,
        version,
        flags,
        node_id,
        payload_bytes,
        sequence,
        timestamp_us,
        channel,
        rssi_dbm,
        noise_floor_dbm,
        n_subcarriers,
    ) = _HEADER.unpack_from(data, 0)

    if magic != CSI_PACKET_MAGIC:
        raise CsiParseError(f"bad magic 0x{magic:04X}")
    if version != CSI_PROTOCOL_VERSION:
        raise CsiParseError(f"unsupported version {version}")
    if n_subcarriers == 0:
        raise CsiParseError("zero subcarriers")
    if n_subcarriers > CSI_MAX_SUBCARRIERS:
        raise CsiParseError(
            f"too many subcarriers ({n_subcarriers} > {CSI_MAX_SUBCARRIERS})"
        )
    expected = n_subcarriers * 2
    if payload_bytes != expected:
        raise CsiParseError(
            f"payload_bytes mismatch ({payload_bytes} != {expected})"
        )
    if len(data) < CSI_HEADER_BYTES + expected:
        raise CsiParseError("truncated payload")

    raw = np.frombuffer(
        data, dtype=np.int8, count=expected, offset=CSI_HEADER_BYTES
    ).reshape(n_subcarriers, 2)
    # ESP32 convention: buf[2i] is imaginary, buf[2i+1] is real.
    csi = (raw[:, 1].astype(np.float32) + 1j * raw[:, 0].astype(np.float32)).astype(
        np.complex64
    )

    invalid_edges = 0
    if flags & CSI_FLAG_FIRST_WORD_INVALID and csi.size > 2:
        csi = csi[1:]
        invalid_edges += 1
    if flags & CSI_FLAG_LAST_WORD_INVALID and csi.size > 2:
        csi = csi[:-1]
        invalid_edges += 1

    return CsiFrame(
        node_id=node_id,
        sequence=sequence,
        timestamp_us=timestamp_us,
        csi=csi,
        channel=channel,
        rssi_dbm=int(rssi_dbm),
        noise_floor_dbm=int(noise_floor_dbm),
        flags=flags,
        invalid_edges=invalid_edges,
    )


def _from_native(native_frame: object) -> CsiFrame:
    """Convert a native ``CsiFrame`` binding into the Python dataclass."""
    header = native_frame.header  # type: ignore[attr-defined]
    return CsiFrame(
        node_id=int(header.node_id),
        sequence=int(header.sequence),
        timestamp_us=int(header.timestamp_us),
        csi=np.asarray(native_frame.csi_array(), dtype=np.complex64),  # type: ignore[attr-defined]
        channel=int(header.channel),
        rssi_dbm=int(header.rssi_dbm),
        noise_floor_dbm=int(header.noise_floor_dbm),
        flags=int(header.flags),
        invalid_edges=int(native_frame.invalid_edges),  # type: ignore[attr-defined]
    )


def parse_csi_datagram(data: bytes) -> CsiFrame:
    """Decode one UDP payload into a :class:`CsiFrame`.

    Raises:
        CsiParseError: if the datagram is malformed or truncated.
    """
    if len(data) > CSI_MAX_DATAGRAM_BYTES:
        raise CsiParseError(
            f"datagram too large ({len(data)} > {CSI_MAX_DATAGRAM_BYTES} bytes)"
        )
    if HAVE_NATIVE:
        try:
            return _from_native(NATIVE.parse_csi_packet(data))
        except CsiParseError:
            raise
        except Exception as exc:  # native raises RuntimeError on malformed input
            raise CsiParseError(str(exc)) from exc
    return _parse_python(data)


def pack_csi_datagram(
    csi: npt.ArrayLike,
    *,
    node_id: int = 1,
    sequence: int = 0,
    timestamp_us: int = 0,
    channel: int = 6,
    rssi_dbm: int = -45,
    noise_floor_dbm: int = -95,
    flags: int = 0,
) -> bytes:
    """Encode a CSI vector into a wire datagram.

    Used by the synthetic source and the test-suite. Values are rounded and
    clamped to the ``int8`` range the format provides, mirroring the firmware.
    """
    arr = np.asarray(csi, dtype=np.complex64).reshape(-1)
    n = arr.size
    if not 1 <= n <= CSI_MAX_SUBCARRIERS:
        raise ValueError(f"n_subcarriers must be in 1..{CSI_MAX_SUBCARRIERS}, got {n}")

    imag = np.clip(np.rint(arr.imag), -128, 127).astype(np.int8)
    real = np.clip(np.rint(arr.real), -128, 127).astype(np.int8)
    interleaved = np.empty(n * 2, dtype=np.int8)
    interleaved[0::2] = imag
    interleaved[1::2] = real

    header = _HEADER.pack(
        CSI_PACKET_MAGIC,
        CSI_PROTOCOL_VERSION,
        flags & 0xFF,
        node_id & 0xFFFF,
        n * 2,
        sequence & 0xFFFFFFFF,
        timestamp_us & 0xFFFFFFFFFFFFFFFF,
        channel & 0xFF,
        max(-128, min(127, int(rssi_dbm))),
        max(-128, min(127, int(noise_floor_dbm))),
        n,
    )
    return header + interleaved.tobytes()


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class FrameBuffer:
    """Bounded FIFO of decoded frames with explicit drop accounting.

    A :class:`collections.deque` is the right structure on the Python side: it
    is O(1) at both ends, drops the *oldest* frame when full (which is what a
    real-time sensing pipeline wants -- stale CSI is worthless), and avoids a
    Python<->C++ conversion per frame. The native ``CsiRingBuffer`` is the
    equivalent primitive for C++ embedders.
    """

    __slots__ = ("_buffer", "capacity", "dropped")

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._buffer: Deque[CsiFrame] = deque(maxlen=self.capacity)
        self.dropped = 0

    def push(self, frame: CsiFrame) -> None:
        if len(self._buffer) == self.capacity:
            self.dropped += 1
        self._buffer.append(frame)

    def pop(self) -> CsiFrame | None:
        return self._buffer.popleft() if self._buffer else None

    def drain(self, max_frames: int = 0) -> list[CsiFrame]:
        out: list[CsiFrame] = []
        while self._buffer and (max_frames <= 0 or len(out) < max_frames):
            out.append(self._buffer.popleft())
        return out

    def latest(self) -> CsiFrame | None:
        return self._buffer[-1] if self._buffer else None

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# UDP ingestion
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReceiverStats:
    """Counters describing receiver health."""

    datagrams: int = 0
    parsed: int = 0
    parse_errors: int = 0
    oversized: int = 0
    last_error: str | None = None
    last_datagram_at: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.parse_errors / self.datagrams if self.datagrams else 0.0


class UdpCsiReceiver:
    """Asyncio UDP endpoint that decodes CSI datagrams into frames.

    Usage::

        receiver = UdpCsiReceiver(port=5500)
        await receiver.start()
        frame = await receiver.get()

    Frames are also handed to an optional ``on_frame`` callback, which runs on
    the event loop; keep it cheap or dispatch to a worker.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5500,
        *,
        max_datagram_bytes: int = 1472,
        stale_after_seconds: float = 1.0,
        capacity: int = 512,
        on_frame: Callable[[CsiFrame], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.max_datagram_bytes = int(max_datagram_bytes)
        self.stale_after_seconds = float(stale_after_seconds)
        self.stats = ReceiverStats()
        self.buffer = FrameBuffer(capacity)
        self.on_frame = on_frame
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: "_CsiDatagramProtocol | None" = None
        self._waiters: Deque[asyncio.Future[CsiFrame]] = deque()
        self._bound_port: int | None = None

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> int:
        """Bind the socket. Returns the actual bound port."""
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _CsiDatagramProtocol(self),
            local_addr=(self.host, self.port),
        )
        self._transport = transport  # type: ignore[assignment]
        self._protocol = protocol  # type: ignore[assignment]
        sockname = transport.get_extra_info("sockname")
        self._bound_port = int(sockname[1]) if sockname else self.port
        self.port = self._bound_port
        return self._bound_port

    async def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        for waiter in self._waiters:
            if not waiter.done():
                waiter.cancel()
        self._waiters.clear()

    @property
    def running(self) -> bool:
        return self._transport is not None

    @property
    def bound_port(self) -> int | None:
        return self._bound_port

    # -- frame access -----------------------------------------------------
    async def get(self) -> CsiFrame:
        """Await the next frame."""
        frame = self.buffer.pop()
        if frame is not None:
            return frame
        loop = asyncio.get_running_loop()
        future: asyncio.Future[CsiFrame] = loop.create_future()
        self._waiters.append(future)
        return await future

    def _deliver(self, frame: CsiFrame) -> None:
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(frame)
                break
        else:
            self.buffer.push(frame)
        if self.on_frame is not None:
            self.on_frame(frame)

    # -- datagram handling ------------------------------------------------
    def _handle_datagram(self, data: bytes) -> None:
        self.stats.datagrams += 1
        self.stats.last_datagram_at = time.time()

        if len(data) > self.max_datagram_bytes:
            self.stats.oversized += 1
            self.stats.parse_errors += 1
            self.stats.last_error = f"oversized datagram ({len(data)} bytes)"
            return

        try:
            frame = parse_csi_datagram(data)
        except CsiParseError as exc:
            self.stats.parse_errors += 1
            self.stats.last_error = str(exc)
            return

        self.stats.parsed += 1

        # Reject stragglers: a frame whose sender timestamp lags the newest we
        # have seen by more than the staleness budget would land out of order in
        # the STFT and corrupt the phase track.
        newest = self.buffer.latest()
        if (
            newest is not None
            and frame.timestamp_us < newest.timestamp_us
            and (newest.timestamp_us - frame.timestamp_us) * 1e-6 > self.stale_after_seconds
        ):
            self.stats.last_error = "stale frame dropped"
            return

        self._deliver(frame)


class _CsiDatagramProtocol(asyncio.DatagramProtocol):
    """Thin asyncio protocol bridging datagrams into :class:`UdpCsiReceiver`."""

    def __init__(self, receiver: UdpCsiReceiver) -> None:
        self._receiver = receiver

    def datagram_received(self, data: bytes, addr: object) -> None:
        del addr
        self._receiver._handle_datagram(data)

    def error_received(self, exc: Exception) -> None:
        self._receiver.stats.last_error = f"socket error: {exc}"

    def connection_lost(self, exc: Exception | None) -> None:
        if exc is not None:
            self._receiver.stats.last_error = f"connection lost: {exc}"


def frames_to_batch(frames: Iterable[CsiFrame]) -> npt.NDArray[np.complex64]:
    """Stack frames into a ``(n_frames, n_subcarriers)`` complex64 matrix.

    Raises:
        ValueError: if the frames do not share a subcarrier count.
    """
    frame_list = list(frames)
    if not frame_list:
        return np.empty((0, 0), dtype=np.complex64)
    widths = {f.n_subcarriers for f in frame_list}
    if len(widths) != 1:
        raise ValueError(f"inconsistent subcarrier counts in batch: {sorted(widths)}")
    return np.stack([f.csi for f in frame_list]).astype(np.complex64)

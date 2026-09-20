"""Ingestion tests: wire codec, framing, UDP transport and the synthetic source.

These run without hardware. The codec tests are deliberately byte-level: the
24-byte header is a contract shared with the ESP32 firmware and the C++ core, so
an offset or width mistake silently corrupts every downstream measurement rather
than raising.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from csi_fixtures import DEFAULT_SUBCARRIERS, random_csi, synthetic_source
from spectraflow._native import HAVE_NATIVE, NATIVE
from spectraflow.ingestion import (
    CSI_HEADER_BYTES,
    CSI_MAX_DATAGRAM_BYTES,
    CSI_MAX_SUBCARRIERS,
    CSI_PACKET_MAGIC,
    CSI_PROTOCOL_VERSION,
    CSI_FLAG_FIRST_WORD_INVALID,
    CSI_FLAG_LAST_WORD_INVALID,
    CsiFrame,
    CsiParseError,
    FrameBuffer,
    UdpCsiReceiver,
    pack_csi_datagram,
    parse_csi_datagram,
)
from spectraflow.ingestion.udp_receiver import _HEADER, _parse_python


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


def test_header_is_exactly_24_bytes():
    assert _HEADER.size == CSI_HEADER_BYTES == 24


def test_header_layout_matches_the_frozen_contract():
    """Pin every offset so a future edit cannot silently shift a field.

    This is a regression test: an earlier revision placed `sequence` (uint32) at
    offset 6 and `timestamp_us` (uint64) at offset 8, which overlap by two bytes
    and corrupted both. Any such change must fail here.
    """
    expected = {
        "magic": (0, 2),
        "version": (2, 1),
        "flags": (3, 1),
        "node_id": (4, 2),
        "payload_bytes": (6, 2),
        "sequence": (8, 4),
        "timestamp_us": (12, 8),
        "channel": (20, 1),
        "rssi_dbm": (21, 1),
        "noise_floor_dbm": (22, 1),
        "n_subcarriers": (23, 1),
    }
    assert _HEADER.format == "<HBBHHIQBbbB"

    offset = 0
    for name, (start, size) in expected.items():
        assert start == offset, f"{name} should start at {offset}, is at {start}"
        offset += size
    assert offset == CSI_HEADER_BYTES


def test_all_header_fields_survive_a_round_trip():
    csi = (np.arange(48) % 13 - 6) + 1j * (np.arange(48) % 7 - 3)
    raw = pack_csi_datagram(
        csi,
        node_id=4242,
        sequence=0xDEADBEEF,
        timestamp_us=123_456_789_012,
        channel=11,
        rssi_dbm=-37,
        noise_floor_dbm=-91,
        flags=CSI_FLAG_FIRST_WORD_INVALID,
    )
    frame = parse_csi_datagram(raw)

    # A truncated read of the old layout produced a wildly wrong sequence here.
    assert frame.sequence == 0xDEADBEEF
    assert frame.timestamp_us == 123_456_789_012
    assert frame.node_id == 4242
    assert frame.channel == 11
    assert frame.rssi_dbm == -37
    assert frame.noise_floor_dbm == -91
    assert frame.flags == CSI_FLAG_FIRST_WORD_INVALID


def test_iq_payload_round_trips_with_imaginary_first():
    """The payload is interleaved (imag, real), per the ESP32 CSI convention."""
    # first_word_invalid would excise an edge word, so keep flags clear here.
    csi = np.array([3 + 4j, -5 - 6j, 7 + 0j, 0 - 8j], dtype=np.complex64)
    frame = parse_csi_datagram(pack_csi_datagram(csi, flags=0))
    assert np.allclose(frame.csi, csi)

    raw = pack_csi_datagram(csi, flags=0)
    payload = raw[CSI_HEADER_BYTES:]
    assert payload[0] == 4  # subcarrier 0 imaginary
    assert payload[1] == 3  # subcarrier 0 real


def test_pack_clamps_to_int8_range():
    csi = np.array([1000 + 1000j, -1000 - 1000j], dtype=np.complex64)
    frame = parse_csi_datagram(pack_csi_datagram(csi, flags=0))
    assert frame.csi[0].real == 127 and frame.csi[0].imag == 127
    assert frame.csi[1].real == -128 and frame.csi[1].imag == -128


@pytest.mark.parametrize("n", [1, 2, CSI_MAX_SUBCARRIERS])
def test_subcarrier_count_boundaries_round_trip(n):
    csi = np.ones(n, dtype=np.complex64) * (1 + 1j)
    frame = parse_csi_datagram(pack_csi_datagram(csi, flags=0))
    assert frame.n_subcarriers == n


def test_subcarrier_count_above_the_cap_is_rejected_at_encode():
    with pytest.raises(ValueError):
        pack_csi_datagram(np.ones(CSI_MAX_SUBCARRIERS + 1, dtype=np.complex64))


def test_datagram_stays_below_the_mtu_budget():
    full = pack_csi_datagram(np.ones(CSI_MAX_SUBCARRIERS, dtype=np.complex64), flags=0)
    assert len(full) == CSI_MAX_DATAGRAM_BYTES
    assert len(full) < 1472, "must not fragment at the IP layer"


# ---------------------------------------------------------------------------
# Malformed input handling
# ---------------------------------------------------------------------------


def test_empty_datagram_is_rejected():
    with pytest.raises(CsiParseError):
        parse_csi_datagram(b"")


def test_truncated_header_is_rejected():
    with pytest.raises(CsiParseError):
        parse_csi_datagram(b"\x43\x5f\x01")


def test_truncated_payload_is_rejected():
    raw = pack_csi_datagram(np.ones(32, dtype=np.complex64), flags=0)
    with pytest.raises(CsiParseError):
        parse_csi_datagram(raw[:-4])


def test_bad_magic_is_rejected():
    raw = bytearray(pack_csi_datagram(np.ones(8, dtype=np.complex64), flags=0))
    raw[0] ^= 0xFF
    with pytest.raises(CsiParseError):
        parse_csi_datagram(bytes(raw))


def test_unsupported_version_is_rejected():
    raw = bytearray(pack_csi_datagram(np.ones(8, dtype=np.complex64), flags=0))
    raw[2] = 99
    with pytest.raises(CsiParseError):
        parse_csi_datagram(bytes(raw))


def test_payload_length_mismatch_is_rejected():
    raw = bytearray(pack_csi_datagram(np.ones(8, dtype=np.complex64), flags=0))
    raw[6] = 0xFF  # corrupt payload_bytes
    raw[7] = 0x7F
    with pytest.raises(CsiParseError):
        parse_csi_datagram(bytes(raw))


def test_oversized_datagram_is_rejected():
    with pytest.raises(CsiParseError):
        parse_csi_datagram(b"\x43\x5f" + b"\x00" * (CSI_MAX_DATAGRAM_BYTES + 100))


def test_zero_subcarriers_is_rejected():
    raw = bytearray(pack_csi_datagram(np.ones(4, dtype=np.complex64), flags=0))
    raw[6] = 0  # payload_bytes
    raw[7] = 0
    raw[23] = 0  # n_subcarriers
    with pytest.raises(CsiParseError):
        parse_csi_datagram(bytes(raw))


def test_constant_magic_and_version_are_what_the_firmware_writes():
    raw = pack_csi_datagram(np.ones(4, dtype=np.complex64))
    assert int.from_bytes(raw[0:2], "little") == CSI_PACKET_MAGIC == 0x5F43
    assert raw[2] == CSI_PROTOCOL_VERSION == 1


# ---------------------------------------------------------------------------
# Invalid edge word handling
# ---------------------------------------------------------------------------


def test_invalid_edge_flags_excise_words():
    csi = np.arange(10, dtype=np.float32) + 1j * np.arange(10, dtype=np.float32)
    plain = parse_csi_datagram(pack_csi_datagram(csi, flags=0))
    first = parse_csi_datagram(pack_csi_datagram(csi, flags=CSI_FLAG_FIRST_WORD_INVALID))
    last = parse_csi_datagram(pack_csi_datagram(csi, flags=CSI_FLAG_LAST_WORD_INVALID))
    both = parse_csi_datagram(
        pack_csi_datagram(csi, flags=CSI_FLAG_FIRST_WORD_INVALID | CSI_FLAG_LAST_WORD_INVALID)
    )

    assert plain.n_subcarriers == 10
    assert first.n_subcarriers == 9 and first.invalid_edges == 1
    assert last.n_subcarriers == 9 and last.invalid_edges == 1
    assert both.n_subcarriers == 8 and both.invalid_edges == 2
    assert np.allclose(first.csi, csi[1:])


def test_edge_excision_never_empties_a_tiny_frame():
    csi = np.array([1 + 1j, 2 + 2j], dtype=np.complex64)
    frame = parse_csi_datagram(
        pack_csi_datagram(csi, flags=CSI_FLAG_FIRST_WORD_INVALID | CSI_FLAG_LAST_WORD_INVALID)
    )
    assert frame.n_subcarriers == 2, "must not remove words from a 2-subcarrier frame"


# ---------------------------------------------------------------------------
# Backend equivalence
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_native_and_python_parsers_agree():
    rng = np.random.default_rng(3)
    for n in (4, 16, 64, 128):
        csi = (
            rng.normal(size=n) + 1j * rng.normal(size=n)
        ).astype(np.complex64)
        csi = np.clip(np.rint(csi * 5), -128, 127).astype(np.complex64)
        raw = pack_csi_datagram(csi, node_id=9, sequence=77, timestamp_us=999, flags=0)

        native = parse_csi_datagram(raw)
        python = _parse_python(raw)

        assert native.sequence == python.sequence == 77
        assert native.timestamp_us == python.timestamp_us == 999
        assert native.node_id == python.node_id == 9
        assert np.array_equal(native.csi, python.csi)


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_native_parser_rejects_malformed_input():
    assert NATIVE.try_parse_csi_packet(b"garbage") is None
    with pytest.raises(Exception):
        NATIVE.parse_csi_packet(b"garbage")


# ---------------------------------------------------------------------------
# Frame buffer
# ---------------------------------------------------------------------------


def _frame(seq: int) -> CsiFrame:
    return CsiFrame(
        node_id=1,
        sequence=seq,
        timestamp_us=seq * 50_000,
        csi=np.ones(4, dtype=np.complex64),
    )


def test_frame_buffer_drops_the_oldest_when_full():
    buffer = FrameBuffer(capacity=3)
    for seq in range(5):
        buffer.push(_frame(seq))

    assert len(buffer) == 3
    assert buffer.dropped == 2
    assert [f.sequence for f in buffer.drain()] == [2, 3, 4]


def test_frame_buffer_rejects_a_zero_capacity():
    with pytest.raises(ValueError):
        FrameBuffer(capacity=0)


def test_frame_buffer_latest_and_clear():
    buffer = FrameBuffer(capacity=4)
    assert buffer.latest() is None
    buffer.push(_frame(1))
    buffer.push(_frame(2))
    assert buffer.latest().sequence == 2
    buffer.clear()
    assert buffer.latest() is None and len(buffer) == 0


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_native_ring_buffer_is_lock_free_spsc_and_reports_drops():
    def native_frame(seq: int):
        """Build a native frame the way a C++ embedder would."""
        header = NATIVE.CsiHeader()
        header.node_id = 1
        header.sequence = seq
        header.timestamp_us = seq * 50_000
        return NATIVE.make_csi_frame(
            header, np.ones(4, dtype=np.complex64) * (1 + 1j)
        )

    ring = NATIVE.CsiRingBuffer(8)
    assert ring.capacity() == 8
    assert ring.empty()

    for seq in range(8):
        assert ring.push(native_frame(seq)) is True
    assert ring.full() is True

    # The producer never blocks; it drops and counts instead.
    assert ring.push(native_frame(99)) is False
    stats = ring.stats()
    assert stats.dropped == 1 and stats.pushed == 8 and stats.size == 8

    popped = ring.pop()
    assert popped.header.sequence == 0, "FIFO order must be preserved"
    assert ring.size() == 7

    ring.clear()
    assert ring.empty() and ring.pop() is None


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_native_frame_accepts_a_numpy_csi_array():
    header = NATIVE.CsiHeader()
    header.node_id = 5
    csi = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64)
    frame = NATIVE.make_csi_frame(header, csi)

    assert frame.n_subcarriers == 3
    assert frame.header.payload_bytes == 6
    assert np.allclose(frame.csi_array(), csi)
    assert np.allclose(np.asarray(frame.csi), csi)  # property getter


# ---------------------------------------------------------------------------
# UDP transport
# ---------------------------------------------------------------------------


def test_udp_receiver_receives_and_decodes_over_a_real_socket():
    async def scenario() -> tuple[int, list, object]:
        receiver = UdpCsiReceiver(host="127.0.0.1", port=0)
        port = await receiver.start()
        assert port > 0

        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol, remote_addr=("127.0.0.1", port)
        )
        source = synthetic_source()
        expected = []
        for i in range(5):
            frame = source.frame()
            expected.append(frame.csi.copy())
            transport.sendto(pack_csi_datagram(frame.csi, sequence=i, flags=0))
        await asyncio.sleep(0.2)

        received = []
        for _ in range(5):
            received.append(await asyncio.wait_for(receiver.get(), timeout=2.0))

        station = len(received)
        stats = receiver.stats
        await receiver.stop()
        transport.close()
        return station, received, stats

    count, frames, stats = asyncio.run(scenario())
    assert count == 5
    assert stats.datagrams == 5 and stats.parsed == 5 and stats.parse_errors == 0
    assert [f.sequence for f in frames] == [0, 1, 2, 3, 4]


def test_udp_receiver_counts_rather_than_crashes_on_garbage():
    async def scenario():
        receiver = UdpCsiReceiver(host="127.0.0.1", port=0)
        port = await receiver.start()

        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol, remote_addr=("127.0.0.1", port)
        )
        transport.sendto(b"not a csi packet")
        transport.sendto(b"\x00" * 30)
        transport.sendto(pack_csi_datagram(np.ones(8, dtype=np.complex64), flags=0))
        await asyncio.sleep(0.2)

        frame = await asyncio.wait_for(receiver.get(), timeout=2.0)
        await asyncio.sleep(0.05)
        stats = receiver.stats
        await receiver.stop()
        transport.close()
        return frame, stats

    frame, stats = asyncio.run(scenario())
    assert frame.sequence == 0
    assert stats.datagrams == 3
    assert stats.parsed == 1
    assert stats.parse_errors == 2
    assert stats.error_rate > 0.0


# ---------------------------------------------------------------------------
# Synthetic source
# ---------------------------------------------------------------------------


def test_synthetic_source_produces_the_declared_shape_and_dtype():
    source = synthetic_source()
    frames = list(source.frames(2.0))
    assert len(frames) == 40  # 2 s at 20 Hz
    for frame in frames:
        assert frame.csi.shape == (DEFAULT_SUBCARRIERS,)
        assert frame.csi.dtype == np.complex64
        assert frame.n_subcarriers == DEFAULT_SUBCARRIERS


def test_synthetic_source_is_deterministic_for_a_given_seed():
    a = synthetic_source(seed=42).matrix(5.0)
    b = synthetic_source(seed=42).matrix(5.0)
    assert np.array_equal(a, b)

    c = synthetic_source(seed=43).matrix(5.0)
    assert not np.array_equal(a, c)


def test_synthetic_source_quantises_to_the_int8_wire_range():
    matrix = synthetic_source().matrix(3.0)
    assert matrix.real.min() >= -128 and matrix.real.max() <= 127
    assert matrix.imag.min() >= -128 and matrix.imag.max() <= 127
    assert np.all(matrix.real == np.rint(matrix.real)), "must be integer-valued"


def test_synthetic_frames_survive_the_wire_codec():
    source = synthetic_source()
    for _ in range(4):
        frame = source.frame()
        decoded = parse_csi_datagram(
            pack_csi_datagram(frame.csi, sequence=frame.sequence, flags=0)
        )
        assert np.allclose(decoded.csi, frame.csi)


def test_synthetic_timestamps_are_uniform():
    source = synthetic_source(rate_hz=20.0)
    frames = list(source.frames(3.0))
    deltas = np.diff([f.timestamp_seconds for f in frames])
    assert np.allclose(deltas, 0.05, atol=1e-9)


def test_still_subject_has_no_chest_displacement():
    still = synthetic_source(still=True)
    assert still.subject.respiration_amplitude_m == 0.0
    assert still.subject.heart_amplitude_m == 0.0


def test_subject_rejects_an_unusable_subcarrier_count():
    with pytest.raises(ValueError):
        synthetic_source(n_subcarriers=0)

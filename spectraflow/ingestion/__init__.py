"""CSI ingestion: UDP transport, wire codec and the synthetic frame source."""

from __future__ import annotations

from spectraflow.ingestion.synthetic import Subject, SyntheticCsiSource
from spectraflow.ingestion.udp_receiver import (
    CSI_FLAG_FIRST_WORD_INVALID,
    CSI_FLAG_LAST_WORD_INVALID,
    CSI_HEADER_BYTES,
    CSI_MAX_DATAGRAM_BYTES,
    CSI_MAX_SUBCARRIERS,
    CSI_PACKET_MAGIC,
    CSI_PROTOCOL_VERSION,
    CsiFrame,
    CsiParseError,
    FrameBuffer,
    ReceiverStats,
    UdpCsiReceiver,
    frames_to_batch,
    pack_csi_datagram,
    parse_csi_datagram,
)

__all__ = [
    "CSI_FLAG_FIRST_WORD_INVALID",
    "CSI_FLAG_LAST_WORD_INVALID",
    "CSI_HEADER_BYTES",
    "CSI_MAX_DATAGRAM_BYTES",
    "CSI_MAX_SUBCARRIERS",
    "CSI_PACKET_MAGIC",
    "CSI_PROTOCOL_VERSION",
    "CsiFrame",
    "CsiParseError",
    "FrameBuffer",
    "ReceiverStats",
    "Subject",
    "SyntheticCsiSource",
    "UdpCsiReceiver",
    "frames_to_batch",
    "pack_csi_datagram",
    "parse_csi_datagram",
]

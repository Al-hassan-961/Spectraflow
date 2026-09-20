// Spectraflow — CSI wire-format definition and parsing.
//
// This header is the single source of truth for the binary contract between the
// ESP32 firmware (firmware/esp32_csi_node/main/csi_collector.c), the native
// ingestion core and the Python parser in spectraflow/ingestion/udp_receiver.py.
//
// All multi-byte integers are LITTLE-ENDIAN. The header is 24 bytes and is
// deliberately treated as an unaligned byte stream: we never memcpy a packed
// struct onto the wire, because struct padding/layout is compiler-dependent and
// the 16-bit fields sit at offsets that are not naturally aligned.
//
// Do not relax that. `timestamp_us` at offset 12 is 4-byte aligned but *not*
// 8-byte aligned, so a plain (non-packed) struct would only work on the 32-bit
// firmware target and would silently misparse in a 64-bit host build. Every
// codec on every side is byte-wise on purpose.
//
//   Offset | Size | Type   | Field
//   -------+------+--------+-----------------
//        0 |    2 | uint16 | magic          (always 0x5F43)
//        2 |    1 | uint8  | version        (always 1)
//        3 |    1 | uint8  | flags
//        4 |    2 | uint16 | node_id
//        6 |    2 | uint16 | payload_bytes  (= n_subcarriers * 2)
//        8 |    4 | uint32 | sequence
//       12 |    8 | uint64 | timestamp_us   (sender uptime, microseconds)
//       20 |    1 | uint8  | channel
//       21 |    1 | int8   | rssi_dbm
//       22 |    1 | int8   | noise_floor_dbm
//       23 |    1 | uint8  | n_subcarriers  (1..128)
//       24 |  ... | int8[] | CSI payload: for each subcarrier, imag then real
//
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spectraflow {

/// Magic number at offset 0. Little-endian bytes on the wire are 0x43 0x5F.
inline constexpr std::uint16_t kCsiPacketMagic = 0x5F43;
/// Protocol version in the header.
inline constexpr std::uint8_t kCsiProtocolVersion = 1;
/// Size of the fixed header, in bytes.
inline constexpr std::size_t kCsiHeaderBytes = 24;
/// Upper bound on subcarriers accepted from the wire.
inline constexpr std::size_t kCsiMaxSubcarriers = 128;
/// Largest datagram we will ever accept (keeps us below the 1472-byte MTU
/// budget and therefore clear of IP fragmentation).
inline constexpr std::size_t kCsiMaxDatagramBytes =
    kCsiHeaderBytes + kCsiMaxSubcarriers * 2;

/// Bit flags carried in the header `flags` byte.
enum CsiFlag : std::uint8_t {
  kCsiFlagFirstWordInvalid = 1u << 0,
  kCsiFlagLastWordInvalid = 1u << 1,
};

/// Decoded header of a CSI datagram.
struct CsiHeader {
  std::uint8_t version = kCsiProtocolVersion;
  std::uint8_t flags = 0;
  std::uint16_t node_id = 0;
  std::uint32_t sequence = 0;
  std::uint64_t timestamp_us = 0;
  std::uint16_t n_subcarriers = 0;
  std::uint8_t channel = 0;
  std::int8_t rssi_dbm = 0;
  std::int8_t noise_floor_dbm = 0;
  std::uint16_t payload_bytes = 0;

  bool first_word_invalid() const {
    return (flags & kCsiFlagFirstWordInvalid) != 0;
  }
  bool last_word_invalid() const {
    return (flags & kCsiFlagLastWordInvalid) != 0;
  }
};

/// A fully decoded CSI frame: header plus complex channel estimates.
struct CsiFrame {
  CsiHeader header;
  /// Per-subcarrier complex channel estimate H(f), in subcarrier order.
  std::vector<std::complex<float>> csi;
  /// Number of complex samples that were discarded as invalid lead/trail words.
  std::size_t invalid_edges = 0;

  /// Frame capture time in seconds, taken from the sender clock.
  double timestamp_seconds() const {
    return static_cast<double>(header.timestamp_us) * 1e-6;
  }

  /// Amplitude of each subcarrier in dB (20*log10|H|), floored at -160 dB.
  std::vector<float> amplitude_db() const;
  /// Unwrapped-free raw phase of each subcarrier, in radians.
  std::vector<float> phase_radians() const;
  /// Linear power (|H|^2) per subcarrier.
  std::vector<float> power_linear() const;
};

/// Parse a datagram. Returns false and fills `error` on malformed input.
///
/// `out` is only modified on success, so a partially-parsed frame can never
/// leak into the pipeline.
bool parse_csi_packet(const std::uint8_t* data, std::size_t len, CsiFrame& out,
                      std::string& error);

/// Convenience overload for a byte vector.
bool parse_csi_packet(const std::vector<std::uint8_t>& data, CsiFrame& out,
                      std::string& error);

/// Serialize a frame header + payload into `out` (cleared first).
///
/// Used by the test-suite and by any replay tooling; the firmware has its own
/// byte-level serializer written in C for the constrained target.
void serialize_csi_packet(const CsiHeader& header,
                          const std::vector<std::complex<float>>& csi,
                          std::vector<std::uint8_t>& out);

}  // namespace spectraflow

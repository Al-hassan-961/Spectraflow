// Spectraflow — CSI datagram parsing and serialization.
#include "csi_packet.hpp"

#include <cmath>
#include <cstring>

namespace spectraflow {
namespace {

// Explicit little-endian codecs. We never cast the buffer to a packed struct:
// the header fields sit at odd offsets and struct layout is compiler-defined.
inline std::uint16_t read_u16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (static_cast<std::uint16_t>(p[1]) << 8));
}

inline std::uint32_t read_u32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t read_u64(const std::uint8_t* p) {
  std::uint64_t v = 0;
  for (int i = 7; i >= 0; --i) {
    v = (v << 8) | static_cast<std::uint64_t>(p[i]);
  }
  return v;
}

inline void write_u16(std::uint8_t* p, std::uint16_t v) {
  p[0] = static_cast<std::uint8_t>(v & 0xFF);
  p[1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
}

inline void write_u32(std::uint8_t* p, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    p[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF);
  }
}

inline void write_u64(std::uint8_t* p, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    p[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF);
  }
}

/// Amplitude floor used whenever |H| is zero, so 20*log10 never yields -inf.
constexpr float kMinAmplitude = 1e-8f;

}  // namespace

std::vector<float> CsiFrame::amplitude_db() const {
  std::vector<float> out;
  out.reserve(csi.size());
  for (const auto& h : csi) {
    const float mag = std::abs(h);
    out.push_back(20.0f * std::log10(mag > kMinAmplitude ? mag : kMinAmplitude));
  }
  return out;
}

std::vector<float> CsiFrame::phase_radians() const {
  std::vector<float> out;
  out.reserve(csi.size());
  for (const auto& h : csi) {
    out.push_back(std::arg(h));
  }
  return out;
}

std::vector<float> CsiFrame::power_linear() const {
  std::vector<float> out;
  out.reserve(csi.size());
  for (const auto& h : csi) {
    out.push_back(std::norm(h));
  }
  return out;
}

bool parse_csi_packet(const std::uint8_t* data, std::size_t len, CsiFrame& out,
                      std::string& error) {
  out.csi.clear();
  out.invalid_edges = 0;

  if (data == nullptr) {
    error = "null buffer";
    return false;
  }
  if (len < kCsiHeaderBytes) {
    error = "truncated header (" + std::to_string(len) + " < " +
            std::to_string(kCsiHeaderBytes) + " bytes)";
    return false;
  }

  const std::uint16_t magic = read_u16(data);
  if (magic != kCsiPacketMagic) {
    error = "bad magic";
    return false;
  }

  CsiHeader h;
  h.version = data[2];
  h.flags = data[3];
  h.node_id = read_u16(data + 4);
  h.payload_bytes = read_u16(data + 6);
  h.sequence = read_u32(data + 8);
  h.timestamp_us = read_u64(data + 12);
  h.channel = data[20];
  h.rssi_dbm = static_cast<std::int8_t>(data[21]);
  h.noise_floor_dbm = static_cast<std::int8_t>(data[22]);
  h.n_subcarriers = data[23];

  if (h.version != kCsiProtocolVersion) {
    error = "unsupported version " + std::to_string(h.version);
    return false;
  }
  if (h.n_subcarriers == 0) {
    error = "zero subcarriers";
    return false;
  }
  if (h.n_subcarriers > kCsiMaxSubcarriers) {
    error = "too many subcarriers (" + std::to_string(h.n_subcarriers) + " > " +
            std::to_string(kCsiMaxSubcarriers) + ")";
    return false;
  }

  const std::size_t expected = static_cast<std::size_t>(h.n_subcarriers) * 2u;
  if (h.payload_bytes != expected) {
    error = "payload_bytes mismatch (" + std::to_string(h.payload_bytes) +
            " != " + std::to_string(expected) + ")";
    return false;
  }
  if (len < kCsiHeaderBytes + expected) {
    error = "truncated payload";
    return false;
  }

  const std::uint8_t* payload = data + kCsiHeaderBytes;
  std::vector<std::complex<float>> csi;
  csi.reserve(h.n_subcarriers);
  for (std::size_t i = 0; i < h.n_subcarriers; ++i) {
    // ESP32 convention: buf[2i] is the imaginary part, buf[2i+1] the real part.
    const auto imag = static_cast<std::int8_t>(payload[2 * i]);
    const auto real = static_cast<std::int8_t>(payload[2 * i + 1]);
    csi.emplace_back(static_cast<float>(real), static_cast<float>(imag));
  }

  // The ESP32 CSI engine cannot always populate the first/last OFDM word; those
  // words carry no information, so we excise them rather than let a garbage
  // value blow up the linear phase fit.
  if (h.first_word_invalid() && csi.size() > 2) {
    csi.erase(csi.begin());
    out.invalid_edges += 1;
  }
  if (h.last_word_invalid() && csi.size() > 2) {
    csi.pop_back();
    out.invalid_edges += 1;
  }

  out.header = h;
  out.csi = std::move(csi);
  return true;
}

bool parse_csi_packet(const std::vector<std::uint8_t>& data, CsiFrame& out,
                      std::string& error) {
  return parse_csi_packet(data.data(), data.size(), out, error);
}

void serialize_csi_packet(const CsiHeader& header,
                          const std::vector<std::complex<float>>& csi,
                          std::vector<std::uint8_t>& out) {
  CsiHeader h = header;
  h.version = kCsiProtocolVersion;
  h.n_subcarriers = static_cast<std::uint16_t>(csi.size());
  h.payload_bytes = static_cast<std::uint16_t>(csi.size() * 2u);

  out.assign(kCsiHeaderBytes + csi.size() * 2u, 0u);

  write_u16(out.data() + 0, kCsiPacketMagic);
  out[2] = h.version;
  out[3] = h.flags;
  write_u16(out.data() + 4, h.node_id);
  write_u16(out.data() + 6, h.payload_bytes);
  write_u32(out.data() + 8, h.sequence);
  write_u64(out.data() + 12, h.timestamp_us);
  out[20] = h.channel;
  out[21] = static_cast<std::uint8_t>(h.rssi_dbm);
  out[22] = static_cast<std::uint8_t>(h.noise_floor_dbm);
  out[23] = static_cast<std::uint8_t>(
      csi.size() > 255 ? 255 : static_cast<int>(csi.size()));

  std::uint8_t* payload = out.data() + kCsiHeaderBytes;
  for (std::size_t i = 0; i < csi.size(); ++i) {
    const float re = csi[i].real();
    const float im = csi[i].imag();
    // Clamp to the int8 range the wire format provides.
    const auto ri = static_cast<std::int8_t>(
        re > 127.0f ? 127 : (re < -128.0f ? -128 : static_cast<int>(std::lround(re))));
    const auto ii = static_cast<std::int8_t>(
        im > 127.0f ? 127 : (im < -128.0f ? -128 : static_cast<int>(std::lround(im))));
    payload[2 * i] = static_cast<std::uint8_t>(ii);
    payload[2 * i + 1] = static_cast<std::uint8_t>(ri);
  }
}

}  // namespace spectraflow

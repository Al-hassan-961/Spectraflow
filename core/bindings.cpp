// Spectraflow — pybind11 bindings for the native ingestion + DSP core.
//
// Exposes the C++ primitives as the `spectraflow_core` extension module. The
// Python package (spectraflow/) imports this module when it is available and
// transparently falls back to an equivalent pure-NumPy implementation when it
// is not, so the pipeline runs on hosts where the extension cannot be built.
//
// Callable surface, mirroring the NumPy fallbacks one-for-one:
//   parse_csi_packet / serialize_csi_packet / csi_header_size
//   CsiFrame, CsiHeader
//   sanitize_phase / unwrap_phase / complex_to_phase
//   ClutterRemover
//   BiquadCoeffs / design_butterworth_bandpass / Biquad
//   fft / hann_window / magnitude_spectrum / bin_width_hz
//   SpectralPeak / find_band_peak / snr_to_confidence
//   CsiRingBuffer
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "csi_packet.hpp"
#include "dsp_filters.hpp"
#include "ring_buffer.hpp"

namespace py = pybind11;
using namespace spectraflow;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using ComplexArray =
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>;

/// Copy any 1-D float array-like into a std::vector<float>.
std::vector<float> to_float_vector(const FloatArray& arr) {
  const auto info = arr.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("expected a 1-D array");
  }
  const auto* ptr = static_cast<const float*>(info.ptr);
  return std::vector<float>(ptr, ptr + info.shape[0]);
}

std::vector<std::complex<float>> to_complex_vector(const ComplexArray& arr) {
  const auto info = arr.request();
  if (info.ndim != 1) {
    throw std::invalid_argument("expected a 1-D complex array");
  }
  const auto* ptr = static_cast<const std::complex<float>*>(info.ptr);
  return std::vector<std::complex<float>>(ptr, ptr + info.shape[0]);
}

py::array_t<float> from_float_vector(const std::vector<float>& v) {
  py::array_t<float> out(static_cast<py::ssize_t>(v.size()));
  if (!v.empty()) {
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
  }
  return out;
}

py::array_t<double> from_double_vector(const std::vector<double>& v) {
  py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
  if (!v.empty()) {
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
  }
  return out;
}

py::array_t<std::complex<float>> from_complex_vector(
    const std::vector<std::complex<float>>& v) {
  py::array_t<std::complex<float>> out(static_cast<py::ssize_t>(v.size()));
  if (!v.empty()) {
    std::memcpy(out.mutable_data(), v.data(),
                v.size() * sizeof(std::complex<float>));
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(spectraflow_core, m) {
  m.doc() =
      "Spectraflow native core: CSI datagram codec, lock-free ring buffer and "
      "DSP primitives (phase sanitization, clutter rejection, Butterworth "
      "filtering, FFT and spectral peak scoring).";

  m.attr("CSI_PACKET_MAGIC") = py::int_(kCsiPacketMagic);
  m.attr("CSI_PROTOCOL_VERSION") = py::int_(kCsiProtocolVersion);
  m.attr("CSI_HEADER_BYTES") = py::int_(kCsiHeaderBytes);
  m.attr("CSI_MAX_SUBCARRIERS") = py::int_(kCsiMaxSubcarriers);
  m.attr("CSI_MAX_DATAGRAM_BYTES") = py::int_(kCsiMaxDatagramBytes);

  // -------------------------------------------------------------------------
  // CSI packet codec
  // -------------------------------------------------------------------------
  py::class_<CsiHeader>(m, "CsiHeader")
      .def(py::init<>())
      .def_readwrite("version", &CsiHeader::version)
      .def_readwrite("flags", &CsiHeader::flags)
      .def_readwrite("node_id", &CsiHeader::node_id)
      .def_readwrite("sequence", &CsiHeader::sequence)
      .def_readwrite("timestamp_us", &CsiHeader::timestamp_us)
      .def_readwrite("n_subcarriers", &CsiHeader::n_subcarriers)
      .def_readwrite("channel", &CsiHeader::channel)
      .def_readwrite("rssi_dbm", &CsiHeader::rssi_dbm)
      .def_readwrite("noise_floor_dbm", &CsiHeader::noise_floor_dbm)
      .def_readwrite("payload_bytes", &CsiHeader::payload_bytes)
      .def_property_readonly("first_word_invalid", &CsiHeader::first_word_invalid)
      .def_property_readonly("last_word_invalid", &CsiHeader::last_word_invalid)
      .def("__repr__", [](const CsiHeader& h) {
        return "<CsiHeader node=" + std::to_string(h.node_id) +
               " seq=" + std::to_string(h.sequence) +
               " subcarriers=" + std::to_string(h.n_subcarriers) +
               " rssi=" + std::to_string(static_cast<int>(h.rssi_dbm)) + "dBm>";
      });

  py::class_<CsiFrame>(m, "CsiFrame")
      .def(py::init<>())
      .def_readwrite("header", &CsiFrame::header)
      // `csi` is exposed as a numpy complex64 array in both directions so a
      // frame can be constructed from Python (needed to feed CsiRingBuffer);
      // a std::vector<std::complex<float>> member would only accept a Python
      // list, not the arrays the rest of the pipeline passes around.
      .def_property(
          "csi",
          [](const CsiFrame& f) { return from_complex_vector(f.csi); },
          [](CsiFrame& f, const ComplexArray& arr) {
            f.csi = to_complex_vector(arr);
            f.header.n_subcarriers =
                static_cast<std::uint16_t>(f.csi.size());
            f.header.payload_bytes =
                static_cast<std::uint16_t>(f.csi.size() * 2u);
          })
      .def_readwrite("invalid_edges", &CsiFrame::invalid_edges)
      .def_property_readonly("timestamp_seconds", &CsiFrame::timestamp_seconds)
      .def_property_readonly("n_subcarriers",
                             [](const CsiFrame& f) { return f.csi.size(); })
      .def("amplitude_db",
           [](const CsiFrame& f) { return from_float_vector(f.amplitude_db()); })
      .def("phase_radians",
           [](const CsiFrame& f) { return from_float_vector(f.phase_radians()); })
      .def("power_linear",
           [](const CsiFrame& f) { return from_float_vector(f.power_linear()); })
      .def("csi_array",
           [](const CsiFrame& f) { return from_complex_vector(f.csi); })
      .def("__repr__", [](const CsiFrame& f) {
        return "<CsiFrame node=" + std::to_string(f.header.node_id) +
               " seq=" + std::to_string(f.header.sequence) + " n=" +
               std::to_string(f.csi.size()) + ">";
      });

  m.def(
      "make_csi_frame",
      [](const CsiHeader& header, const ComplexArray& csi) {
        CsiFrame frame;
        frame.header = header;
        frame.csi = to_complex_vector(csi);
        frame.header.n_subcarriers =
            static_cast<std::uint16_t>(frame.csi.size());
        frame.header.payload_bytes =
            static_cast<std::uint16_t>(frame.csi.size() * 2u);
        return frame;
      },
      py::arg("header"), py::arg("csi"),
      "Build a native CsiFrame from a header and a 1-D complex CSI array.");

  m.def(
      "parse_csi_packet",
      [](const py::bytes& payload) {
        // Hold the buffer alive while we parse it.
        const std::string buf = payload;
        CsiFrame frame;
        std::string error;
        if (!parse_csi_packet(reinterpret_cast<const std::uint8_t*>(buf.data()),
                              buf.size(), frame, error)) {
          throw std::runtime_error("malformed CSI packet: " + error);
        }
        return frame;
      },
      py::arg("payload"),
      "Parse a CSI datagram into a CsiFrame. Raises RuntimeError on malformed "
      "input.");

  m.def(
      "try_parse_csi_packet",
      [](const py::bytes& payload) -> std::optional<CsiFrame> {
        const std::string buf = payload;
        CsiFrame frame;
        std::string error;
        if (!parse_csi_packet(reinterpret_cast<const std::uint8_t*>(buf.data()),
                              buf.size(), frame, error)) {
          return std::nullopt;
        }
        return frame;
      },
      py::arg("payload"),
      "Non-throwing variant of parse_csi_packet; returns None when malformed.");

  m.def(
      "serialize_csi_packet",
      [](const CsiHeader& header, const ComplexArray& csi) {
        std::vector<std::uint8_t> out;
        serialize_csi_packet(header, to_complex_vector(csi), out);
        return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
      },
      py::arg("header"), py::arg("csi"),
      "Serialize a header + complex CSI vector into wire bytes.");

  // -------------------------------------------------------------------------
  // Ring buffer
  // -------------------------------------------------------------------------
  py::class_<RingBufferStats>(m, "RingBufferStats")
      .def_readonly("pushed", &RingBufferStats::pushed)
      .def_readonly("popped", &RingBufferStats::popped)
      .def_readonly("dropped", &RingBufferStats::dropped)
      .def_readonly("capacity", &RingBufferStats::capacity)
      .def_readonly("size", &RingBufferStats::size)
      .def("__repr__", [](const RingBufferStats& s) {
        return "<RingBufferStats size=" + std::to_string(s.size) + "/" +
               std::to_string(s.capacity) + " pushed=" +
               std::to_string(s.pushed) + " dropped=" +
               std::to_string(s.dropped) + ">";
      });

  py::class_<CsiRingBuffer>(m, "CsiRingBuffer")
      .def(py::init<std::size_t>(), py::arg("capacity"))
      .def("push", py::overload_cast<const CsiFrame&>(&CsiRingBuffer::push),
           py::arg("frame"),
           "Push a frame; returns False when the buffer is full (frame dropped).")
      .def("pop", &CsiRingBuffer::pop, "Pop the oldest frame, or None if empty.")
      .def("peek_latest", &CsiRingBuffer::peek_latest,
           "Return the newest frame without removing it.")
      .def("drain", &CsiRingBuffer::drain, py::arg("out"), py::arg("max_frames") = 0)
      .def("clear", &CsiRingBuffer::clear)
      .def("size", &CsiRingBuffer::size)
      .def("capacity", &CsiRingBuffer::capacity)
      .def("empty", &CsiRingBuffer::empty)
      .def("full", &CsiRingBuffer::full)
      .def("stats", &CsiRingBuffer::stats)
      .def("__len__", &CsiRingBuffer::size);

  // -------------------------------------------------------------------------
  // Phase sanitization
  // -------------------------------------------------------------------------
  m.def(
      "sanitize_phase",
      [](const FloatArray& phase) {
        return from_float_vector(sanitize_phase(to_float_vector(phase)));
      },
      py::arg("phase"),
      "Linear phase unwrapping removing CFO/SFO ramp and mean offset.");

  m.def(
      "unwrap_phase",
      [](const FloatArray& phase) {
        return from_float_vector(unwrap_phase(to_float_vector(phase)));
      },
      py::arg("phase"), "Unwrap phase so consecutive steps stay within +/- pi.");

  m.def(
      "complex_to_phase",
      [](const ComplexArray& csi) {
        return from_float_vector(complex_to_phase(to_complex_vector(csi)));
      },
      py::arg("csi"), "Raw (unwrapped-free) phase of each subcarrier, in radians.");

  // -------------------------------------------------------------------------
  // Clutter rejection
  // -------------------------------------------------------------------------
  py::class_<ClutterRemover>(m, "ClutterRemover")
      .def(py::init<std::size_t>(), py::arg("window") = 100)
      .def("process",
           [](ClutterRemover& self, const ComplexArray& h) {
             return from_complex_vector(self.process(to_complex_vector(h)));
           },
           py::arg("h"),
           "Return H minus the trailing mean over the configured window.")
      .def("reset", &ClutterRemover::reset)
      .def_property_readonly("window", &ClutterRemover::window)
      .def_property_readonly("count", &ClutterRemover::count)
      .def_property_readonly("warm", &ClutterRemover::warm)
      .def("__repr__", [](const ClutterRemover& c) {
        return "<ClutterRemover window=" + std::to_string(c.window()) +
               " count=" + std::to_string(c.count()) + ">";
      });

  // -------------------------------------------------------------------------
  // IIR filtering
  // -------------------------------------------------------------------------
  py::class_<BiquadCoeffs>(m, "BiquadCoeffs")
      .def(py::init<>())
      .def_readonly("b0", &BiquadCoeffs::b0)
      .def_readonly("b1", &BiquadCoeffs::b1)
      .def_readonly("b2", &BiquadCoeffs::b2)
      .def_readonly("a1", &BiquadCoeffs::a1)
      .def_readonly("a2", &BiquadCoeffs::a2)
      .def("__repr__", [](const BiquadCoeffs& c) {
        return "<BiquadCoeffs b=[" + std::to_string(c.b0) + ", " +
               std::to_string(c.b1) + ", " + std::to_string(c.b2) + "] a=[1, " +
               std::to_string(c.a1) + ", " + std::to_string(c.a2) + "]>";
      });

  m.def("design_butterworth_bandpass", &design_butterworth_bandpass,
        py::arg("f_lo"), py::arg("f_hi"), py::arg("fs"),
        "Design a 2nd-order Butterworth band-pass biquad (bilinear transform "
        "with pre-warping, unity gain at the band centre).");

  py::class_<Biquad>(m, "Biquad")
      .def(py::init<>())
      .def(py::init<const BiquadCoeffs&>(), py::arg("coeffs"))
      .def("process", &Biquad::process, py::arg("x"),
           "Filter one sample, advancing internal state.")
      .def("process_block",
           [](Biquad& self, const FloatArray& x) {
             std::vector<double> xd(x.size());
             const auto* p = static_cast<const float*>(x.request().ptr);
             for (py::ssize_t i = 0; i < x.size(); ++i) {
               xd[static_cast<std::size_t>(i)] = p[i];
             }
             std::vector<double> out;
             self.process_block(xd, out);
             return from_double_vector(out);
           },
           py::arg("x"), "Filter a whole block, returning a float64 array.")
      .def("reset", &Biquad::reset)
      .def_property_readonly("initialised", &Biquad::initialised)
      .def_property_readonly("coeffs", &Biquad::coeffs);

  // -------------------------------------------------------------------------
  // FFT / windows / spectra
  // -------------------------------------------------------------------------
  m.def("is_power_of_two", &is_power_of_two, py::arg("n"));
  m.def("next_power_of_two", &next_power_of_two, py::arg("n"));

  m.def(
      "fft",
      [](ComplexArray data) {
        auto info = data.request();
        if (info.ndim != 1) {
          throw std::invalid_argument("fft expects a 1-D array");
        }
        std::vector<std::complex<double>> buf(
            static_cast<std::size_t>(info.shape[0]));
        const auto* src = static_cast<const std::complex<float>*>(info.ptr);
        for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
          buf[static_cast<std::size_t>(i)] =
              std::complex<double>(src[i].real(), src[i].imag());
        }
        fft(buf);
        py::array_t<std::complex<float>> out(info.shape[0]);
        auto* dst = static_cast<std::complex<float>*>(out.request().ptr);
        for (std::size_t i = 0; i < buf.size(); ++i) {
          dst[i] = std::complex<float>(static_cast<float>(buf[i].real()),
                                       static_cast<float>(buf[i].imag()));
        }
        return out;
      },
      py::arg("data"),
      "In-place style radix-2 FFT; returns a new complex64 array. Length must "
      "be a power of two.");

  m.def(
      "hann_window",
      [](std::size_t n) { return from_double_vector(hann_window(n)); },
      py::arg("n"), "Periodic Hann window of length n.");

  m.def(
      "magnitude_spectrum",
      [](const FloatArray& signal, std::size_t nfft) {
        std::vector<double> sig(signal.size());
        const auto* p = static_cast<const float*>(signal.request().ptr);
        for (py::ssize_t i = 0; i < signal.size(); ++i) {
          sig[static_cast<std::size_t>(i)] = p[i];
        }
        return from_double_vector(magnitude_spectrum(sig, nfft));
      },
      py::arg("signal"), py::arg("nfft"),
      "One-sided Hann-windowed amplitude spectrum, zero-padded to nfft.");

  m.def("bin_width_hz", &bin_width_hz, py::arg("nfft"), py::arg("fs"));

  py::class_<SpectralPeak>(m, "SpectralPeak")
      .def(py::init<>())
      .def_readonly("frequency_hz", &SpectralPeak::frequency_hz)
      .def_readonly("magnitude", &SpectralPeak::magnitude)
      .def_readonly("peak_db", &SpectralPeak::peak_db)
      .def_readonly("snr_db", &SpectralPeak::snr_db)
      .def_readonly("snr_linear", &SpectralPeak::snr_linear)
      .def_readonly("confidence", &SpectralPeak::confidence)
      .def_readonly("valid", &SpectralPeak::valid)
      .def("__repr__", [](const SpectralPeak& p) {
        return "<SpectralPeak f=" + std::to_string(p.frequency_hz) +
               "Hz snr=" + std::to_string(p.snr_db) + "dB conf=" +
               std::to_string(p.confidence) +
               (p.valid ? " valid>" : " invalid>");
      });

  m.def(
      "find_band_peak",
      [](const FloatArray& mag, double bin_hz, double f_lo, double f_hi,
         double snr_reference_db, double min_snr_db, double noise_floor) {
        std::vector<double> m(mag.size());
        const auto* p = static_cast<const float*>(mag.request().ptr);
        for (py::ssize_t i = 0; i < mag.size(); ++i) {
          m[static_cast<std::size_t>(i)] = p[i];
        }
        return find_band_peak(m, bin_hz, f_lo, f_hi, snr_reference_db, min_snr_db,
                              noise_floor);
      },
      py::arg("magnitude"), py::arg("bin_hz"), py::arg("f_lo"), py::arg("f_hi"),
      py::arg("snr_reference_db") = 30.0, py::arg("min_snr_db") = 20.0,
      py::arg("noise_floor") = 0.0,
      "Locate the dominant in-band peak with sub-bin interpolation and score "
      "it against a noise floor. Pass noise_floor > 0 to supply a reference "
      "measured on a broadband spectrum of the same signal.");

  m.def("snr_to_confidence", &snr_to_confidence, py::arg("snr_db"),
        py::arg("reference_db") = 30.0,
        "Logistic SNR(dB) -> confidence in [0, 1].");
}

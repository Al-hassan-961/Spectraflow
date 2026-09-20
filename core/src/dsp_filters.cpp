// Spectraflow — native DSP implementations.
#include "dsp_filters.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace spectraflow {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kMinDb = -300.0;

/// Floor for dB/ratio conversions so log10 never sees zero.
constexpr double kEpsilon = 1e-20;

}  // namespace

// ---------------------------------------------------------------------------
// Phase sanitization
// ---------------------------------------------------------------------------

void sanitize_phase(const float* phase, std::size_t n, float* out) {
  if (phase == nullptr || out == nullptr || n == 0) {
    return;
  }
  if (n == 1) {
    out[0] = 0.0f;
    return;
  }

  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<double>(phase[i]);
  }
  const double mean = sum / static_cast<double>(n);

  // Slope of the linear term introduced by CFO/SFO, estimated from the
  // endpoints of the (already unwrapped) phase response.
  const double slope =
      (static_cast<double>(phase[n - 1]) - static_cast<double>(phase[0])) /
      static_cast<double>(n - 1);

  for (std::size_t i = 0; i < n; ++i) {
    const double v = static_cast<double>(phase[i]) - slope * static_cast<double>(i) - mean;
    out[i] = static_cast<float>(v);
  }
}

std::vector<float> sanitize_phase(const std::vector<float>& phase) {
  std::vector<float> out(phase.size());
  sanitize_phase(phase.data(), phase.size(), out.data());
  return out;
}

std::vector<float> unwrap_phase(const std::vector<float>& phase) {
  std::vector<float> out(phase.size());
  if (phase.empty()) {
    return out;
  }
  const double two_pi = 2.0 * kPi;
  out[0] = phase[0];
  double offset = 0.0;
  for (std::size_t i = 1; i < phase.size(); ++i) {
    const double raw = static_cast<double>(phase[i]) + offset;
    const double prev = static_cast<double>(out[i - 1]);
    double delta = raw - prev;
    // Bring the step into (-pi, pi] so the phase track stays continuous.
    if (delta > kPi) {
      const double k = std::floor((delta + kPi) / two_pi);
      offset -= k * two_pi;
    } else if (delta < -kPi) {
      const double k = std::floor((-delta + kPi) / two_pi);
      offset += k * two_pi;
    }
    out[i] = static_cast<float>(static_cast<double>(phase[i]) + offset);
  }
  return out;
}

std::vector<float> complex_to_phase(const std::vector<std::complex<float>>& csi) {
  std::vector<float> out;
  out.reserve(csi.size());
  for (const auto& h : csi) {
    out.push_back(std::arg(h));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Clutter rejection
// ---------------------------------------------------------------------------

ClutterRemover::ClutterRemover(std::size_t window)
    : window_(std::max<std::size_t>(window, 2)) {}

void ClutterRemover::ensure_capacity(std::size_t n) {
  if (subcarriers_ == n) {
    return;
  }
  // Subcarrier count changed (channel retune): the stored history is no longer
  // comparable, so start clean rather than mixing incompatible vectors.
  subcarriers_ = n;
  sum_.assign(n, std::complex<double>(0.0, 0.0));
  history_.assign(n * window_, std::complex<double>(0.0, 0.0));
  count_ = 0;
  write_idx_ = 0;
}

void ClutterRemover::process_inplace(std::vector<std::complex<float>>& h) {
  const std::size_t n = h.size();
  if (n == 0) {
    return;
  }
  ensure_capacity(n);

  const double denom = count_ > 0 ? static_cast<double>(count_) : 1.0;
  for (std::size_t i = 0; i < n; ++i) {
    // Keep the RAW sample: history must model the static background, so it can
    // never be fed the already-cleaned output (that would drive the background
    // estimate to zero and destroy the cancellation).
    const std::complex<double> orig(static_cast<double>(h[i].real()),
                                    static_cast<double>(h[i].imag()));

    // Trailing mean over the previous `count_` frames, evaluated before the
    // current frame is admitted to the window.
    const std::complex<double> mean =
        count_ > 0
            ? std::complex<double>(sum_[i].real() / denom, sum_[i].imag() / denom)
            : std::complex<double>(0.0, 0.0);

    h[i] = std::complex<float>(
        static_cast<float>(orig.real() - mean.real()),
        static_cast<float>(orig.imag() - mean.imag()));

    // Roll the window: evict the oldest frame once full, then store the raw one.
    std::complex<double>& slot = history_[write_idx_ * n + i];
    if (count_ == window_) {
      sum_[i] -= slot;
    }
    slot = orig;
    sum_[i] += orig;
  }

  if (count_ < window_) {
    ++count_;
  }
  write_idx_ = (write_idx_ + 1) % window_;
}

std::vector<std::complex<float>> ClutterRemover::process(
    const std::vector<std::complex<float>>& h) {
  std::vector<std::complex<float>> out = h;
  process_inplace(out);
  return out;
}

void ClutterRemover::reset() {
  std::fill(sum_.begin(), sum_.end(), std::complex<double>(0.0, 0.0));
  std::fill(history_.begin(), history_.end(), std::complex<double>(0.0, 0.0));
  count_ = 0;
  write_idx_ = 0;
}

// ---------------------------------------------------------------------------
// IIR band-pass filtering
// ---------------------------------------------------------------------------

BiquadCoeffs design_butterworth_bandpass(double f_lo, double f_hi, double fs) {
  if (!(fs > 0.0)) {
    throw std::invalid_argument("design_butterworth_bandpass: fs must be > 0");
  }
  if (!(f_lo > 0.0)) {
    throw std::invalid_argument("design_butterworth_bandpass: f_lo must be > 0");
  }
  if (!(f_hi > f_lo)) {
    throw std::invalid_argument("design_butterworth_bandpass: require f_hi > f_lo");
  }
  const double nyquist = fs / 2.0;
  if (f_hi >= nyquist) {
    throw std::invalid_argument(
        "design_butterworth_bandpass: f_hi must be below the Nyquist frequency");
  }

  // Bilinear transform with frequency pre-warping, then the low-pass -> band-pass
  // substitution s -> (s^2 + w0^2) / (s * BW) on a 1st-order Butterworth
  // prototype. The result is a 2nd-order band-pass with unity gain at w0:
  //
  //   H(z) = ( K * (1 - z^-2) ) / ( a0 + a1 z^-1 + a2 z^-2 )
  //
  // with K = BW, w0sq = Wa_lo * Wa_hi, and
  //   a0 = 1 + K + w0sq
  //   a1 = -2 + 2 * w0sq
  //   a2 = 1 - K + w0sq
  const double wa_lo = std::tan(kPi * f_lo / fs);
  const double wa_hi = std::tan(kPi * f_hi / fs);
  const double bw = wa_hi - wa_lo;
  const double w0sq = wa_lo * wa_hi;

  const double a0 = 1.0 + bw + w0sq;
  const double a1 = -2.0 + 2.0 * w0sq;
  const double a2 = 1.0 - bw + w0sq;

  BiquadCoeffs c;
  c.b0 = bw / a0;
  c.b1 = 0.0;
  c.b2 = -bw / a0;
  c.a1 = a1 / a0;
  c.a2 = a2 / a0;
  return c;
}

double Biquad::process(double x) {
  // Transposed direct form II: two state variables, best numerical behaviour
  // for a single-precision-hostile recursive filter like this one.
  const double y = c_.b0 * x + z1_;
  z1_ = c_.b1 * x - c_.a1 * y + z2_;
  z2_ = c_.b2 * x - c_.a2 * y;
  initialised_ = true;
  return y;
}

void Biquad::process_block(const std::vector<double>& x, std::vector<double>& out) {
  out.clear();
  out.reserve(x.size());
  for (const double v : x) {
    out.push_back(process(v));
  }
}

// ---------------------------------------------------------------------------
// FFT / windows
// ---------------------------------------------------------------------------

bool is_power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

std::size_t next_power_of_two(std::size_t n) {
  std::size_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

void fft(std::vector<std::complex<double>>& data) {
  const std::size_t n = data.size();
  if (n <= 1) {
    return;
  }
  if (!is_power_of_two(n)) {
    throw std::invalid_argument("fft: length " + std::to_string(n) +
                                " is not a power of two");
  }

  // Bit-reversal permutation.
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      std::swap(data[i], data[j]);
    }
  }

  // Iterative radix-2 Cooley-Tukey butterflies.
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * kPi / static_cast<double>(len);
    const std::complex<double> wlen(std::cos(ang), std::sin(ang));
    for (std::size_t i = 0; i < n; i += len) {
      std::complex<double> w(1.0, 0.0);
      for (std::size_t k = 0; k < len / 2; ++k) {
        const std::complex<double> u = data[i + k];
        const std::complex<double> v = data[i + k + len / 2] * w;
        data[i + k] = u + v;
        data[i + k + len / 2] = u - v;
        w *= wlen;
      }
    }
  }
}

std::vector<double> hann_window(std::size_t n) {
  std::vector<double> w(n, 1.0);
  if (n <= 1) {
    return w;
  }
  // Periodic (not symmetric) Hann: the correct choice for spectral analysis
  // with an FFT, because it keeps the window periodic in the DFT's frame.
  for (std::size_t i = 0; i < n; ++i) {
    w[i] = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) /
                                 static_cast<double>(n)));
  }
  return w;
}

double bin_width_hz(std::size_t nfft, double fs) {
  if (nfft == 0) {
    return 0.0;
  }
  return fs / static_cast<double>(nfft);
}

std::vector<double> magnitude_spectrum(const std::vector<double>& signal,
                                       std::size_t nfft) {
  const std::size_t n = next_power_of_two(std::max<std::size_t>(nfft, 2));
  const std::size_t half = n / 2 + 1;
  std::vector<double> out(half, 0.0);
  if (signal.empty()) {
    return out;
  }

  const std::vector<double> win = hann_window(signal.size());
  std::vector<std::complex<double>> buf(n, std::complex<double>(0.0, 0.0));

  double coherent_gain = 0.0;
  const std::size_t m = std::min(signal.size(), n);
  for (std::size_t i = 0; i < m; ++i) {
    buf[i] = std::complex<double>(signal[i] * win[i], 0.0);
    coherent_gain += win[i];
  }
  if (coherent_gain <= kEpsilon) {
    return out;
  }

  fft(buf);

  // One-sided amplitude spectrum, normalised by the window's coherent gain so
  // magnitudes are comparable across different window lengths.
  const double scale = 2.0 / coherent_gain;
  for (std::size_t i = 0; i < half; ++i) {
    out[i] = std::abs(buf[i]) * scale;
  }
  // DC and Nyquist bins are not mirrored, so they must not be doubled.
  out[0] *= 0.5;
  if (n / 2 < half) {
    out[n / 2] *= 0.5;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Spectral peak / confidence
// ---------------------------------------------------------------------------

double snr_to_confidence(double snr_db, double reference_db) {
  // Logistic ramp centred on `reference_db`. The 5 dB scale spans the ~15-45 dB
  // range that a broadband-referenced peak occupies, so the bar is responsive
  // across the whole useful range rather than saturating at either end.
  const double slope = 5.0;
  const double x = (snr_db - reference_db) / slope;
  // Guard against overflow in exp() for extreme inputs.
  if (x > 60.0) {
    return 1.0;
  }
  if (x < -60.0) {
    return 0.0;
  }
  return 1.0 / (1.0 + std::exp(-x));
}

SpectralPeak find_band_peak(const std::vector<double>& mag, double bin_hz,
                            double f_lo, double f_hi, double snr_reference_db,
                            double min_snr_db, double noise_floor_override) {
  SpectralPeak peak;
  if (mag.empty() || bin_hz <= 0.0 || !(f_hi > f_lo)) {
    return peak;
  }

  const std::size_t lo_bin =
      static_cast<std::size_t>(std::max(0.0, std::ceil(f_lo / bin_hz)));
  const std::size_t hi_bin =
      static_cast<std::size_t>(std::min<double>(
          static_cast<double>(mag.size() - 1), std::floor(f_hi / bin_hz)));
  if (lo_bin > hi_bin || lo_bin >= mag.size()) {
    return peak;
  }

  // Peak inside the band.
  std::size_t peak_bin = lo_bin;
  for (std::size_t i = lo_bin; i <= hi_bin; ++i) {
    if (mag[i] > mag[peak_bin]) {
      peak_bin = i;
    }
  }

  // Noise reference. When the caller supplies one (measured on a broadband
  // spectrum of the same signal) it is a true reference. Otherwise derive one
  // from this spectrum -- but note that a caller which band-passed first has
  // already attenuated these bins, which inflates the ratio for any input.
  double noise_floor = 0.0;
  if (noise_floor_override > 0.0) {
    noise_floor = noise_floor_override;
  } else {
    std::vector<double> noise;
    noise.reserve(hi_bin - lo_bin + 1);
    const std::size_t guard_lo = peak_bin > 0 ? peak_bin - 1 : 0;
    const std::size_t guard_hi = peak_bin + 1;
    for (std::size_t i = lo_bin; i <= hi_bin; ++i) {
      if (i >= guard_lo && i <= guard_hi) {
        continue;
      }
      noise.push_back(mag[i]);
    }
    if (noise.size() < 3) {
      // Too few in-band reference bins (a very narrow band): fall back to the
      // out-of-band floor, which is better than no reference at all.
      noise.clear();
      for (std::size_t i = 1; i < mag.size(); ++i) {
        if (i < lo_bin || i > hi_bin) {
          noise.push_back(mag[i]);
        }
      }
    }
    if (!noise.empty()) {
      const std::size_t mid = noise.size() / 2;
      std::nth_element(noise.begin(), noise.begin() + static_cast<long>(mid),
                       noise.end());
      noise_floor = noise[mid];
    }
  }

  const double peak_mag = mag[peak_bin];
  if (peak_mag <= kEpsilon) {
    // A flat/empty spectrum has no peak at all. Reporting the first in-band bin
    // as though it were the peak location would be misleading, so return the
    // default (invalid, zero-frequency) result instead.
    return peak;
  }
  const double noise_pow = std::max(noise_floor * noise_floor, kEpsilon);
  const double peak_pow = peak_mag * peak_mag;
  const double snr_db = 10.0 * std::log10(std::max(peak_pow, kEpsilon) / noise_pow);

  // Sub-bin refinement by fitting a parabola through the log magnitudes of the
  // three bins around the peak. Without this the frequency resolution is
  // quantised to the bin width, which at a 5 s window is 0.2 Hz -- far too
  // coarse to distinguish 14 from 18 breaths/min.
  double refined_bin = static_cast<double>(peak_bin);
  if (peak_bin > 0 && peak_bin + 1 < mag.size()) {
    const double a = std::log(std::max(mag[peak_bin - 1], kEpsilon));
    const double b = std::log(std::max(mag[peak_bin], kEpsilon));
    const double c = std::log(std::max(mag[peak_bin + 1], kEpsilon));
    const double denom = a - 2.0 * b + c;
    if (std::abs(denom) > 1e-12) {
      const double delta = 0.5 * (a - c) / denom;
      if (delta > -0.5 && delta < 0.5) {
        refined_bin += delta;
      }
    }
  }

  peak.frequency_hz = refined_bin * bin_hz;
  peak.magnitude = peak_mag;
  peak.peak_db = peak_mag > kEpsilon ? 20.0 * std::log10(peak_mag) : kMinDb;
  peak.snr_db = snr_db;
  peak.snr_linear = std::pow(10.0, snr_db / 10.0);
  peak.confidence = snr_to_confidence(snr_db, snr_reference_db);
  peak.valid = snr_db >= min_snr_db;
  return peak;
}

}  // namespace spectraflow

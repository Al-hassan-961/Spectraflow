// Spectraflow — native DSP primitives.
//
// These are the building blocks shared by the phase-sanitization, clutter
// rejection and vital-sign extraction stages of the Python pipeline. They live
// in C++ because the STFT + Butterworth chain runs on every frame at 20-100 Hz
// and has to fit inside the thermal budget of a mobile ARM core.
//
// Every routine here has a numerically equivalent pure-NumPy implementation in
// spectraflow/dsp/. The Python package prefers this native core when the
// extension is importable and transparently falls back to NumPy otherwise, so
// behaviour is identical on hosts where the extension cannot be built.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "csi_packet.hpp"

namespace spectraflow {

// ---------------------------------------------------------------------------
// Phase sanitization
// ---------------------------------------------------------------------------

/// Linear phase unwrapping across subcarriers, removing the linear ramp that
/// Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO) impose on
/// the raw per-subcarrier phase:
///
///   phase_sanitized[i] = phase[i] - ((phase[N-1] - phase[0]) / (N - 1)) * i
///                                - mean(phase)
///
/// This is the classic linear-transform sanitization used by Wi-Fi sensing
/// systems before any phase-based displacement estimate.
///
/// @param phase Raw unwrapped phase, length N >= 2.
/// @param n     Number of subcarriers.
/// @param out   Destination, length >= n. May alias `phase`.
void sanitize_phase(const float* phase, std::size_t n, float* out);

/// std::vector convenience overload.
std::vector<float> sanitize_phase(const std::vector<float>& phase);

/// Unwrap a raw phase vector so consecutive samples never jump by more than
/// +/- pi, making the linear model above well-posed.
std::vector<float> unwrap_phase(const std::vector<float>& phase);

/// Raw complex -> phase for each subcarrier, in radians, without unwrapping.
std::vector<float> complex_to_phase(const std::vector<std::complex<float>>& csi);

// ---------------------------------------------------------------------------
// Clutter rejection
// ---------------------------------------------------------------------------

/// Rolling dynamic background cancellation.
///
///   H_cleaned(f, t) = H(f, t) - mean(H(f, t - m))   for m in 1..M
///
/// The static multipath environment (walls, furniture) dominates the raw CSI by
/// orders of magnitude; subtracting the trailing mean over a configurable
/// window M is what exposes the small perturbations caused by a breathing
/// chest or a moving limb. The window must be long compared to the vital-sign
/// period but short compared to environmental drift.
class ClutterRemover {
 public:
  /// @param window Frames in the averaging window (M). Clamped to >= 2;
  ///               the pipeline default is 100.
  explicit ClutterRemover(std::size_t window = 100);

  /// Subtract the trailing mean. Adapts automatically when the subcarrier
  /// count changes (e.g. the node retunes to a different channel width).
  std::vector<std::complex<float>> process(
      const std::vector<std::complex<float>>& h);

  /// In-place variant; avoids an allocation per frame in the hot path.
  void process_inplace(std::vector<std::complex<float>>& h);

  void reset();

  std::size_t window() const { return window_; }
  std::size_t count() const { return count_; }
  bool warm() const { return count_ >= window_; }

 private:
  void ensure_capacity(std::size_t n);

  std::size_t window_;
  std::size_t count_ = 0;
  std::size_t write_idx_ = 0;
  std::size_t subcarriers_ = 0;
  std::vector<std::complex<double>> sum_;
  std::vector<std::complex<double>> history_;
};

// ---------------------------------------------------------------------------
// IIR band-pass filtering (Butterworth)
// ---------------------------------------------------------------------------

/// Direct-form-I biquad section.
struct BiquadCoeffs {
  double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
};

/// Design a 2nd-order Butterworth band-pass section via the bilinear transform
/// with frequency pre-warping.
///
/// @param f_lo Low cut-off in Hz (exclusive, > 0).
/// @param f_hi High cut-off in Hz (< fs/2).
/// @param fs   Sampling rate in Hz.
/// @throws std::invalid_argument on an unsatisfiable band.
BiquadCoeffs design_butterworth_bandpass(double f_lo, double f_hi, double fs);

/// Single-section IIR filter with persistent state, so it can run sample by
/// sample across frames without discontinuities at frame boundaries.
class Biquad {
 public:
  Biquad() = default;
  explicit Biquad(const BiquadCoeffs& c) : c_(c) {}

  /// Filter one sample and advance the internal state.
  double process(double x);

  /// Filter a whole block, appending to `out` (cleared first).
  void process_block(const std::vector<double>& x, std::vector<double>& out);

  void set_coeffs(const BiquadCoeffs& c) { c_ = c; }
  const BiquadCoeffs& coeffs() const { return c_; }
  void reset() { z1_ = z2_ = 0.0; }
  bool initialised() const { return initialised_; }
  void set_initialised(bool v) { initialised_ = v; }

 private:
  BiquadCoeffs c_{};
  double z1_ = 0.0;
  double z2_ = 0.0;
  bool initialised_ = false;
};

// ---------------------------------------------------------------------------
// FFT / windows
// ---------------------------------------------------------------------------

/// In-place radix-2 decimation-in-time Cooley-Tukey FFT.
/// @throws std::invalid_argument when the length is not a power of two.
void fft(std::vector<std::complex<double>>& data);

/// True when `n` is a power of two.
bool is_power_of_two(std::size_t n);

/// Smallest power of two >= n.
std::size_t next_power_of_two(std::size_t n);

/// Periodic Hann window of length `n`.
std::vector<double> hann_window(std::size_t n);

/// One-sided magnitude spectrum of a real signal, zero-padded to `nfft`
/// (rounded up to a power of two) and Hann-windowed.
/// Length of the result is nfft/2 + 1.
std::vector<double> magnitude_spectrum(const std::vector<double>& signal,
                                       std::size_t nfft);

/// Frequency spacing, in Hz, of a spectrum of `nfft` points at rate `fs`.
double bin_width_hz(std::size_t nfft, double fs);

// ---------------------------------------------------------------------------
// Spectral peak / confidence
// ---------------------------------------------------------------------------

/// Result of a band-limited spectral peak search.
struct SpectralPeak {
  double frequency_hz = 0.0;   ///< Peak location inside the search band.
  double magnitude = 0.0;      ///< Raw magnitude at the peak bin.
  double peak_db = -300.0;     ///< 20*log10(magnitude), floored.
  double snr_db = 0.0;         ///< Peak power vs the out-of-band noise floor.
  double snr_linear = 0.0;     ///< 10^(snr_db/10).
  /// Confidence in [0, 1] derived from `snr_db` via a smooth logistic ramp;
  /// this is what the UI shows as the signal-confidence bar.
  double confidence = 0.0;
  bool valid = false;
};

/// Locate the dominant spectral peak inside [f_lo, f_hi] and score it against a
/// noise floor.
///
/// @param mag    One-sided magnitude spectrum from magnitude_spectrum().
/// @param bin_hz Bin spacing in Hz.
/// @param f_lo   Lower edge of the search band (inclusive).
/// @param f_hi   Upper edge of the search band (inclusive).
/// @param snr_reference_db SNR (dB) at which confidence reaches ~0.5; the
///                          logistic slope is derived from this.
/// @param min_snr_db Peaks below this SNR are reported with valid == false.
/// @param noise_floor Explicit noise-floor *magnitude* to score against.
///                    Pass <= 0 to derive one from `mag` itself.
///
/// The explicit floor exists because callers band-pass before taking the
/// spectrum: the out-of-band bins of a filtered spectrum are already attenuated
/// by the filter, so deriving the floor from them inflates the SNR for every
/// input, including pure noise. A floor measured on a *broadband* spectrum of
/// the same signal is a true reference -- it separates a real vital sign
/// (~40 dB) from an empty room (~15 dB) by a wide margin, whereas the in-band
/// ratio does not separate them at all.
SpectralPeak find_band_peak(const std::vector<double>& mag, double bin_hz,
                            double f_lo, double f_hi,
                            double snr_reference_db = 30.0,
                            double min_snr_db = 20.0,
                            double noise_floor = 0.0);

/// Logistic map from SNR (dB) to a confidence in [0, 1].
double snr_to_confidence(double snr_db, double reference_db = 30.0);

}  // namespace spectraflow

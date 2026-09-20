"""DSP tests: phase sanitization, clutter rejection, filtering and vitals.

Physics under test, not just code paths:

* the linear phase transform removes the CFO/SFO ramp while preserving the
  non-linear residual that carries displacement information;
* clutter cancellation removes a static multipath background that is orders of
  magnitude larger than the signal of interest;
* the Butterworth band-pass has unity gain at its centre and rejects outside;
* the spectral estimator recovers a planted frequency to sub-bin accuracy;
* the vital-sign estimator recovers respiration rate, and -- critically --
  reports *nothing* for an empty room instead of a confident wrong number.

The last point is the one that matters most. An estimator that always returns a
plausible number is worse than useless in a health context, so several tests
below assert that the pipeline abstains.
"""

from __future__ import annotations

import numpy as np
import pytest

from csi_fixtures import synthetic_source
from spectraflow._native import HAVE_NATIVE, NATIVE
from spectraflow.config import default_config
from spectraflow.dsp import _backend, _numpy_dsp
from spectraflow.dsp.clutter_removal import ClutterRemover
from spectraflow.dsp.phase_sanitizer import (
    PhaseSanitizer,
    sanitize_phase,
    unwrap_phase,
)
from spectraflow.dsp.vitals_extractor import VitalSignsExtractor
from spectraflow.ingestion import Subject


def _drain(source, seconds, **config_changes):
    """Run ``seconds`` of simulated CSI through the vitals estimator."""
    config = default_config()
    if config_changes:
        config = config.replace(**config_changes)
    extractor = VitalSignsExtractor(config)
    result = None
    for frame in source.frames(seconds):
        result = extractor.update(frame)
    return result


# ---------------------------------------------------------------------------
# Phase sanitization
# ---------------------------------------------------------------------------


def test_sanitize_removes_a_linear_ramp():
    """The defining property: the output has no endpoint-to-endpoint slope."""
    truth = np.linspace(-0.3, 0.9, 64).astype(np.float32)
    planted = truth + 0.7 * np.arange(64) + 1.3  # CFO/SFO ramp + constant offset

    cleaned = sanitize_phase(planted)
    slope = (cleaned[-1] - cleaned[0]) / (cleaned.size - 1)
    assert abs(slope) < 1e-4, f"ramp not removed (residual slope {slope})"


def test_sanitize_is_invariant_to_an_added_constant():
    phase = np.linspace(0.0, 2.0, 32).astype(np.float32)
    assert np.allclose(sanitize_phase(phase), sanitize_phase(phase + 17.0), atol=1e-5)


def test_sanitize_accepts_complex_csi_and_uses_its_angle():
    """Regression: a dtype cast would silently keep the real part instead.

    ``np.asarray(complex_array, dtype=np.float32)`` discards the imaginary
    component without raising, so a caller passing complex CSI would have been
    given the channel's in-phase component rather than its phase.
    """
    csi = np.exp(1j * np.linspace(0, 1.5, 40)).astype(np.complex64)
    from_complex = sanitize_phase(csi)
    from_phase = sanitize_phase(np.angle(csi).astype(np.float32))
    assert np.allclose(from_complex, from_phase, atol=1e-5)


def test_unwrap_makes_a_wrapped_track_continuous():
    truth = np.linspace(0.0, 4.0 * np.pi, 200)
    wrapped = np.angle(np.exp(1j * truth)).astype(np.float32)
    assert np.ptp(wrapped) <= 2 * np.pi + 1e-6, "test input should be wrapped"

    unwrapped = unwrap_phase(wrapped)
    steps = np.abs(np.diff(unwrapped))
    assert steps.max() < np.pi, "unwrapped track must not jump by more than pi"


def test_sanitizer_class_handles_complex_and_reports_state():
    sanitizer = PhaseSanitizer()
    phase = sanitizer.process(np.ones(16, dtype=np.complex64) * (1 + 1j))
    assert phase.shape == (16,)
    assert sanitizer.frames_processed == 1

    sanitizer.reset()
    assert sanitizer.frames_processed == 0


@pytest.mark.parametrize("n", [0, 1, 2])
def test_sanitize_handles_degenerate_lengths(n):
    result = sanitize_phase(np.zeros(n, dtype=np.float32))
    assert result.shape == (n,)


# ---------------------------------------------------------------------------
# Clutter removal
# ---------------------------------------------------------------------------


def test_clutter_remover_cancels_a_static_background():
    """A static reflector must vanish; the dynamic term must survive."""
    remover = ClutterRemover(window=50)
    static = np.full(64, 3.0 + 4.0j, dtype=np.complex64)  # |H| = 5.0

    out = None
    for i in range(60):
        dynamic = 0.5 * np.exp(1j * 2 * np.pi * 0.3 * i / 20)
        out = remover.process(static + dynamic)

    # |static| is 5.0; the residual must be the order of the 0.5 dynamic term.
    assert abs(out[0]) < 1.0, f"background not cancelled (|residual|={abs(out[0])})"
    assert remover.warm is True
    assert remover.count == 50


def test_clutter_remover_window_must_be_sane():
    with pytest.raises(ValueError):
        ClutterRemover(window=1)


def test_clutter_remover_resets_when_the_subcarrier_count_changes():
    remover = ClutterRemover(window=8)
    for _ in range(10):
        remover.process(np.ones(32, dtype=np.complex64))
    assert remover.count == 8

    remover.process(np.ones(64, dtype=np.complex64))
    assert remover.count == 1, "history must restart on a channel retune"


def test_clutter_remover_does_not_feed_back_its_own_output():
    """Regression: storing the *cleaned* frame drives the background to zero.

    If the history held cleaned values, the trailing mean would collapse and the
    cancellation would stop working after a few frames.
    """
    remover = ClutterRemover(window=10)
    static = np.full(8, 10.0 + 0.0j, dtype=np.complex64)

    for _ in range(40):
        out = remover.process(static)

    assert abs(out[0]) < 1e-3, "a perfectly static channel must cancel to ~0"


def test_clutter_removal_must_not_be_applied_before_phase_extraction():
    """Pins a *measured* design decision, not a stylistic one.

    Subtracting the rolling complex mean is the right way to expose amplitude
    motion, but it is the wrong preprocessing step for a phase-based estimator.
    The phase of a noisy complex sample has an error proportional to ``1/|H|``,
    and clutter removal shrinks ``|H|`` by an order of magnitude while leaving
    the absolute noise floor untouched -- so ``arg(H)`` becomes far noisier even
    though the static reflector is gone.

    Feeding clutter-removed CSI into the vitals pipeline was measured to drop
    subject respiration SNR from ~18 dB to ~1 dB and make the rate
    undetectable. The estimator therefore removes static clutter by detrending
    *within* the analysis window, which does not shrink the magnitude used to
    form the phase.
    """
    rng = np.random.default_rng(0)
    fs, f, n = 20.0, 0.27, 400
    t = np.arange(n) / fs
    noiseless = 12.0 + 0.6 * np.exp(1j * 2 * np.pi * f * t)
    noisy = noiseless + (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.3

    def clean(sequence):
        remover = ClutterRemover(window=100)
        return np.array([remover.process(np.array([v]))[0] for v in sequence])

    cleaned_noisy = clean(noisy)
    cleaned_noiseless = clean(noiseless)
    tail = slice(150, n)

    # The magnitude collapses...
    assert np.abs(cleaned_noisy[tail]).mean() < 0.15 * np.abs(noisy[tail]).mean()

    # ...while the phase error grows by more than an order of magnitude.
    raw_error = np.std(np.angle(noisy[tail]) - np.angle(noiseless[tail]))
    cleaned_error = np.std(np.angle(cleaned_noisy[tail]) - np.angle(cleaned_noiseless[tail]))
    assert cleaned_error > 10 * raw_error, (
        f"expected clutter removal to degrade phase accuracy, "
        f"got {cleaned_error:.4f} vs {raw_error:.4f} rad"
    )


# ---------------------------------------------------------------------------
# Butterworth band-pass
# ---------------------------------------------------------------------------


def _steady_state_gain(biquad: "_numpy_dsp.Biquad", frequency: float, fs: float) -> float:
    """Amplitude gain at ``frequency`` measured through the real filter."""
    seconds = 300.0
    n = int(seconds * fs)
    t = np.arange(n) / fs
    out = biquad.process_block(np.sin(2 * np.pi * frequency * t))
    tail = out[-n // 4 :]
    return float(np.sqrt(2.0) * np.sqrt(np.mean(tail**2)))  # RMS -> amplitude


@pytest.mark.parametrize(
    "band",
    [(0.1, 0.5), (0.8, 2.5)],
)
def test_butterworth_bandpass_has_unity_gain_at_centre(band):
    fs = 20.0
    coeffs = _backend.design_bandpass(band[0], band[1], fs)
    centre = float(np.sqrt(band[0] * band[1]))
    gain = _steady_state_gain(_numpy_dsp.Biquad(*coeffs), centre, fs)
    assert 0.85 < gain < 1.15, f"gain at band centre should be ~1, got {gain}"


def test_butterworth_rejects_outside_the_band():
    fs = 20.0
    coeffs = _backend.design_bandpass(0.8, 2.5, fs)
    centre_gain = _steady_state_gain(_numpy_dsp.Biquad(*coeffs), 1.414, fs)
    low_gain = _steady_state_gain(_numpy_dsp.Biquad(*coeffs), 0.05, fs)
    high_gain = _steady_state_gain(_numpy_dsp.Biquad(*coeffs), 8.0, fs)

    assert low_gain < centre_gain / 5.0, "0.05 Hz must be strongly attenuated"
    assert high_gain < centre_gain / 5.0, "8 Hz must be strongly attenuated"


@pytest.mark.parametrize(
    "band",
    [(0.0, 0.5), (0.5, 0.1), (0.1, 10.0)],
)
def test_butterworth_rejects_unsatisfiable_bands(band):
    with pytest.raises(ValueError):
        _numpy_dsp.design_butterworth_bandpass(band[0], band[1], 20.0)


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_butterworth_design_matches_between_backends():
    for band in [(0.1, 0.5), (0.8, 2.5), (0.2, 3.0)]:
        native = _backend.design_bandpass(*band, 20.0)
        reference = _numpy_dsp.design_butterworth_bandpass(band[0], band[1], 20.0)
        assert np.allclose(native, reference, atol=1e-12), band


def test_biquad_streaming_matches_block_processing():
    coeffs = _backend.design_bandpass(0.1, 0.5, 20.0)
    signal = np.sin(2 * np.pi * 0.27 * np.arange(200) / 20.0)

    block = _numpy_dsp.Biquad(*coeffs).process_block(signal)
    stream = []
    biquad = _numpy_dsp.Biquad(*coeffs)
    for value in signal:
        stream.append(biquad.process(float(value)))
    assert np.allclose(block, np.asarray(stream))


# ---------------------------------------------------------------------------
# FFT and spectral estimation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_fft_matches_numpy():
    """The native radix-2 FFT must agree with NumPy's transform."""
    x = np.random.default_rng(0).normal(size=256).astype(np.complex64)
    expected = np.fft.fft(x.astype(np.complex128))
    got = np.asarray(NATIVE.fft(x))
    assert np.abs(got - expected).max() < 1e-3


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_native_fft_rejects_a_non_power_of_two_length():
    with pytest.raises(Exception):
        NATIVE.fft(np.ones(100, dtype=np.complex64))


def test_next_power_of_two_and_bin_width():
    assert _numpy_dsp.next_power_of_two(1) == 1
    assert _numpy_dsp.next_power_of_two(100) == 128
    assert _numpy_dsp.next_power_of_two(256) == 256
    assert _backend.bin_width_hz(256, 20.0) == pytest.approx(20.0 / 256)


def test_hann_window_is_the_periodic_variant():
    """The periodic (not symmetric) Hann is the correct choice for FFT work."""
    n = 64
    window = _numpy_dsp.hann_window(n)
    expected = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / n))

    assert np.allclose(window, expected)
    assert window[0] == pytest.approx(0.0, abs=1e-12)
    assert window[n // 2] == pytest.approx(1.0, abs=1e-12)
    # A symmetric window is not periodic in the DFT's frame and leaks energy, so
    # the asymmetry here is a feature, not an artefact.
    assert not np.allclose(window, window[::-1])


def test_magnitude_spectrum_peaks_at_the_input_frequency():
    fs = 20.0
    signal = np.sin(2 * np.pi * 1.2 * np.arange(400) / fs)
    spectrum = _backend.magnitude_spectrum(signal, 512)
    bin_hz = _backend.bin_width_hz(512, fs)
    peak = int(np.argmax(spectrum))
    assert abs(peak * bin_hz - 1.2) < 2 * bin_hz
    # A unit-amplitude sinusoid should read back as ~1.0.
    assert 0.8 < spectrum[peak] < 1.2


def test_find_band_peak_recovers_a_planted_frequency():
    fs = 20.0
    signal = np.sin(2 * np.pi * 0.27 * np.arange(512) / fs)
    spectrum = _backend.magnitude_spectrum(signal, 512)
    peak = _backend.find_band_peak(
        spectrum, _backend.bin_width_hz(512, fs), 0.1, 0.5
    )
    assert peak.valid is True
    # Sub-bin interpolation must beat the 0.039 Hz bin spacing.
    assert abs(peak.frequency_hz - 0.27) < 0.02, peak.frequency_hz
    assert peak.confidence > 0.5


def test_confidence_increases_with_snr():
    low = _backend.snr_to_confidence(10.0)
    mid = _backend.snr_to_confidence(15.0)
    high = _backend.snr_to_confidence(25.0)
    assert low < mid < high
    assert 0.0 <= low and high <= 1.0


def test_find_band_peak_abstains_on_an_empty_band():
    peak = _backend.find_band_peak(np.zeros(64), 0.078, 0.1, 0.5)
    assert peak.valid is False
    assert peak.frequency_hz == 0.0


def test_an_explicit_noise_floor_is_used_when_supplied():
    """The caller-supplied broadband floor must override any internal estimate."""
    spectrum = np.ones(64)
    spectrum[20] = 50.0

    loose = _backend.find_band_peak(spectrum, 0.078, 0.1, 2.0, noise_floor=1.0)
    strict = _backend.find_band_peak(spectrum, 0.078, 0.1, 2.0, noise_floor=40.0)

    assert loose.snr_db > strict.snr_db
    assert loose.valid is True and strict.valid is False


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_backends_agree_on_band_peak_scoring():
    rng = np.random.default_rng(5)
    spectrum = np.abs(rng.normal(size=256)) + 0.1
    spectrum[80] = 9.0
    kwargs = dict(
        bin_hz=0.078, f_lo=0.1, f_hi=0.5,
        snr_reference_db=15.0, min_snr_db=7.0, noise_floor=0.3,
    )
    native = _backend.find_band_peak(spectrum, **kwargs)
    reference = _numpy_dsp.find_band_peak(spectrum, **kwargs)

    assert native.frequency_hz == pytest.approx(reference.frequency_hz, abs=1e-6)
    assert native.snr_db == pytest.approx(reference.snr_db, abs=1e-3)
    assert native.confidence == pytest.approx(reference.confidence, abs=1e-6)
    assert native.valid == reference.valid


@pytest.mark.skipif(not HAVE_NATIVE, reason="native core not built")
def test_backends_agree_on_phase_sanitization_and_clutter():
    phase = np.cumsum(np.random.default_rng(1).normal(size=48)).astype(np.float32)
    assert np.allclose(sanitize_phase(phase), _numpy_dsp.sanitize_phase(phase), atol=1e-4)

    rng = np.random.default_rng(2)
    csi = (rng.normal(size=32) + 1j * rng.normal(size=32)).astype(np.complex64) * 10
    native_remover = ClutterRemover(window=8, use_native=True)
    numpy_remover = ClutterRemover(window=8, use_native=False)
    for k in range(12):
        frame = csi + (k % 3)
        a = native_remover.process(frame)
        b = numpy_remover.process(frame)
        assert np.allclose(a, b, atol=1e-4), f"backends diverged at frame {k}"


# ---------------------------------------------------------------------------
# Vital signs -- end to end
# ---------------------------------------------------------------------------


def test_respiration_rate_is_recovered_from_a_synthetic_subject():
    """A 0.27 Hz breath is 16.2 RPM; a 20 s window resolves it comfortably."""
    source = synthetic_source(respiration_hz=0.27, heart_hz=1.2, seed=5)
    result = _drain(source, 150.0, window_seconds=20.0)

    assert result.ready is True
    assert result.rpm is not None, "respiration should be detectable"
    assert abs(result.rpm - 16.2) < 2.0, f"got {result.rpm} RPM"
    assert result.confidence > 0.3


@pytest.mark.parametrize("respiration_hz,expected_rpm", [(0.2, 12.0), (0.33, 19.8)])
def test_respiration_tracks_different_rates(respiration_hz, expected_rpm):
    source = synthetic_source(respiration_hz=respiration_hz, heart_hz=1.2, seed=11)
    result = _drain(source, 150.0, window_seconds=20.0)
    assert result.rpm is not None
    assert abs(result.rpm - expected_rpm) < 2.0, f"got {result.rpm}, want {expected_rpm}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_empty_room_reports_no_vital_signs(seed):
    """The single most important behaviour: abstain rather than invent.

    A slow environmental drift produces a strong low-frequency spectral peak
    that a naive peak-picker reports as breathing. Scoring against a broadband
    noise floor, plus cross-window stability gating, must reject it.
    """
    source = synthetic_source(still=True, seed=seed)
    result = _drain(source, 120.0)

    assert result.rpm is None, f"false respiration {result.rpm} RPM on an empty room"
    assert result.bpm is None, f"false heart rate {result.bpm} BPM on an empty room"
    assert result.confidence == 0.0
    assert result.presence is False


def test_high_snr_subject_yields_a_heart_rate():
    """When the cardiac signal is strong enough, BPM is recovered.

    The default cardiac excursion is below the reliable detection floor (see
    the module README section on limitations), so this test uses a deliberately
    stronger pulse to prove the heart-rate path itself works.
    """
    source = synthetic_source(
        respiration_hz=0.27, heart_hz=1.2, heart_amplitude_m=0.012, seed=7
    )
    result = _drain(source, 150.0, window_seconds=20.0)

    assert result.bpm is not None, "heart rate should be detectable at high SNR"
    assert abs(result.bpm - 72.0) < 4.0, f"got {result.bpm} BPM"
    assert result.rpm is not None and abs(result.rpm - 16.2) < 2.0


def test_a_realistic_subject_never_reports_a_confidently_wrong_heart_rate():
    """At a realistic cardiac amplitude the estimator must abstain, not guess."""
    source = synthetic_source(heart_amplitude_m=0.0006, seed=13)
    result = _drain(source, 120.0)

    if result.bpm is not None:
        assert abs(result.bpm - 72.0) < 10.0, (
            f"reported a heart rate of {result.bpm} BPM, which is neither "
            "correct nor abstention"
        )
    else:
        assert result.bpm_confidence < 0.5


def test_gross_motion_raises_the_motion_score():
    still = _drain(synthetic_source(still=True, seed=21), 60.0)

    moving_source = synthetic_source(seed=21)
    moving_source.subject.motion_std_m = 0.05  # a person shifting about
    moving = _drain(moving_source, 60.0)

    assert moving.motion > still.motion, (
        f"motion {moving.motion:.3f} should exceed a still subject's "
        f"{still.motion:.3f}"
    )


def test_process_matrix_batch_api_agrees_with_streaming():
    source = synthetic_source(respiration_hz=0.27, seed=31)
    frames = list(source.frames(150.0))
    matrix = np.stack([f.csi for f in frames])

    batch = VitalSignsExtractor(default_config().replace(window_seconds=20.0))
    batch_result = batch.process_matrix(matrix, fs=20.0)

    assert batch_result.ready is True
    assert batch_result.rpm is not None
    assert abs(batch_result.rpm - 16.2) < 2.0


def test_vitals_result_serialises_to_the_frozen_contract():
    source = synthetic_source(seed=41)
    result = _drain(source, 40.0)
    payload = result.as_dict()

    assert set(payload) == {
        "bpm", "rpm", "confidence", "bpm_confidence",
        "rpm_confidence", "bpm_snr", "rpm_snr",
    }
    for key, value in payload.items():
        assert value is None or isinstance(value, float), (key, type(value))


def test_extractor_reset_clears_state():
    extractor = VitalSignsExtractor(default_config())
    for frame in synthetic_source(seed=51).frames(20.0):
        extractor.update(frame)
    assert extractor.frames_seen > 0

    extractor.reset()
    assert extractor.frames_seen == 0
    assert extractor.result.ready is False


def test_configuration_is_validated():
    config = default_config()
    # 12 Hz exceeds the 10 Hz Nyquist limit implied by a 20 Hz frame rate.
    with pytest.raises(ValueError):
        config.replace(heart_band_hz=(0.8, 12.0))
    with pytest.raises(ValueError):
        config.replace(respiration_band_hz=(0.5, 0.1))  # hi <= lo
    with pytest.raises(ValueError):
        config.replace(clutter_window=1)
    with pytest.raises(ValueError):
        config.replace(window_seconds=0.0)

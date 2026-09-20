"""Pose inference tests.

The estimator has two backends: ONNX Runtime when a model is configured and
available, and a deterministic analytic estimator otherwise. These tests cover
the properties the rest of the system depends on -- the COCO-17 shape, JSON
safety, determinism, graceful degradation when no model is present, and the
temporal smoothing that keeps the rendered avatar from shaking.
"""

from __future__ import annotations

import numpy as np
import pytest

from csi_fixtures import random_csi, synthetic_source
from spectraflow.config import default_config
from spectraflow.inference import CANONICAL_SKELETON, KP, PoseEstimator, PoseResult


def _window(seed: int = 0, frames: int = 20, subcarriers: int = 64):
    return random_csi(frames, subcarriers, seed=seed)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_estimator_returns_coco17_keypoints():
    estimator = PoseEstimator()
    result = estimator.estimate(_window())

    assert isinstance(result, PoseResult)
    assert result.keypoints is not None
    assert result.keypoints.shape == (17, 3)
    assert np.isfinite(result.keypoints).all()
    assert 0.0 <= result.confidence <= 1.0


def test_keypoint_index_map_covers_every_joint():
    assert sorted(KP.values()) == list(range(17))
    assert CANONICAL_SKELETON.shape == (17, 3)
    # A standing adult: feet near the floor, head near 1.6-1.7 m.
    assert CANONICAL_SKELETON[:, 1].min() < 0.2
    assert 1.5 < CANONICAL_SKELETON[:, 1].max() < 1.8


def test_keypoints_are_json_safe():
    estimator = PoseEstimator()
    payload = estimator.estimate(_window()).as_list()

    assert payload is not None and len(payload) == 17
    for point in payload:
        assert len(point) == 3
        for value in point:
            # json.dumps must not choke on numpy scalars.
            assert isinstance(value, float), type(value)


def test_no_pose_is_reported_for_a_degenerate_window():
    estimator = PoseEstimator()
    assert estimator.estimate(np.zeros((0, 64), dtype=np.complex64)).keypoints is None
    assert estimator.estimate(np.zeros((1, 64), dtype=np.complex64)).keypoints is None
    assert estimator.estimate(np.zeros((4, 0), dtype=np.complex64)).keypoints is None


def test_estimator_handles_a_single_subcarrier():
    estimator = PoseEstimator()
    result = estimator.estimate(random_csi(20, 1, seed=2))
    assert result.keypoints is not None
    assert result.keypoints.shape == (17, 3)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_estimator_is_deterministic_for_the_same_input():
    window = _window(seed=5)
    first = PoseEstimator().estimate(window)
    second = PoseEstimator().estimate(window)
    assert np.array_equal(first.keypoints, second.keypoints)
    assert first.confidence == second.confidence


def test_estimator_responds_to_different_input():
    """Different CSI must produce a different pose, or it is not data-driven."""
    a = PoseEstimator().estimate(_window(seed=1))
    b = PoseEstimator().estimate(_window(seed=999))
    assert not np.allclose(a.keypoints, b.keypoints)


def test_pose_moves_are_small_on_a_realistic_timescale():
    """The skeleton must not teleport between adjacent frames of real CSI."""
    source = synthetic_source(seed=8)
    estimator = PoseEstimator()
    window = []
    previous = None
    biggest = 0.0
    for frame in source.frames(4.0):
        window.append(frame.csi)
        window = window[-20:]
        if len(window) < 20:
            continue
        result = estimator.estimate(np.stack(window))
        if previous is not None:
            biggest = max(biggest, float(np.abs(result.keypoints - previous).max()))
        previous = result.keypoints

    assert biggest < 0.35, f"joint jumped {biggest:.3f} m between adjacent frames"


def test_temporal_smoothing_damps_an_abrupt_change():
    """A sudden input change must not move the skeleton all the way at once."""
    estimator = PoseEstimator()
    first = estimator.estimate(_window(seed=1))
    second = estimator.estimate(_window(seed=2))  # very different input

    moved = float(np.abs(second.keypoints - first.keypoints).max())
    target = float(
        np.abs(
            PoseEstimator().estimate(_window(seed=2)).keypoints - first.keypoints
        ).max()
    )
    assert target > 1e-6, "the two windows should produce different poses"
    assert moved < target, "smoothing did not damp the transition"


def test_confidence_tracks_channel_activity():
    estimator = PoseEstimator()
    static = np.ones((20, 64), dtype=np.complex64) * (5 + 5j)
    quiet = estimator.estimate(static)

    estimator.reset()
    busy = estimator.estimate(random_csi(20, 64, seed=3) * 30)

    assert quiet.confidence < busy.confidence
    # A perfectly static channel carries no pose information, so the estimator
    # should be near-certain of nothing. (Not exactly zero: float32 rounding in
    # the amplitude statistics leaves ~1e-6 of numerical noise.)
    assert quiet.confidence == pytest.approx(0.0, abs=1e-3)


def test_reset_clears_the_smoothing_state():
    estimator = PoseEstimator()
    estimator.estimate(_window(seed=1))
    assert estimator._smoothed is not None
    estimator.reset()
    assert estimator._smoothed is None


# ---------------------------------------------------------------------------
# Backend selection and graceful degradation
# ---------------------------------------------------------------------------


def test_absent_model_falls_back_to_the_analytic_estimator():
    estimator = PoseEstimator(model_path="/nonexistent/model.onnx")
    assert estimator.backend == "analytic"
    assert estimator.estimate(_window()).keypoints is not None


def test_corrupt_model_does_not_break_estimation(tmp_path):
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"this is not a valid onnx model")
    estimator = PoseEstimator(model_path=broken)
    # Whether onnxruntime is present or not, estimation must keep working.
    assert estimator.estimate(_window()).keypoints is not None


def test_configured_model_path_is_honoured():
    config = default_config().replace(onnx_model_path="/tmp/definitely-missing.onnx")
    estimator = PoseEstimator(config)
    assert estimator.backend == "analytic"


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("onnxruntime") is None,
    reason="onnxruntime not installed",
)
def test_onnx_backend_is_used_when_a_real_model_is_present(tmp_path):  # pragma: no cover
    """Sanity check the ONNX wiring end to end with a trivial identity model."""
    import onnx
    import torch

    class Identity(torch.nn.Module):
        def forward(self, x):
            # (1, 1, subcarriers, frames) -> (1, 17, 3)
            return torch.zeros(1, 17, 3, dtype=torch.float32)

    model_path = tmp_path / "pose.onnx"
    torch.onnx.export(
        Identity(),
        torch.zeros(1, 1, 64, 20),
        str(model_path),
        input_names=["csi"],
        output_names=["keypoints"],
        opset_version=13,
    )
    assert onnx is not None

    estimator = PoseEstimator(model_path=model_path)
    assert estimator.backend == "onnx"
    result = estimator.estimate(random_csi(20, 64, seed=1))
    assert result.keypoints.shape == (17, 3)
    assert result.source == "onnx"

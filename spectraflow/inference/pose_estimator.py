"""3D human pose estimation from CSI amplitude/phase windows.

Two backends, selected automatically:

* **ONNX Runtime** -- used whenever ``onnxruntime`` is importable *and* a model
  file is present. The session is driven with the tensor contract the rest of
  the pipeline assumes: input ``(1, 1, n_subcarriers, window_size)``, output
  ``(17, 3)`` in COCO-17 order.
* **Analytic estimator** -- a deterministic, data-driven fallback used when no
  model is available (``onnxruntime`` has no Android/aarch64 wheel, so this is
  the normal path on the reference Termux deployment).

The fallback is not a stub and does not emit random or canned data. It projects
features measured from the actual CSI window onto a low-dimensional latent space
and maps that onto a canonical skeleton, so the rendered avatar genuinely
responds to the observed channel -- a moving subject produces a moving skeleton
with correlated limb displacement. What it does **not** do is infer real
anatomy: it has no learned model of the human body, so its output should be read
as a motion visualisation, not as metrically accurate joint positions. The
server reports which backend produced each frame, and the UI can surface it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from spectraflow.config import SpectraflowConfig, default_config
from spectraflow.dsp.phase_sanitizer import sanitize_phase

logger = logging.getLogger(__name__)

__all__ = ["PoseEstimator", "PoseResult", "CANONICAL_SKELETON", "KP"]

#: COCO-17 keypoint order, as consumed by the Three.js front-end.
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

#: Canonical standing pose in metres, Y-up, origin at the floor centre of the
#: sensing volume. Roughly a 1.7 m adult, used as the rest pose that the
#: analytic estimator perturbs.
CANONICAL_SKELETON: npt.NDArray[np.float32] = np.array(
    [
        [0.00, 1.60, 0.02],   # 0  nose
        [0.03, 1.63, 0.06],   # 1  left_eye
        [-0.03, 1.63, 0.06],  # 2  right_eye
        [0.07, 1.62, 0.02],   # 3  left_ear
        [-0.07, 1.62, 0.02],  # 4  right_ear
        [0.18, 1.42, 0.00],   # 5  left_shoulder
        [-0.18, 1.42, 0.00],  # 6  right_shoulder
        [0.25, 1.15, 0.00],   # 7  left_elbow
        [-0.25, 1.15, 0.00],  # 8  right_elbow
        [0.28, 0.90, 0.00],   # 9  left_wrist
        [-0.28, 0.90, 0.00],  # 10 right_wrist
        [0.10, 0.90, 0.00],   # 11 left_hip
        [-0.10, 0.90, 0.00],  # 12 right_hip
        [0.11, 0.50, 0.00],   # 13 left_knee
        [-0.11, 0.50, 0.00],  # 14 right_knee
        [0.12, 0.08, 0.00],   # 15 left_ankle
        [-0.12, 0.08, 0.00],  # 16 right_ankle
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class PoseResult:
    """One pose estimate."""

    #: ``(17, 3)`` float32 array of [x, y, z] metres, or ``None`` when no pose
    #: could be produced (no person detected / insufficient data).
    keypoints: npt.NDArray[np.float32] | None = None
    #: Confidence in ``[0, 1]``.
    confidence: float = 0.0
    #: ``"onnx"`` or ``"analytic"``.
    source: str = "analytic"
    #: True when the estimator believes a subject is present.
    present: bool = False

    def as_list(self) -> list[list[float]] | None:
        """JSON-ready nested list, matching the WebSocket contract."""
        if self.keypoints is None:
            return None
        return [[round(float(v), 4) for v in row] for row in self.keypoints]


class PoseEstimator:
    """CSI -> 3D keypoints.

    Args:
        config: Pipeline configuration.
        model_path: Explicit ``.onnx`` path; falls back to
            ``config.onnx_model_path``.

    Example::

        estimator = PoseEstimator()
        pose = estimator.estimate(csi_window)   # (window, subcarriers)
    """

    def __init__(
        self,
        config: SpectraflowConfig | None = None,
        model_path: str | Path | None = None,
    ) -> None:
        self.config = config or default_config()
        self.model_path = Path(model_path) if model_path else (
            Path(self.config.onnx_model_path) if self.config.onnx_model_path else None
        )

        self._session: Any = None
        self._input_name: str = ""
        self._smoothed: npt.NDArray[np.float32] | None = None

        # Fixed projection from CSI features to pose latent space. Seeded once
        # so the analytic estimator is fully deterministic across runs, which
        # makes its output reproducible and testable.
        rng = np.random.default_rng(0x5EC7A)
        self._n_features = 24
        self._n_latent = 6
        self._projection = rng.normal(0.0, 1.0, size=(self._n_latent, self._n_features))
        self._projection /= np.sqrt(self._n_features)
        # Per-joint, per-latent, per-axis displacement basis, scaled so the
        # induced motion stays physically plausible (centimetres, not metres).
        self._joint_basis = rng.normal(
            0.0, 1.0, size=(self.config.num_keypoints, self._n_latent, 3)
        ).astype(np.float32) * 0.04

        self._load_session()

    # -- backend selection -------------------------------------------------
    def _load_session(self) -> None:
        """Try to bring up ONNX Runtime; silently fall back if unavailable."""
        if self.model_path is None:
            logger.info(
                "PoseEstimator: no ONNX model configured; using the analytic "
                "CSI-driven pose estimator."
            )
            return
        if not self.model_path.is_file():
            logger.warning(
                "PoseEstimator: model %s not found; using the analytic estimator.",
                self.model_path,
            )
            return
        try:
            import onnxruntime  # noqa: PLC0415 - optional, imported on demand
        except ImportError:
            logger.warning(
                "PoseEstimator: onnxruntime is not installed (no Android/aarch64 "
                "wheel upstream); using the analytic estimator."
            )
            return
        try:
            options = onnxruntime.SessionOptions()
            # The inference runs on the same ARM cores as the DSP; cap the
            # thread pool so pose estimation cannot starve the sensing loop.
            options.intra_op_num_threads = 2
            options.log_severity_level = 3
            self._session = onnxruntime.InferenceSession(
                str(self.model_path), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info(
                "PoseEstimator: loaded ONNX model %s (input '%s')",
                self.model_path, self._input_name,
            )
        except Exception as exc:  # pragma: no cover - depends on model file
            logger.warning(
                "PoseEstimator: failed to load %s (%s); using the analytic estimator.",
                self.model_path, exc,
            )
            self._session = None

    @property
    def backend(self) -> str:
        """``"onnx"`` when a model is live, otherwise ``"analytic"``."""
        return "onnx" if self._session is not None else "analytic"

    # -- inference ---------------------------------------------------------
    def estimate(self, csi_window: npt.ArrayLike) -> PoseResult:
        """Estimate a pose from a window of CSI frames.

        Args:
            csi_window: ``(n_frames, n_subcarriers)`` complex CSI, oldest first.

        Returns:
            A :class:`PoseResult`; ``keypoints`` is ``None`` when the window is
            too small to say anything.
        """
        window = np.asarray(csi_window)
        if window.ndim != 2 or window.shape[0] < 2 or window.shape[1] < 1:
            return PoseResult(keypoints=None, confidence=0.0, source=self.backend)

        if self._session is not None:
            keypoints, confidence = self._infer_onnx(window)
        else:
            keypoints, confidence = self._infer_analytic(window)

        keypoints = self._smooth(keypoints)
        return PoseResult(
            keypoints=keypoints,
            confidence=confidence,
            source=self.backend,
            present=confidence > 0.05,
        )

    def _infer_onnx(
        self, window: npt.NDArray[np.complex64]
    ) -> tuple[npt.NDArray[np.float32], float]:
        """Run the configured model.

        The wired contract is ``(1, 1, n_subcarriers, window_size)`` float32.
        """
        amplitude_db = 20.0 * np.log10(np.maximum(np.abs(window), 1e-8))
        # (frames, subcarriers) -> (1, 1, subcarriers, frames)
        tensor = amplitude_db.T.astype(np.float32)[None, None, :, :]
        tensor = np.ascontiguousarray(tensor)

        outputs = self._session.run(None, {self._input_name: tensor})
        raw = np.asarray(outputs[0], dtype=np.float32).reshape(-1, 3)
        if raw.shape[0] < self.config.num_keypoints:
            # A model that emits fewer joints than the contract requires is not
            # usable; degrade rather than emit a malformed skeleton.
            logger.warning(
                "PoseEstimator: model returned %d keypoints, need %d",
                raw.shape[0], self.config.num_keypoints,
            )
            return self._infer_analytic(window)
        keypoints = raw[: self.config.num_keypoints]

        confidence = 0.9
        if len(outputs) > 1:
            score = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
            if score.size:
                confidence = float(np.clip(score.mean(), 0.0, 1.0))
        return keypoints, confidence

    def _features(self, window: npt.NDArray[np.complex64]) -> npt.NDArray[np.float64]:
        """Build a fixed-length feature vector from a CSI window.

        Deliberately mixes amplitude and *sanitized phase* statistics: phase is
        where the subject's displacement actually lives, while amplitude
        carries the reflection strength.
        """
        amplitude_db = 20.0 * np.log10(np.maximum(np.abs(window), 1e-8))
        phase = np.stack([sanitize_phase(row) for row in window])

        # Aggregate across subcarriers, then describe the resulting time series.
        amp_track = amplitude_db.mean(axis=1)
        phase_track = phase.mean(axis=1)

        def describe(track: npt.NDArray[np.float64]) -> list[float]:
            track = np.asarray(track, dtype=np.float64)
            centred = track - track.mean()
            scale = float(np.std(centred)) + 1e-9
            # Normalised autocorrelation at a few lags: captures the dominant
            # periodicity without an FFT.
            lags = [1, 2, 4, max(1, track.size // 4)]
            ac = [
                float(np.mean(centred[:-lag] * centred[lag:]) / (scale * scale))
                if track.size > lag
                else 0.0
                for lag in lags
            ]
            return [
                float(track.mean()),
                scale,
                float(np.mean(np.abs(np.diff(track)))) if track.size > 1 else 0.0,
                *ac,
            ]

        features = np.asarray(
            describe(amp_track) + describe(phase_track), dtype=np.float64
        )
        # Pad or trim to the fixed projection width.
        if features.size < self._n_features:
            features = np.pad(features, (0, self._n_features - features.size))
        return features[: self._n_features]

    def _infer_analytic(
        self, window: npt.NDArray[np.complex64]
    ) -> tuple[npt.NDArray[np.float32], float]:
        """Deterministic CSI-driven pose estimate (see the module docstring)."""
        features = self._features(window)
        # Normalise so the latent stays bounded regardless of link gain.
        features = features / (np.linalg.norm(features) + 1e-9)
        latent = np.tanh(self._projection @ features)

        keypoints = CANONICAL_SKELETON.astype(np.float32).copy()
        keypoints += np.einsum("jkl,k->jl", self._joint_basis, latent).astype(np.float32)

        # Confidence from how much the window actually varies: a perfectly
        # static channel carries no pose information, and saying so is more
        # useful than reporting a confident standing pose.
        amplitude_db = 20.0 * np.log10(np.maximum(np.abs(window), 1e-8))
        activity = float(np.std(amplitude_db.mean(axis=1)))
        confidence = float(np.clip(activity / 3.0, 0.0, 1.0))
        return keypoints, confidence

    def _smooth(
        self, keypoints: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Temporally smooth the skeleton to suppress frame-to-frame jitter.

        The network is applied to a sliding window, so its output already trends
        smoothly; this removes the residual high-frequency wobble that would
        otherwise read as a shaking avatar.
        """
        if self._smoothed is None or self._smoothed.shape != keypoints.shape:
            self._smoothed = keypoints.astype(np.float32).copy()
        else:
            self._smoothed = (
                0.7 * self._smoothed + 0.3 * keypoints.astype(np.float32)
            )
        return self._smoothed.copy()

    def reset(self) -> None:
        """Clear temporal state."""
        self._smoothed = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<PoseEstimator backend={self.backend} model={self.model_path}>"

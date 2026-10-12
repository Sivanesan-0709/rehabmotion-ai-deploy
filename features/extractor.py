"""
AI Exercise Form & Posture Intelligence Platform
Unified Kinematic Feature Extractor
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from cv.landmarks import PoseLandmarks
from features.angles import extract_all_angles
from features.geometry import extract_all_geometric_features
from features.temporal import TemporalFeatureTracker


@dataclass
class KinematicFeatures:
    """Consolidated biomechanical and spatial features for a single frame."""
    timestamp: float
    angles: Dict[str, float]
    geometry: Dict[str, float]
    velocities: Dict[str, float]
    accelerations: Dict[str, float]
    smoothness_score: float
    normalized_landmarks: np.ndarray
    is_valid: bool = True

    def to_ml_feature_vector(self) -> np.ndarray:
        """
        Flattens core biomechanical features into a standard numerical vector for ML classification.
        Order:
        - Angles (13)
        - Geometry & Symmetry (10)
        - Velocities (4)
        - Accelerations (4)
        - Smoothness (1)
        - Normalized key landmarks (selected 12 key joints * 3 = 36)
        Total vector length = 68 features
        """
        feats = []

        # 1. Angles
        for k in [
            "left_knee", "right_knee", "mean_knee",
            "left_hip", "right_hip", "mean_hip",
            "left_elbow", "right_elbow", "mean_elbow",
            "left_shoulder", "right_shoulder", "mean_shoulder",
            "torso_inclination"
        ]:
            feats.append(self.angles.get(k, 0.0))

        # 2. Geometry & Symmetry
        for k in [
            "knee_symmetry", "elbow_symmetry", "hip_symmetry", "overall_symmetry",
            "shoulder_tilt", "hip_tilt",
            "left_knee_valgus", "right_knee_valgus",
            "left_elbow_drift", "right_elbow_drift"
        ]:
            feats.append(self.geometry.get(k, 0.0))

        # 3. Velocities & Accelerations
        for k in ["knee", "hip", "elbow", "shoulder"]:
            feats.append(self.velocities.get(k, 0.0))
        for k in ["knee", "hip", "elbow", "shoulder"]:
            feats.append(self.accelerations.get(k, 0.0))

        # 4. Smoothness
        feats.append(self.smoothness_score)

        # 5. Core 12 landmark 3D positions (indices 11 to 28)
        # 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28 -> 12 * 3 = 36
        if len(self.normalized_landmarks) >= 99:
            # extract indices for key landmarks
            for joint_idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                start_i = joint_idx * 3
                feats.extend(self.normalized_landmarks[start_i:start_i + 3].tolist())
        else:
            feats.extend([0.0] * 36)

        return np.array(feats, dtype=np.float32)


class FeatureExtractor:
    """High-level extractor orchestrating spatial, geometric, and temporal pipeline."""

    def __init__(self):
        self.temporal_tracker = TemporalFeatureTracker()

    def extract(self, pose: PoseLandmarks) -> KinematicFeatures:
        """Extracts complete feature set from raw pose landmarks."""
        if not pose.is_valid:
            return KinematicFeatures(
                timestamp=pose.timestamp,
                angles={},
                geometry={},
                velocities={},
                accelerations={},
                smoothness_score=0.0,
                normalized_landmarks=np.zeros(99, dtype=np.float32),
                is_valid=False
            )

        angles = extract_all_angles(pose)
        geometry = extract_all_geometric_features(pose)

        # Update temporal tracker
        self.temporal_tracker.update(angles, pose.timestamp)

        velocities = {
            "knee": self.temporal_tracker.get_velocity("knee"),
            "hip": self.temporal_tracker.get_velocity("hip"),
            "elbow": self.temporal_tracker.get_velocity("elbow"),
            "shoulder": self.temporal_tracker.get_velocity("shoulder"),
        }

        accelerations = {
            "knee": self.temporal_tracker.get_acceleration("knee"),
            "hip": self.temporal_tracker.get_acceleration("hip"),
            "elbow": self.temporal_tracker.get_acceleration("elbow"),
            "shoulder": self.temporal_tracker.get_acceleration("shoulder"),
        }

        smoothness = self.temporal_tracker.compute_smoothness_score("knee")
        norm_landmarks = pose.get_normalized_vector()

        return KinematicFeatures(
            timestamp=pose.timestamp,
            angles=angles,
            geometry=geometry,
            velocities=velocities,
            accelerations=accelerations,
            smoothness_score=smoothness,
            normalized_landmarks=norm_landmarks,
            is_valid=True
        )

    def reset(self) -> None:
        self.temporal_tracker.reset()

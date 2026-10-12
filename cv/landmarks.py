"""
AI Exercise Form & Posture Intelligence Platform
Landmark Definitions & Normalization Utilities
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np


class LandmarkIndex(IntEnum):
    """MediaPipe Pose 33 Landmark Indices"""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class Point3D:
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def to_2d_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


# Stable Canonical Alias
PoseLandmark = Point3D


@dataclass
class PoseLandmarks:
    """Encapsulates 33 3D body landmarks with confidence scores and normalization."""
    landmarks: Dict[int, Point3D]
    timestamp: float
    is_valid: bool = True
    confidence: float = 0.0
    world_landmarks: Optional[Dict[int, Point3D]] = None

    @property
    def detected(self) -> bool:
        """Alias property indicating whether a valid pose was detected."""
        return self.is_valid and len(self.landmarks) > 0

    def get(self, idx: LandmarkIndex) -> Optional[Point3D]:
        return self.landmarks.get(int(idx))

    def get_numpy(self, idx: LandmarkIndex, dim3: bool = True) -> np.ndarray:
        pt = self.landmarks.get(int(idx))
        if pt is None:
            return np.zeros(3 if dim3 else 2, dtype=np.float32)
        return pt.to_numpy() if dim3 else pt.to_2d_numpy()

    def get_visibility(self, idx: LandmarkIndex) -> float:
        pt = self.landmarks.get(int(idx))
        return pt.visibility if pt else 0.0

    def get_midpoint(self, idx1: LandmarkIndex, idx2: LandmarkIndex) -> Point3D:
        p1 = self.landmarks.get(int(idx1))
        p2 = self.landmarks.get(int(idx2))
        if not p1 or not p2:
            return Point3D(0.0, 0.0, 0.0, 0.0)
        return Point3D(
            x=(p1.x + p2.x) / 2.0,
            y=(p1.y + p2.y) / 2.0,
            z=(p1.z + p2.z) / 2.0,
            visibility=min(p1.visibility, p2.visibility)
        )

    def get_torso_height(self) -> float:
        """Computes distance between shoulder midpoint and hip midpoint for scale normalization."""
        shoulder_mid = self.get_midpoint(LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER)
        hip_mid = self.get_midpoint(LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP)
        dist = np.linalg.norm(shoulder_mid.to_2d_numpy() - hip_mid.to_2d_numpy())
        return float(dist) if dist > 1e-4 else 1.0

    def get_normalized_vector(self) -> np.ndarray:
        """
        Extracts a normalized, translation and scale-invariant 1D feature array (99 features = 33*3).
        Origin is centered at hip midpoint, scaled by torso height.
        """
        hip_mid = self.get_midpoint(LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP).to_numpy()
        torso_h = self.get_torso_height()

        features = []
        for i in range(33):
            pt = self.landmarks.get(i)
            if pt is not None:
                norm_pt = (pt.to_numpy() - hip_mid) / torso_h
                features.extend(norm_pt.tolist())
            else:
                features.extend([0.0, 0.0, 0.0])
        return np.array(features, dtype=np.float32)

    def check_visibility_threshold(self, indices: List[LandmarkIndex], min_vis: float = 0.5) -> bool:
        """Verifies whether critical landmarks meet minimum visibility."""
        for idx in indices:
            if self.get_visibility(idx) < min_vis:
                return False
        return True


# Stable Canonical Alias
PoseResult = PoseLandmarks


"""
AI Exercise Form & Posture Intelligence Platform
Test Helper Utilities
"""

import time
from cv.landmarks import LandmarkIndex, Point3D, PoseLandmarks


def create_synthetic_pose(
    knee_angle_deg: float = 170.0,
    elbow_angle_deg: float = 170.0,
    torso_lean_deg: float = 0.0
) -> PoseLandmarks:
    """Constructs an idealized 3D pose landmarks set for deterministic testing."""
    landmarks = {}

    # Shoulders
    landmarks[int(LandmarkIndex.LEFT_SHOULDER)] = Point3D(0.6, 0.25, 0.0, 0.99)
    landmarks[int(LandmarkIndex.RIGHT_SHOULDER)] = Point3D(0.4, 0.25, 0.0, 0.99)

    # Hips
    landmarks[int(LandmarkIndex.LEFT_HIP)] = Point3D(0.55, 0.50, 0.0, 0.99)
    landmarks[int(LandmarkIndex.RIGHT_HIP)] = Point3D(0.45, 0.50, 0.0, 0.99)

    # Elbows and Wrists
    landmarks[int(LandmarkIndex.LEFT_ELBOW)] = Point3D(0.62, 0.38, 0.0, 0.99)
    landmarks[int(LandmarkIndex.RIGHT_ELBOW)] = Point3D(0.38, 0.38, 0.0, 0.99)
    landmarks[int(LandmarkIndex.LEFT_WRIST)] = Point3D(0.62, 0.52, 0.0, 0.99)
    landmarks[int(LandmarkIndex.RIGHT_WRIST)] = Point3D(0.38, 0.52, 0.0, 0.99)

    # Knees and Ankles
    landmarks[int(LandmarkIndex.LEFT_KNEE)] = Point3D(0.55, 0.70, 0.0, 0.99)
    landmarks[int(LandmarkIndex.RIGHT_KNEE)] = Point3D(0.45, 0.70, 0.0, 0.99)
    landmarks[int(LandmarkIndex.LEFT_ANKLE)] = Point3D(0.55, 0.90, 0.0, 0.99)
    landmarks[int(LandmarkIndex.RIGHT_ANKLE)] = Point3D(0.45, 0.90, 0.0, 0.99)
    landmarks[int(LandmarkIndex.LEFT_FOOT_INDEX)] = Point3D(0.58, 0.93, 0.0, 0.99)
    landmarks[int(LandmarkIndex.RIGHT_FOOT_INDEX)] = Point3D(0.42, 0.93, 0.0, 0.99)

    return PoseLandmarks(landmarks=landmarks, timestamp=time.time(), is_valid=True, confidence=0.98)

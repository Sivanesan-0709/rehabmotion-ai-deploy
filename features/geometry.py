"""
AI Exercise Form & Posture Intelligence Platform
Spatial Geometry, Alignment & Biomechanical Symmetry Module
"""

from typing import Dict, Tuple
import numpy as np
from cv.landmarks import LandmarkIndex, PoseLandmarks


def calculate_symmetry_index(left_val: float, right_val: float) -> float:
    """
    Computes a percentage symmetry index between left and right measurements.
    100% = Perfect symmetry.
    0% = Extreme asymmetry.
    """
    denom = max(abs(left_val), abs(right_val), 1e-4)
    diff = abs(left_val - right_val)
    sym = 1.0 - (diff / denom)
    return float(np.clip(sym * 100.0, 0.0, 100.0))


def calculate_knee_valgus(pose: PoseLandmarks) -> Tuple[float, float]:
    """
    Evaluates Knee Valgus (inward caving of knees relative to hip and ankle projection).
    Positive score indicates inward collapse (valgus), negative indicates flaring (varus).
    Returns (left_valgus_score, right_valgus_score) normalized in degrees/ratio.
    """
    l_hip = pose.get_numpy(LandmarkIndex.LEFT_HIP)
    l_knee = pose.get_numpy(LandmarkIndex.LEFT_KNEE)
    l_ankle = pose.get_numpy(LandmarkIndex.LEFT_ANKLE)

    r_hip = pose.get_numpy(LandmarkIndex.RIGHT_HIP)
    r_knee = pose.get_numpy(LandmarkIndex.RIGHT_KNEE)
    r_ankle = pose.get_numpy(LandmarkIndex.RIGHT_ANKLE)

    # For Left leg (in front view, Left Hip is on right side of image if mirrored):
    # Ideal knee X lies on line connecting hip X and ankle X.
    l_expected_knee_x = (l_hip[0] + l_ankle[0]) / 2.0
    r_expected_knee_x = (r_hip[0] + r_ankle[0]) / 2.0

    torso_h = pose.get_torso_height()

    # Valgus delta normalized by torso height
    l_valgus = (l_knee[0] - l_expected_knee_x) / torso_h
    r_valgus = (r_expected_knee_x - r_knee[0]) / torso_h

    return float(l_valgus * 100.0), float(r_valgus * 100.0)


def calculate_horizontal_levelness(pose: PoseLandmarks, idx1: LandmarkIndex, idx2: LandmarkIndex) -> float:
    """
    Calculates the tilt angle (in degrees) of a bilateral pair (e.g. shoulders or hips) from horizontal.
    0 deg = perfectly level.
    """
    p1 = pose.get_numpy(idx1)
    p2 = pose.get_numpy(idx2)
    delta_y = abs(p1[1] - p2[1])
    delta_x = abs(p1[0] - p2[0])
    if delta_x < 1e-6:
        return 90.0
    angle_rad = np.arctan(delta_y / delta_x)
    return float(np.degrees(angle_rad))


def calculate_elbow_torso_drift(pose: PoseLandmarks) -> Tuple[float, float]:
    """
    Measures horizontal displacement of elbow relative to shoulder/hip line.
    Useful for detecting elbow sway in bicep curls and overhead presses.
    """
    torso_h = pose.get_torso_height()

    l_shoulder = pose.get_numpy(LandmarkIndex.LEFT_SHOULDER)
    l_elbow = pose.get_numpy(LandmarkIndex.LEFT_ELBOW)
    r_shoulder = pose.get_numpy(LandmarkIndex.RIGHT_SHOULDER)
    r_elbow = pose.get_numpy(LandmarkIndex.RIGHT_ELBOW)

    l_drift = abs(l_elbow[0] - l_shoulder[0]) / torso_h
    r_drift = abs(r_elbow[0] - r_shoulder[0]) / torso_h

    return float(l_drift * 100.0), float(r_drift * 100.0)


def extract_all_geometric_features(pose: PoseLandmarks) -> Dict[str, float]:
    """Computes all structural alignment and symmetry indicators for current frame."""
    from features.angles import extract_knee_angles, extract_elbow_angles, extract_hip_angles

    l_knee, r_knee = extract_knee_angles(pose)
    l_elbow, r_elbow = extract_elbow_angles(pose)
    l_hip, r_hip = extract_hip_angles(pose)

    knee_sym = calculate_symmetry_index(l_knee, r_knee)
    elbow_sym = calculate_symmetry_index(l_elbow, r_elbow)
    hip_sym = calculate_symmetry_index(l_hip, r_hip)
    overall_sym = (knee_sym + elbow_sym + hip_sym) / 3.0

    shoulder_tilt = calculate_horizontal_levelness(pose, LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER)
    hip_tilt = calculate_horizontal_levelness(pose, LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP)
    l_valgus, r_valgus = calculate_knee_valgus(pose)
    l_drift, r_drift = calculate_elbow_torso_drift(pose)

    return {
        "knee_symmetry": knee_sym,
        "elbow_symmetry": elbow_sym,
        "hip_symmetry": hip_sym,
        "overall_symmetry": overall_sym,
        "shoulder_tilt": shoulder_tilt,
        "hip_tilt": hip_tilt,
        "left_knee_valgus": l_valgus,
        "right_knee_valgus": r_valgus,
        "left_elbow_drift": l_drift,
        "right_elbow_drift": r_drift,
    }

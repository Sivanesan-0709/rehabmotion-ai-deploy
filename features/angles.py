"""
AI Exercise Form & Posture Intelligence Platform
Biomechanical Joint Angle Computation Module
"""

from typing import Dict, Tuple
import numpy as np
from cv.landmarks import LandmarkIndex, PoseLandmarks


def calculate_angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculates the 2D planar angle (in degrees) at vertex B formed by segments BA and BC.
    Output range: [0.0, 180.0] degrees.
    """
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clamp cosine to avoid floating-point errors outside [-1.0, 1.0]
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return float(np.degrees(angle_rad))


def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculates the true 3D spatial angle (in degrees) at vertex B.
    """
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_segment_angle_to_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    """
    Calculates the angle of a segment (from bottom to top) relative to the vertical line (Y-axis).
    0 degrees = perfectly vertical.
    90 degrees = horizontal.
    """
    vec = top[:2] - bottom[:2]
    # In screen coordinates, Y points downward. Vertical vector upwards is [0, -1]
    vertical = np.array([0.0, -1.0], dtype=np.float32)
    norm_vec = np.linalg.norm(vec)

    if norm_vec < 1e-6:
        return 0.0

    cos_angle = np.dot(vec, vertical) / norm_vec
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def extract_knee_angles(pose: PoseLandmarks, dim3: bool = False) -> Tuple[float, float]:
    """Returns (left_knee_angle, right_knee_angle) at knee vertex (Hip - Knee - Ankle)."""
    func = calculate_angle_3d if dim3 else calculate_angle_2d
    l_hip = pose.get_numpy(LandmarkIndex.LEFT_HIP, dim3)
    l_knee = pose.get_numpy(LandmarkIndex.LEFT_KNEE, dim3)
    l_ankle = pose.get_numpy(LandmarkIndex.LEFT_ANKLE, dim3)

    r_hip = pose.get_numpy(LandmarkIndex.RIGHT_HIP, dim3)
    r_knee = pose.get_numpy(LandmarkIndex.RIGHT_KNEE, dim3)
    r_ankle = pose.get_numpy(LandmarkIndex.RIGHT_ANKLE, dim3)

    return func(l_hip, l_knee, l_ankle), func(r_hip, r_knee, r_ankle)


def extract_hip_angles(pose: PoseLandmarks, dim3: bool = False) -> Tuple[float, float]:
    """Returns (left_hip_angle, right_hip_angle) at hip vertex (Shoulder - Hip - Knee)."""
    func = calculate_angle_3d if dim3 else calculate_angle_2d
    l_shoulder = pose.get_numpy(LandmarkIndex.LEFT_SHOULDER, dim3)
    l_hip = pose.get_numpy(LandmarkIndex.LEFT_HIP, dim3)
    l_knee = pose.get_numpy(LandmarkIndex.LEFT_KNEE, dim3)

    r_shoulder = pose.get_numpy(LandmarkIndex.RIGHT_SHOULDER, dim3)
    r_hip = pose.get_numpy(LandmarkIndex.RIGHT_HIP, dim3)
    r_knee = pose.get_numpy(LandmarkIndex.RIGHT_KNEE, dim3)

    return func(l_shoulder, l_hip, l_knee), func(r_shoulder, r_hip, r_knee)


def extract_elbow_angles(pose: PoseLandmarks, dim3: bool = False) -> Tuple[float, float]:
    """Returns (left_elbow_angle, right_elbow_angle) at elbow vertex (Shoulder - Elbow - Wrist)."""
    func = calculate_angle_3d if dim3 else calculate_angle_2d
    l_shoulder = pose.get_numpy(LandmarkIndex.LEFT_SHOULDER, dim3)
    l_elbow = pose.get_numpy(LandmarkIndex.LEFT_ELBOW, dim3)
    l_wrist = pose.get_numpy(LandmarkIndex.LEFT_WRIST, dim3)

    r_shoulder = pose.get_numpy(LandmarkIndex.RIGHT_SHOULDER, dim3)
    r_elbow = pose.get_numpy(LandmarkIndex.RIGHT_ELBOW, dim3)
    r_wrist = pose.get_numpy(LandmarkIndex.RIGHT_WRIST, dim3)

    return func(l_shoulder, l_elbow, l_wrist), func(r_shoulder, r_elbow, r_wrist)


def extract_shoulder_angles(pose: PoseLandmarks, dim3: bool = False) -> Tuple[float, float]:
    """Returns (left_shoulder_angle, right_shoulder_angle) at shoulder vertex (Hip - Shoulder - Elbow)."""
    func = calculate_angle_3d if dim3 else calculate_angle_2d
    l_hip = pose.get_numpy(LandmarkIndex.LEFT_HIP, dim3)
    l_shoulder = pose.get_numpy(LandmarkIndex.LEFT_SHOULDER, dim3)
    l_elbow = pose.get_numpy(LandmarkIndex.LEFT_ELBOW, dim3)

    r_hip = pose.get_numpy(LandmarkIndex.RIGHT_HIP, dim3)
    r_shoulder = pose.get_numpy(LandmarkIndex.RIGHT_SHOULDER, dim3)
    r_elbow = pose.get_numpy(LandmarkIndex.RIGHT_ELBOW, dim3)

    return func(l_hip, l_shoulder, l_elbow), func(r_hip, r_shoulder, r_elbow)


def extract_ankle_angles(pose: PoseLandmarks, dim3: bool = False) -> Tuple[float, float]:
    """Returns (left_ankle_angle, right_ankle_angle) at ankle vertex (Knee - Ankle - Foot Index)."""
    func = calculate_angle_3d if dim3 else calculate_angle_2d
    l_knee = pose.get_numpy(LandmarkIndex.LEFT_KNEE, dim3)
    l_ankle = pose.get_numpy(LandmarkIndex.LEFT_ANKLE, dim3)
    l_foot = pose.get_numpy(LandmarkIndex.LEFT_FOOT_INDEX, dim3)

    r_knee = pose.get_numpy(LandmarkIndex.RIGHT_KNEE, dim3)
    r_ankle = pose.get_numpy(LandmarkIndex.RIGHT_ANKLE, dim3)
    r_foot = pose.get_numpy(LandmarkIndex.RIGHT_FOOT_INDEX, dim3)

    return func(l_knee, l_ankle, l_foot), func(r_knee, r_ankle, r_foot)


def extract_torso_inclination(pose: PoseLandmarks) -> float:
    """Calculates forward torso lean angle from vertical (0 deg = straight upright)."""
    shoulder_mid = pose.get_midpoint(LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER).to_numpy()
    hip_mid = pose.get_midpoint(LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP).to_numpy()
    return calculate_segment_angle_to_vertical(top=shoulder_mid, bottom=hip_mid)


def extract_all_angles(pose: PoseLandmarks) -> Dict[str, float]:
    """Computes a dictionary of all standard biomechanical angles for the current frame."""
    l_knee, r_knee = extract_knee_angles(pose)
    l_hip, r_hip = extract_hip_angles(pose)
    l_elbow, r_elbow = extract_elbow_angles(pose)
    l_shoulder, r_shoulder = extract_shoulder_angles(pose)
    l_ankle, r_ankle = extract_ankle_angles(pose)
    torso_lean = extract_torso_inclination(pose)

    return {
        "left_knee": l_knee,
        "right_knee": r_knee,
        "mean_knee": (l_knee + r_knee) / 2.0,
        "left_hip": l_hip,
        "right_hip": r_hip,
        "mean_hip": (l_hip + r_hip) / 2.0,
        "left_elbow": l_elbow,
        "right_elbow": r_elbow,
        "mean_elbow": (l_elbow + r_elbow) / 2.0,
        "left_shoulder": l_shoulder,
        "right_shoulder": r_shoulder,
        "mean_shoulder": (l_shoulder + r_shoulder) / 2.0,
        "left_ankle": l_ankle,
        "right_ankle": r_ankle,
        "mean_ankle": (l_ankle + r_ankle) / 2.0,
        "torso_inclination": torso_lean,
    }

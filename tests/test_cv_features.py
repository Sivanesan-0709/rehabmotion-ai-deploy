"""
Unit Tests for Computer Vision, Angles, Geometry, and Temporal Feature Modules
"""

import time
import numpy as np
import pytest

from cv.landmarks import LandmarkIndex, Point3D, PoseLandmarks
from cv.pose_estimator import create_pose_estimator
from features.angles import (
    calculate_angle_2d,
    calculate_angle_3d,
    calculate_segment_angle_to_vertical,
    extract_all_angles,
    extract_knee_angles,
    extract_elbow_angles,
    extract_hip_angles,
    extract_torso_inclination,
)
from features.geometry import (
    calculate_symmetry_index,
    calculate_knee_valgus,
    calculate_horizontal_levelness,
    calculate_elbow_torso_drift,
    extract_all_geometric_features,
)
from features.temporal import TemporalFeatureTracker
from features.extractor import FeatureExtractor


from tests.helpers import create_synthetic_pose


def test_angle_2d_right_angle():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    angle = calculate_angle_2d(a, b, c)
    assert pytest.approx(angle, 0.01) == 90.0


def test_angle_2d_straight_line():
    a = np.array([0.0, -1.0])
    b = np.array([0.0, 0.0])
    c = np.array([0.0, 1.0])
    angle = calculate_angle_2d(a, b, c)
    assert pytest.approx(angle, 0.01) == 180.0


def test_angle_3d_orthogonality():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    angle = calculate_angle_3d(a, b, c)
    assert pytest.approx(angle, 0.01) == 90.0


def test_vertical_inclination():
    top = np.array([0.0, -1.0])
    bottom = np.array([0.0, 0.0])
    angle = calculate_segment_angle_to_vertical(top, bottom)
    assert pytest.approx(angle, 0.01) == 0.0


def test_symmetry_index():
    assert pytest.approx(calculate_symmetry_index(90.0, 90.0), 0.01) == 100.0
    assert calculate_symmetry_index(100.0, 50.0) < 60.0


def test_pose_landmarks_normalization():
    pose = create_synthetic_pose()
    norm_vec = pose.get_normalized_vector()
    assert len(norm_vec) == 99
    assert isinstance(norm_vec, np.ndarray)


def test_temporal_feature_tracker():
    tracker = TemporalFeatureTracker(max_frames=10)
    t0 = 100.0
    for i in range(5):
        angles = {"mean_knee": 160.0 - i * 10.0, "torso_inclination": 5.0}
        tracker.update(angles, t0 + i * 0.033)

    vel = tracker.get_velocity("knee")
    assert vel < 0.0  # Angle is decreasing (knee bending)
    smoothness = tracker.compute_smoothness_score("knee")
    assert 0.0 <= smoothness <= 100.0


def test_unified_feature_extractor():
    extractor = FeatureExtractor()
    pose = create_synthetic_pose()
    feats = extractor.extract(pose)

    assert feats.is_valid is True
    assert "mean_knee" in feats.angles
    assert "overall_symmetry" in feats.geometry
    ml_vec = feats.to_ml_feature_vector()
    assert len(ml_vec) == 68
    assert not np.isnan(ml_vec).any()


def test_mock_pose_estimator():
    estimator = create_pose_estimator(backend="mock", exercise_pose="squat_bottom")
    synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    pose = estimator.process_frame(synthetic_frame)

    assert pose.is_valid is True
    assert pose.detected is True
    l_knee, r_knee = extract_knee_angles(pose)
    assert l_knee < 120.0  # Bent knee in squat bottom
    estimator.close()


def test_mediapipe_backend_detection():
    from cv.pose_estimator import detect_mediapipe_backend
    backend = detect_mediapipe_backend()
    assert backend in ["tasks", "solutions"]


def test_mediapipe_pose_estimator_adapter_and_empty_frame():
    from cv.pose_estimator import create_pose_estimator, MediaPipePoseEstimator
    estimator = create_pose_estimator(backend="mediapipe")
    assert isinstance(estimator, MediaPipePoseEstimator)
    assert estimator.active_backend_type in ["tasks", "solutions"]

    # Test processing empty/blank frame (no human detected)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = estimator.process_frame(blank_frame)

    assert res.is_valid is False
    assert res.detected is False
    assert len(res.landmarks) == 0
    assert res.confidence == 0.0
    estimator.close()


def test_pose_estimator_missing_model_handling():
    from cv.pose_estimator import ensure_pose_landmarker_model
    with pytest.raises(FileNotFoundError):
        ensure_pose_landmarker_model(
            model_name="completely_nonexistent_model_asset.task",
            allow_download=False
        )


def test_landmark_data_structure_and_aliases():
    from cv.landmarks import Point3D, PoseLandmark, PoseLandmarks, PoseResult

    # Verify alias equivalence
    assert PoseLandmark is Point3D
    assert PoseResult is PoseLandmarks

    pt = Point3D(x=0.5, y=0.4, z=0.1, visibility=0.9, presence=0.95)
    assert pt.x == 0.5
    assert pt.visibility == 0.9
    assert pt.presence == 0.95

    pose = PoseLandmarks(
        landmarks={0: pt},
        timestamp=123.456,
        is_valid=True,
        confidence=0.9,
        world_landmarks={0: pt}
    )
    assert pose.detected is True
    assert pose.world_landmarks is not None
    assert len(pose.world_landmarks) == 1


def test_pose_estimator_factory_backends():
    from cv.pose_estimator import create_pose_estimator, YOLOPoseEstimator, MockPoseEstimator

    yolo_est = create_pose_estimator(backend="yolo")
    assert isinstance(yolo_est, YOLOPoseEstimator)
    yolo_est.close()

    mock_est = create_pose_estimator(backend="mock")
    assert isinstance(mock_est, MockPoseEstimator)
    mock_est.close()

    with pytest.raises(ValueError, match="Unsupported pose estimation backend"):
        create_pose_estimator(backend="unsupported_engine_xyz")


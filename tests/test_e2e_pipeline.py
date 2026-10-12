"""
End-to-End Integration Tests for AI Exercise Form & Posture Intelligence Platform
"""

import tempfile
import time
from pathlib import Path
import numpy as np
import pytest

from app.config import CONFIG
from cv.detector import VideoStream
from cv.pose_estimator import create_pose_estimator
from cv.visualization import HUDVisualizer
from database.db import DatabaseManager
from database.repositories import (
    FeedbackRepository,
    MotionProfileRepository,
    RepetitionRepository,
    SessionRepository,
    UserRepository,
)
from exercises import get_exercise_engine, list_available_exercises
from features.extractor import FeatureExtractor
from intelligence.feedback import MultimodalFeedbackManager
from intelligence.motion_dna import MotionDNAEngine
from intelligence.rep_counter import RepCounter
from intelligence.scoring import FormScoringEngine
from models.model_registry import ModelRegistry
from tests.helpers import create_synthetic_pose


@pytest.fixture
def e2e_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "e2e_test.db"
        manager = DatabaseManager(db_path=db_path)
        yield manager


def test_full_workout_lifecycle_e2e(e2e_db):
    """
    Simulates a complete workout session from frame capture to final analytics report.
    """
    # 1. Initialize DB & Repositories
    user_repo = UserRepository(e2e_db)
    session_repo = SessionRepository(e2e_db)
    rep_repo = RepetitionRepository(e2e_db)
    profile_repo = MotionProfileRepository(e2e_db)
    feedback_repo = FeedbackRepository(e2e_db)

    user = user_repo.get_or_create_user("Test E2E Athlete")
    exercise_name = "Squat"
    session_id = session_repo.create_session(user["id"], exercise_name)

    # 2. Initialize Engines
    exercise_engine = get_exercise_engine(exercise_name)
    feature_extractor = FeatureExtractor()
    rep_counter = RepCounter(exercise_engine, min_rep_duration_s=0.3, min_rep_frames=3)
    motion_dna_engine = MotionDNAEngine(min_calibration_reps=3)
    scoring_engine = FormScoringEngine()
    feedback_manager = MultimodalFeedbackManager(voice_enabled=False, cooldown_seconds=1.0)
    visualizer = HUDVisualizer()

    # 3. Simulate 5 repetitions (4 clean, 1 with shallow depth)
    completed_summaries = []
    base_t = 1000.0
    pose = create_synthetic_pose()

    for rep_idx in range(1, 6):
        is_flawed = (rep_idx == 5)
        # Standing
        f1 = feature_extractor.extract(pose)
        f1.timestamp = base_t + (rep_idx * 10) + 0.0
        f1.angles["mean_knee"] = 170.0
        rep_counter.process_frame(f1, pose)

        # Descending
        f2 = feature_extractor.extract(pose)
        f2.timestamp = base_t + (rep_idx * 10) + 0.4
        f2.angles["mean_knee"] = 120.0
        rep_counter.process_frame(f2, pose)

        # Bottom (84 deg for clean, 94 deg for shallow depth)
        f3 = feature_extractor.extract(pose)
        f3.timestamp = base_t + (rep_idx * 10) + 0.8
        f3.angles["mean_knee"] = 84.0 if not is_flawed else 94.0
        rep_counter.process_frame(f3, pose)

        # Ascending
        f4 = feature_extractor.extract(pose)
        f4.timestamp = base_t + (rep_idx * 10) + 1.2
        f4.angles["mean_knee"] = 135.0
        rep_counter.process_frame(f4, pose)

        # Standing Complete
        for step in range(5):
            f5 = feature_extractor.extract(pose)
            f5.timestamp = base_t + (rep_idx * 10) + 1.6 + step * 0.1
            f5.angles["mean_knee"] = 168.0
            phase, label, score, rep_summary, issues = rep_counter.process_frame(f5, pose)
            if rep_summary:
                rep_repo.save_repetition(session_id, rep_summary)
                completed_summaries.append(rep_summary)
                break

    # 4. Verify Repetition Counts & Integrity
    assert rep_counter.total_reps == 5
    assert len(completed_summaries) == 5
    assert completed_summaries[0].form_score >= 80.0
    # Last rep was shallow depth
    assert completed_summaries[4].form_score < completed_summaries[0].form_score

    # 5. Build Motion DNA Baseline
    profile = motion_dna_engine.build_profile(user["id"], exercise_name, completed_summaries[:4], timestamp=base_t + 100.0)
    assert profile is not None
    assert profile.total_baseline_reps == 4
    profile_repo.save_profile(profile)

    # 6. Verify Deviation on Flawed Rep
    dev = motion_dna_engine.calculate_deviation(completed_summaries[4], profile)
    assert dev.overall_deviation_pct > 0.0

    # 7. End Session and Verify Database Stats
    session_repo.end_session(session_id)
    session_data = session_repo.get_session(session_id)
    assert session_data["total_reps"] == 5
    assert session_data["average_score"] > 60.0

    # 8. Verify Visual HUD Rendering without crash
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = visualizer.draw_hud(
        frame=dummy_frame,
        exercise_name="Squat",
        phase_name="Standing",
        rep_count=5,
        form_score=88.0,
        fps=30.0,
        confidence=0.95
    )
    assert rendered.shape == (720, 1280, 3)


def test_all_exercises_instantiation():
    for name in list_available_exercises():
        engine = get_exercise_engine(name)
        assert engine.name == name
        joints = engine.get_primary_joints_to_highlight()
        assert len(joints) > 0


def test_model_registry_all_supported():
    registry = ModelRegistry()
    for model_name in registry.SUPPORTED_MODELS:
        model = registry.get_model(model_name)
        assert model is not None


def test_app_state_initialization():
    from app.state import AppState
    state = AppState()
    assert state.pose_estimator is not None
    assert state.active_user_name == "Default Athlete"
    assert state.exercise_name == "Squat"

    # Test backend switching to mock
    state.set_pose_backend("mock")
    assert state.pose_backend == "mock"

    # Switch back to mediapipe
    state.set_pose_backend("mediapipe")
    assert state.pose_backend == "mediapipe"
    state.pose_estimator.close()


"""
Unit & Integration Tests for SQLite Database and Repository Layer
"""

import tempfile
import time
from pathlib import Path
import pytest

from database.db import DatabaseManager
from database.repositories import (
    UserRepository,
    SessionRepository,
    RepetitionRepository,
    MotionProfileRepository,
    FeedbackRepository,
)
from exercises.base import RepetitionSummary
from intelligence.motion_dna import MetricDistribution, MotionDNAProfile


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_exercise.db"
        manager = DatabaseManager(db_path=db_path)
        yield manager


def test_user_repository(temp_db):
    user_repo = UserRepository(temp_db)
    user = user_repo.get_or_create_user("Alex Johnson")
    assert user["name"] == "Alex Johnson"
    assert "usr_" in user["id"]

    # Retrieval of existing user
    user2 = user_repo.get_or_create_user("Alex Johnson")
    assert user2["id"] == user["id"]

    all_users = user_repo.list_all_users()
    assert len(all_users) == 1


def test_session_and_repetition_repository(temp_db):
    user_repo = UserRepository(temp_db)
    session_repo = SessionRepository(temp_db)
    rep_repo = RepetitionRepository(temp_db)

    user = user_repo.get_or_create_user("Sam Athlete")
    session_id = session_repo.create_session(user["id"], "Squat")
    assert "ses_" in session_id

    # Create dummy completed reps
    rep1 = RepetitionSummary(
        rep_id="rep_01",
        exercise_name="Squat",
        rep_number=1,
        duration_seconds=2.4,
        form_score=92.0,
        score_breakdown={"joint_alignment": 90.0, "range_of_motion": 95.0},
        range_of_motion=95.0,
        symmetry_score=96.0,
        smoothness_score=88.0,
        torso_stability_score=90.0,
        issues_detected=[],
        is_correct=True,
        peak_angles={"min_knee_angle": 84.0},
        timestamp_start=100.0,
        timestamp_end=102.4,
    )
    rep_repo.save_repetition(session_id, rep1, deviation_pct=5.2)

    session_repo.end_session(session_id)
    session_data = session_repo.get_session(session_id)
    assert session_data["total_reps"] == 1
    assert session_data["correct_reps"] == 1
    assert session_data["average_score"] == 92.0

    reps = rep_repo.get_session_repetitions(session_id)
    assert len(reps) == 1
    assert reps[0]["score"] == 92.0
    assert reps[0]["issues"] == []


def test_motion_profile_repository(temp_db):
    user_repo = UserRepository(temp_db)
    profile_repo = MotionProfileRepository(temp_db)
    user = user_repo.get_or_create_user("Motion User")

    metrics = {
        "range_of_motion": MetricDistribution(mean=92.0, std=4.5, median=92.0, p25=89.0, p75=95.0, min_val=85.0, max_val=98.0),
        "duration_seconds": MetricDistribution(mean=2.3, std=0.2, median=2.3, p25=2.1, p75=2.5, min_val=2.0, max_val=2.7),
    }
    profile = MotionDNAProfile(
        user_id=user["id"],
        exercise_name="Squat",
        total_baseline_reps=10,
        created_at=time.time(),
        updated_at=time.time(),
        metrics=metrics,
        fingerprint_scores={"Range Of Motion": 92.0, "Cadence Stability": 94.0},
        consistency_index=94.5,
    )

    profile_repo.save_profile(profile)
    retrieved = profile_repo.get_profile(user["id"], "Squat")
    assert retrieved is not None
    assert retrieved.consistency_index == 94.5
    assert retrieved.metrics["range_of_motion"].mean == 92.0


def test_feedback_repository(temp_db):
    user_repo = UserRepository(temp_db)
    session_repo = SessionRepository(temp_db)
    feedback_repo = FeedbackRepository(temp_db)

    user = user_repo.get_or_create_user("Feedback User")
    session_id = session_repo.create_session(user["id"], "Squat")

    feedback_repo.log_event(session_id, "Keep chest upright", rep_number=2, priority="warning")

    events = feedback_repo.get_session_events(session_id)
    assert len(events) == 1
    assert events[0]["message"] == "Keep chest upright"

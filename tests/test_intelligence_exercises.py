"""
Unit Tests for Exercises, Repetition State Machines, Scoring Engine & Motion DNA
"""

import time
import pytest
from exercises import get_exercise_engine, list_available_exercises
from exercises.base import MovementPhase
from features.extractor import FeatureExtractor, KinematicFeatures
from intelligence.rep_counter import RepCounter
from intelligence.scoring import FormScoringEngine
from intelligence.motion_dna import MotionDNAEngine
from intelligence.feedback import MultimodalFeedbackManager
from tests.helpers import create_synthetic_pose


def test_list_exercises():
    exercises = list_available_exercises()
    assert "Squat" in exercises
    assert "Bicep Curl" in exercises
    assert "Push-up" in exercises
    assert "Lunge" in exercises
    assert "Shoulder Press" in exercises


def test_squat_state_machine_flow():
    squat = get_exercise_engine("Squat")
    extractor = FeatureExtractor()

    # 1. Standing
    pose_standing = create_synthetic_pose(knee_angle_deg=170.0)
    feats_standing = extractor.extract(pose_standing)
    phase, label, issues = squat.process_frame(feats_standing, pose_standing)
    assert phase in [MovementPhase.START, MovementPhase.TRANSITION_OUT]

    # 2. Descending to bottom (2 frames to transition to inflection)
    feats_bottom = extractor.extract(pose_standing)
    feats_bottom.angles["mean_knee"] = 85.0  # deep squat
    squat.process_frame(feats_bottom, pose_standing)
    phase, label, issues = squat.process_frame(feats_bottom, pose_standing)
    assert phase == MovementPhase.INFLECTION

    # 3. Ascending back to standing
    feats_standing.timestamp += 1.5
    feats_standing.angles["mean_knee"] = 165.0
    for _ in range(4):
        phase, label, issues = squat.process_frame(feats_standing, pose_standing)

    assert phase == MovementPhase.COMPLETED


def test_rep_counter_lifecycle():
    squat = get_exercise_engine("Squat")
    counter = RepCounter(exercise=squat, min_rep_duration_s=0.3, min_rep_frames=3)
    extractor = FeatureExtractor()
    pose = create_synthetic_pose()

    t = 100.0
    # 1. Standing
    f1 = extractor.extract(pose)
    f1.timestamp = t
    f1.angles["mean_knee"] = 170.0
    counter.process_frame(f1, pose)

    # 2. Descending
    f2 = extractor.extract(pose)
    f2.timestamp = t + 0.3
    f2.angles["mean_knee"] = 120.0
    counter.process_frame(f2, pose)

    # 3. Bottom (Inflection)
    f3 = extractor.extract(pose)
    f3.timestamp = t + 0.6
    f3.angles["mean_knee"] = 85.0
    counter.process_frame(f3, pose)

    # 4. Ascending
    f4 = extractor.extract(pose)
    f4.timestamp = t + 0.9
    f4.angles["mean_knee"] = 130.0
    counter.process_frame(f4, pose)

    # 5. Complete
    rep_done = None
    for step in range(6):
        f5 = extractor.extract(pose)
        f5.timestamp = t + 1.2 + step * 0.1
        f5.angles["mean_knee"] = 168.0
        phase, label, score, rep_summary, issues = counter.process_frame(f5, pose)
        if rep_summary is not None:
            rep_done = rep_summary
            break

    assert counter.total_reps >= 1
    assert len(counter.rep_history) >= 1
    assert counter.rep_history[0].exercise_name == "Squat"


def test_form_scoring_engine():
    engine = FormScoringEngine()
    explanation = engine.evaluate_score(
        joint_alignment=90.0,
        range_of_motion=95.0,
        posture_stability=90.0,
        movement_smoothness=88.0,
        symmetry=96.0,
        issues=[]
    )
    assert explanation.overall_score >= 85.0
    assert "Joint Alignment" in explanation.breakdown
    assert len(explanation.positive_feedback) > 0


def test_motion_dna_profiling_and_deviation():
    engine = MotionDNAEngine(min_calibration_reps=3)
    squat = get_exercise_engine("Squat")
    extractor = FeatureExtractor()
    pose = create_synthetic_pose()

    # Generate 4 mock rep summaries
    reps = []
    for i in range(4):
        f = extractor.extract(pose)
        f.angles["mean_knee"] = 85.0
        rep = squat.evaluate_completed_rep(rep_number=i + 1, duration=2.2, features_history=[f] * 15)
        reps.append(rep)

    profile = engine.build_profile(user_id="user_123", exercise_name="Squat", reps=reps, timestamp=time.time())
    assert profile is not None
    assert profile.total_baseline_reps == 4
    assert "range_of_motion" in profile.metrics
    assert profile.consistency_index >= 50.0

    # Test deviation calculation with normal rep
    dev = engine.calculate_deviation(reps[0], profile)
    assert dev.overall_deviation_pct < 50.0

    # Test abnormal rep (extreme duration / poor ROM)
    bad_rep = squat.evaluate_completed_rep(rep_number=5, duration=8.5, features_history=[f] * 15)
    bad_rep.range_of_motion = 30.0
    bad_rep.symmetry_score = 40.0
    dev_bad = engine.calculate_deviation(bad_rep, profile)
    assert dev_bad.is_significant_deviation is True


def test_feedback_cooldown():
    manager = MultimodalFeedbackManager(voice_enabled=False, cooldown_seconds=2.0)
    t0 = 100.0
    # First message should emit
    assert manager.emit_feedback("Keep back straight", timestamp=t0) is True
    # Immediate repeat within cooldown should be blocked
    assert manager.emit_feedback("Keep back straight", timestamp=t0 + 0.5) is False
    # Urgent message should bypass normal cooldown
    assert manager.emit_feedback("Dangerous spine bend", priority="urgent", timestamp=t0 + 0.6) is True
    # After cooldown period expires (2.0s after last message at 100.6)
    assert manager.emit_feedback("Good rep", timestamp=t0 + 3.0) is True

"""
AI Exercise Form & Posture Intelligence Platform
Unit and Integration Tests for Live Coach Session Lifecycle, State Transitions & Error Recovery
"""

import pytest
from app.state import get_app_state
from cv.detector import VideoStream
from cv.landmarks import Point3D, PoseLandmarks, LandmarkIndex


def test_start_session_lifecycle_transition():
    """Verify that starting a workout session sets active state, creates SQLite session, and persists."""
    state = get_app_state()
    state.set_user("Lifecycle Test Athlete")
    state.set_exercise("Squat")
    
    # Ensure state starts idle
    if state.is_workout_active:
        state.end_workout_session()
    
    assert not state.is_workout_active
    assert state.active_session_id is None
    
    # Trigger start session
    state.start_workout_session()
    assert state.is_workout_active is True
    assert state.active_session_id is not None
    sess_id = state.active_session_id
    
    # Verify session recorded in repository
    sess = state.session_repo.get_session(sess_id)
    assert sess is not None
    assert sess["exercise"] == "Squat"
    assert sess["user_id"] == state.active_user_id
    assert sess["end_time"] is None  # Still active


def test_active_session_persists_across_simulated_reruns():
    """Verify that state.is_workout_active and active_session_id remain intact across reruns."""
    state = get_app_state()
    state.set_user("Rerun Test Athlete")
    state.set_exercise("Push-up")
    
    state.start_workout_session()
    sess_id = state.active_session_id
    assert state.is_workout_active is True
    assert sess_id is not None
    
    # Simulate multiple Streamlit execution / rerun cycles by querying get_app_state()
    for _ in range(5):
        s = get_app_state()
        assert s.is_workout_active is True
        assert s.active_session_id == sess_id
        assert s.exercise_name == "Push-up"
    
    # Clean up
    state.end_workout_session()
    assert not state.is_workout_active


def test_end_session_lifecycle_and_data_persistence():
    """Verify that ending a session updates session record with summary stats and marks inactive."""
    state = get_app_state()
    state.set_user("End Session Athlete")
    state.set_exercise("Bicep Curl")
    
    state.start_workout_session()
    sess_id = state.active_session_id
    assert sess_id is not None
    
    # Simulate completed repetition
    from exercises.base import RepetitionSummary
    rep = RepetitionSummary(
        rep_id="rep_test_01",
        exercise_name="Bicep Curl",
        rep_number=1,
        duration_seconds=2.2,
        form_score=94.5,
        score_breakdown={"rom": 95.0, "stability": 94.0},
        range_of_motion=100.0,
        symmetry_score=92.0,
        smoothness_score=91.0,
        torso_stability_score=95.0,
        issues_detected=[],
        is_correct=True,
        peak_angles={"elbow": 145.0},
        timestamp_start=100.0,
        timestamp_end=102.2
    )
    state.rep_counter.rep_history.append(rep)
    state.rep_counter.total_reps = 1
    state.rep_counter.correct_reps = 1
    state.rep_repo.save_repetition(sess_id, rep, deviation_pct=5.0)
    
    # End session
    state.end_workout_session()
    assert state.is_workout_active is False
    assert state.active_session_id is None
    
    # Check database persistence
    sess = state.session_repo.get_session(sess_id)
    assert sess is not None
    assert sess["end_time"] is not None
    assert sess["total_reps"] == 1
    assert sess["average_score"] == pytest.approx(94.5, 0.1)


def test_reset_counter_leaves_active_session_intact():
    """Verify that resetting repetition counter does not cancel or tear down active session."""
    state = get_app_state()
    state.set_user("Reset Counter Athlete")
    state.set_exercise("Lunge")
    
    state.start_workout_session()
    sess_id = state.active_session_id
    assert sess_id is not None
    
    state.rep_counter.total_reps = 5
    state.rep_counter.correct_reps = 4
    
    # Reset rep counter
    state.rep_counter.reset()
    assert state.rep_counter.total_reps == 0
    assert state.rep_counter.correct_reps == 0
    assert len(state.rep_counter.rep_history) == 0
    
    # Session must still remain active
    assert state.is_workout_active is True
    assert state.active_session_id == sess_id
    
    # Clean up
    state.end_workout_session()
    assert not state.is_workout_active


def test_camera_initialization_failure_surfacing(monkeypatch):
    """Verify camera initialization errors are cleanly captured and handled without crashing."""
    import cv2
    
    # Simulate hardware camera open failure
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: type("MockCap", (), {
        "isOpened": lambda self: False,
        "release": lambda self: None,
        "set": lambda self, *a: None,
        "get": lambda self, *a: 0,
    })())
    
    stream = VideoStream(src=999, is_synthetic=False)
    assert stream.is_open is False
    
    # Verify synthetic fallback works reliably
    synth_stream = VideoStream(is_synthetic=True)
    assert synth_stream.is_open is True
    ret, frame, ts = synth_stream.read()
    assert ret is True
    assert frame is not None
    synth_stream.release()


def test_exercise_switching_and_prescription_mode():
    """Verify that switching exercise updates exercise_engine and rep_counter cleanly."""
    state = get_app_state()
    state.set_user("Exercise Switch Athlete")
    state.set_exercise("Squat")
    assert state.exercise_name == "Squat"
    assert state.exercise_engine.name == "Squat"
    
    # Start session
    state.start_workout_session()
    sess_id = state.active_session_id
    assert state.is_workout_active is True
    
    # Switch to Push-up
    state.set_exercise("Push-up")
    assert state.exercise_name == "Push-up"
    assert state.exercise_engine.name == "Push-up"
    assert state.rep_counter.exercise.name == "Push-up"
    
    # Session must remain active
    assert state.is_workout_active is True
    assert state.active_session_id == sess_id
    
    # Toggle Auto-Mode
    state.auto_mode = True
    assert state.auto_mode is True
    state.auto_mode = False
    assert state.auto_mode is False
    
    # End session cleanly
    state.end_workout_session()
    assert not state.is_workout_active


"""
AI Exercise Form & Posture Intelligence Platform
Unit and Integration Tests for Dashboard Pages, Demo Data & Theme Engine
"""

import os
import pandas as pd
import pytest

from app.config import DATA_DIR
from app.state import get_app_state
from dashboard.demo.demo_data import (
    get_demo_kpi_summary,
    get_demo_weekly_activity,
    get_demo_recent_sessions,
    get_demo_exercise_catalog,
    get_demo_achievements,
    get_demo_leaderboard,
    get_demo_motion_dna_profile,
    get_demo_workout_plan,
    get_demo_form_tip,
)
from dashboard.theme import (
    ThemeTokens,
    render_top_bar,
    render_kpi_card,
    render_status_pill,
    render_demo_badge,
    render_info_banner,
    render_section_header,
    safe_render,
)
from dashboard.voice import get_voice_coach, VoiceCoach
from training.evaluate import AcademicEvaluator


def test_demo_data_structures():
    """Verify demo data generator produces expected schemas and non-empty records."""
    kpi = get_demo_kpi_summary()
    assert "total_reps" in kpi
    assert "avg_form_score" in kpi
    assert kpi["total_reps"] > 0

    weekly = get_demo_weekly_activity()
    assert len(weekly) == 7
    assert "Day" in weekly.columns
    assert "Repetitions" in weekly.columns

    sessions = get_demo_recent_sessions()
    assert len(sessions) >= 5
    assert all("exercise" in s and "avg_score" in s for s in sessions)

    catalog = get_demo_exercise_catalog()
    assert len(catalog) >= 5
    assert any(ex["name"] == "Squat" for ex in catalog)

    achievements = get_demo_achievements()
    assert len(achievements) >= 6

    leaderboard = get_demo_leaderboard()
    assert len(leaderboard) >= 5

    profile = get_demo_motion_dna_profile("Squat")
    assert "scores" in profile
    assert len(profile["metrics_table"]) > 0

    plan = get_demo_workout_plan()
    assert len(plan) == 7

    tip = get_demo_form_tip()
    assert "title" in tip and "tip" in tip


def test_theme_rendering_helpers():
    """Ensure HTML component generators produce valid non-empty markup."""
    card_html = render_kpi_card("Test Metric", "95.5%", "High accuracy", icon="🎯")
    assert "Test Metric" in card_html
    assert "95.5%" in card_html
    assert "kpi-container" in card_html

    pill_html = render_status_pill("Active", variant="success")
    assert "status-pill" in pill_html
    assert "Active" in pill_html

    demo_badge_html = render_demo_badge()
    assert "DEMO DATA" in demo_badge_html

    banner_html = render_info_banner("Test Announcement")
    assert "Test Announcement" in banner_html
    assert "info-banner" in banner_html

    header_html = render_section_header("Section Title", "Section Subtitle")
    assert "Section Title" in header_html
    assert "Section Subtitle" in header_html


def test_safe_render_decorator():
    """Verify safe_render catches unexpected exceptions without crashing."""
    @safe_render("Faulty Component")
    def faulty_func():
        raise ValueError("Simulated render fault")

    # Should safely catch the error and not raise
    faulty_func()


def test_voice_coach_instance():
    """Verify voice coach singleton and priority queue."""
    coach = get_voice_coach()
    assert isinstance(coach, VoiceCoach)
    assert coach.enabled is not None

    # Test speech dispatches
    coach.speak_cue("Unit test voice cue")
    coach.check_and_announce_rep(1, 95.0)
    coach.check_and_announce_rep(2, 92.0)
    assert coach.last_announced_rep == 2


def test_demo_seeding_creates_only_demo_user():
    """Verify that demo data seeding writes records ONLY under 'Demo Athlete (DEMO)'."""
    state = get_app_state()

    # Create a real user
    real_user = state.user_repo.get_or_create_user("Real Test Athlete 01")
    real_user_id = real_user["id"]
    initial_real_sessions = state.session_repo.list_user_sessions(real_user_id)

    # Seed demo data under Demo Athlete
    demo_user = state.user_repo.get_or_create_user("Demo Athlete (DEMO)")
    demo_user_id = demo_user["id"]

    demo_items = get_demo_recent_sessions()
    for item in demo_items:
        sess_id = state.session_repo.create_session(demo_user_id, item["exercise"])
        state.session_repo.end_session(sess_id)

    # Verify sessions belong strictly to Demo Athlete
    demo_sessions = state.session_repo.list_user_sessions(demo_user_id)
    assert len(demo_sessions) >= len(demo_items)

    # Verify real user sessions are completely untouched
    final_real_sessions = state.session_repo.list_user_sessions(real_user_id)
    assert len(final_real_sessions) == len(initial_real_sessions)


def test_clearing_demo_leaves_real_users_intact():
    """Verify that clearing demo records deletes ONLY Demo Athlete records and preserves real users."""
    state = get_app_state()

    # Create real user with a session
    real_user = state.user_repo.get_or_create_user("Real Test Athlete Unique")
    real_user_id = real_user["id"]
    initial_real_count = len(state.session_repo.list_user_sessions(real_user_id))

    real_sess_id = state.session_repo.create_session(real_user_id, "Squat")
    state.session_repo.end_session(real_sess_id)

    # Ensure Demo Athlete exists with sessions
    demo_user = state.user_repo.get_or_create_user("Demo Athlete (DEMO)")
    demo_user_id = demo_user["id"]
    demo_sess_id = state.session_repo.create_session(demo_user_id, "Push-up")
    state.session_repo.end_session(demo_sess_id)

    # Clear ONLY Demo Athlete records
    state.session_repo.delete_user_sessions(demo_user_id)
    state.user_repo.delete_user(demo_user_id)

    # Verify Demo Athlete is cleared
    assert len(state.session_repo.list_user_sessions(demo_user_id)) == 0

    # Verify Real Athlete's session is still present and intact
    real_sessions_after = state.session_repo.list_user_sessions(real_user_id)
    assert len(real_sessions_after) == initial_real_count + 1
    assert any(s["id"] == real_sess_id for s in real_sessions_after)


def test_academic_benchmarks_never_touch_sqlite_or_demo_user(monkeypatch):
    """
    Verify that benchmark loaders operate purely on artifact/dataset files
    and NEVER touch SQLite sessions, repetitions, or demo user records.
    Fails immediately if any database access occurs during benchmark loading.
    """
    state = get_app_state()

    # Create demo user and real user in SQLite
    demo_user = state.user_repo.get_or_create_user("Demo Athlete (DEMO)")
    demo_user_id = demo_user["id"]
    sess_id = state.session_repo.create_session(demo_user_id, "Squat")
    state.session_repo.end_session(sess_id)

    db_touched = {"touched": False}

    def forbidden_db_call(*args, **kwargs):
        db_touched["touched"] = True
        raise AssertionError("Academic benchmark loader violated isolation by querying the SQLite database!")

    # Patch database repository query methods
    monkeypatch.setattr(state.session_repo, "list_user_sessions", forbidden_db_call)
    monkeypatch.setattr(state.session_repo, "get_session", forbidden_db_call)
    monkeypatch.setattr(state.rep_repo, "get_user_all_repetitions", forbidden_db_call)
    monkeypatch.setattr(state.rep_repo, "get_session_repetitions", forbidden_db_call)

    # 1. Load synthetic benchmark summary
    summary_synth = AcademicEvaluator.load_benchmark_summary(source="synthetic")
    assert summary_synth is not None
    assert "Model" in summary_synth.columns
    assert "Accuracy (%)" in summary_synth.columns

    # 2. Load classification reports
    reports = AcademicEvaluator.get_classification_reports(source="synthetic")
    assert isinstance(reports, dict)

    # 3. Scan real dataset info
    real_info = AcademicEvaluator.get_real_dataset_info()
    assert "num_subjects" in real_info

    # Verify no database access occurred
    assert not db_touched["touched"], "Benchmark loader accessed SQLite database during execution!"

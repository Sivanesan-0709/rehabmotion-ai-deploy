"""
AI Exercise Form & Posture Intelligence Platform
Global Application State & Session Orchestrator
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import streamlit as st

from app.config import CONFIG
from cv.detector import VideoStream
from cv.landmarks import PoseLandmarks
from cv.pose_estimator import BasePoseEstimator, create_pose_estimator
from cv.visualization import HUDVisualizer
from database.db import DB
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
from intelligence.motion_dna import MotionDNAEngine, MotionDNAProfile, RepetitionDeviation
from intelligence.rep_counter import RepCounter
from models.model_registry import REGISTRY


class AppState:
    """Manages full lifecycle of active workout sessions and state caching."""

    def __init__(self):
        # Repositories
        self.user_repo = UserRepository(DB)
        self.session_repo = SessionRepository(DB)
        self.rep_repo = RepetitionRepository(DB)
        self.profile_repo = MotionProfileRepository(DB)
        self.feedback_repo = FeedbackRepository(DB)

        # Active User & Session
        self.active_user_name: str = "Default Athlete"
        self.active_user_id: str = self.user_repo.get_or_create_user(self.active_user_name)["id"]
        self.active_session_id: Optional[str] = None
        self.is_workout_active: bool = False

        # Configuration & Engines
        self.exercise_name: str = "Squat"
        self.exercise_engine = get_exercise_engine(self.exercise_name)
        self.pose_backend: str = "mediapipe"
        self.pose_estimator: BasePoseEstimator = create_pose_estimator(backend=self.pose_backend)
        self.feature_extractor = FeatureExtractor()
        self.rep_counter = RepCounter(self.exercise_engine)
        self.visualizer = HUDVisualizer()
        self.feedback_manager = MultimodalFeedbackManager(
            voice_enabled=CONFIG.feedback.voice_enabled,
            cooldown_seconds=CONFIG.feedback.cooldown_seconds
        )
        self.motion_dna_engine = MotionDNAEngine()
        self.active_profile: Optional[MotionDNAProfile] = None
        self.latest_deviation: Optional[RepetitionDeviation] = None
        self.auto_mode: bool = True

    def set_user(self, user_name: str):
        self.active_user_name = user_name
        self.active_user_id = self.user_repo.get_or_create_user(user_name)["id"]
        self.refresh_profile()

    def set_exercise(self, exercise_name: str):
        if exercise_name != self.exercise_name:
            self.exercise_name = exercise_name
            self.exercise_engine = get_exercise_engine(exercise_name)
            self.rep_counter = RepCounter(self.exercise_engine)
            self.feature_extractor.reset()
            self.refresh_profile()

    def set_pose_backend(self, backend: str):
        if backend != self.pose_backend:
            self.pose_backend = backend
            self.pose_estimator.close()
            self.pose_estimator = create_pose_estimator(backend=backend)

    def refresh_profile(self):
        self.active_profile = self.profile_repo.get_profile(self.active_user_id, self.exercise_name)

    def start_workout_session(self):
        self.active_session_id = self.session_repo.create_session(self.active_user_id, self.exercise_name)
        self.is_workout_active = True
        self.rep_counter.reset()
        self.feature_extractor.reset()
        self.refresh_profile()

    def end_workout_session(self):
        if self.active_session_id:
            self.session_repo.end_session(self.active_session_id)
            # Check if we can build/update Motion DNA baseline
            all_user_reps = self.rep_repo.get_user_all_repetitions(self.active_user_id, self.exercise_name)
            if len(all_user_reps) >= CONFIG.motion_dna.min_calibration_reps:
                # Convert to RepetitionSummaries
                summaries = []
                for r in all_user_reps:
                    from exercises.base import RepetitionSummary
                    summary = RepetitionSummary(
                        rep_id=r["id"],
                        exercise_name=self.exercise_name,
                        rep_number=r["rep_number"],
                        duration_seconds=r["duration"],
                        form_score=r["score"],
                        score_breakdown=r["breakdown"],
                        range_of_motion=r["rom"],
                        symmetry_score=r["symmetry"],
                        smoothness_score=r["smoothness"],
                        torso_stability_score=r["torso_stability"],
                        issues_detected=r["issues"],
                        is_correct=(r["status"] == "Correct"),
                        peak_angles={},
                        timestamp_start=r["timestamp_start"],
                        timestamp_end=r["timestamp_end"]
                    )
                    summaries.append(summary)

                new_profile = self.motion_dna_engine.build_profile(
                    user_id=self.active_user_id,
                    exercise_name=self.exercise_name,
                    reps=summaries,
                    timestamp=time.time()
                )
                if new_profile:
                    self.profile_repo.save_profile(new_profile)
                    self.active_profile = new_profile

            self.active_session_id = None
            self.is_workout_active = False


def get_app_state() -> AppState:
    """Returns singleton state object cached in Streamlit session state."""
    if "app_state" not in st.session_state:
        st.session_state["app_state"] = AppState()
    return st.session_state["app_state"]

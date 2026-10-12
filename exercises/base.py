"""
AI Exercise Form & Posture Intelligence Platform
Base Exercise Abstract Class & Kinematic Lifecycle Contract
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from cv.landmarks import LandmarkIndex, PoseLandmarks
from features.extractor import KinematicFeatures


class MovementPhase(str, Enum):
    START = "start"
    TRANSITION_OUT = "transition_out"  # e.g., descending, lifting, pressing
    INFLECTION = "inflection"          # e.g., bottom, peak, lockout
    TRANSITION_BACK = "transition_back" # e.g., ascending, lowering, recovering
    COMPLETED = "completed"
    UNKNOWN = "unknown"


@dataclass
class FormIssue:
    code: str
    message: str
    severity: str  # "warning", "critical", "info"
    penalty: float # deduction from 100
    joint_target: Optional[LandmarkIndex] = None


@dataclass
class RepetitionSummary:
    rep_id: str
    exercise_name: str
    rep_number: int
    duration_seconds: float
    form_score: float
    score_breakdown: Dict[str, float]
    range_of_motion: float
    symmetry_score: float
    smoothness_score: float
    torso_stability_score: float
    issues_detected: List[str]
    is_correct: bool
    peak_angles: Dict[str, float]
    timestamp_start: float
    timestamp_end: float


class BaseExercise(ABC):
    """
    Abstract Base Class for all exercise definitions.
    Enforces unified phase detection, rule-based biomechanical validation,
    and repetition lifecycle management.
    """

    def __init__(self, name: str):
        self.name = name
        self.current_phase: MovementPhase = MovementPhase.START
        self.phase_display_name: str = "Starting"
        self.frame_in_phase: int = 0
        self.min_phase_frames: int = 3  # Hysteresis debounce to prevent false transitions

        # Repetition tracking state
        self.rep_start_time: float = 0.0
        self.current_rep_angles: List[Dict[str, float]] = []
        self.current_rep_geometries: List[Dict[str, float]] = []
        self.current_rep_smoothness: List[float] = []
        self.current_rep_issues: Set[str] = set()
        self.penalty_accumulated: float = 0.0

    @abstractmethod
    def process_frame(
        self,
        features: KinematicFeatures,
        pose: PoseLandmarks
    ) -> Tuple[MovementPhase, str, List[FormIssue]]:
        """
        Evaluates the current frame kinematics.
        Returns (current_phase, phase_display_label, active_frame_issues).
        """
        pass

    @abstractmethod
    def evaluate_completed_rep(
        self,
        rep_number: int,
        duration: float,
        features_history: List[KinematicFeatures]
    ) -> RepetitionSummary:
        """
        Evaluates full kinematic trajectory of a completed repetition.
        """
        pass

    @abstractmethod
    def get_primary_joints_to_highlight(self) -> List[LandmarkIndex]:
        """Returns landmark indices to visually emphasize on HUD skeleton."""
        pass

    def reset_rep_buffers(self, timestamp: float):
        self.rep_start_time = timestamp
        self.current_rep_angles.clear()
        self.current_rep_geometries.clear()
        self.current_rep_smoothness.clear()
        self.current_rep_issues.clear()
        self.penalty_accumulated = 0.0
        self.current_phase = MovementPhase.START
        self.frame_in_phase = 0

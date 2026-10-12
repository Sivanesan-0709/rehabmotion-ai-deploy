"""
AI Exercise Form & Posture Intelligence Platform
Repetition Counter & Kinematic State Transition Controller
"""

from typing import List, Optional, Tuple
import time

from cv.landmarks import PoseLandmarks
from exercises.base import BaseExercise, FormIssue, MovementPhase, RepetitionSummary
from features.extractor import KinematicFeatures


class RepCounter:
    """
    Manages repetition state transitions, debounces frame glitches,
    and aggregates completed repetition summaries.
    """

    def __init__(self, exercise: BaseExercise, min_rep_duration_s: float = 0.5, min_rep_frames: int = 4):
        self.exercise = exercise
        self.min_rep_duration_s = min_rep_duration_s
        self.min_rep_frames = min_rep_frames

        self.total_reps: int = 0
        self.correct_reps: int = 0
        self.incorrect_reps: int = 0

        self.rep_history: List[RepetitionSummary] = []
        self.current_rep_features: List[KinematicFeatures] = []
        self.rep_start_time: float = time.time()
        self.current_score: float = 100.0

    def process_frame(
        self,
        features: KinematicFeatures,
        pose: PoseLandmarks
    ) -> Tuple[MovementPhase, str, float, Optional[RepetitionSummary], List[FormIssue]]:
        """
        Processes a single frame and returns:
        (phase, phase_label, current_realtime_score, completed_rep_summary_if_any, active_issues)
        """
        if not features.is_valid:
            return (
                self.exercise.current_phase,
                self.exercise.phase_display_name,
                self.current_score,
                None,
                []
            )

        if not self.current_rep_features:
            self.rep_start_time = features.timestamp

        self.current_rep_features.append(features)

        # Run exercise-specific frame evaluation
        phase, phase_label, active_issues = self.exercise.process_frame(features, pose)

        # Compute real-time instant score based on current frame active issues
        frame_penalty = sum(issue.penalty for issue in active_issues)
        self.current_score = max(20.0, 100.0 - frame_penalty)

        completed_rep: Optional[RepetitionSummary] = None

        if phase == MovementPhase.COMPLETED:
            curr_time = features.timestamp
            rep_duration = curr_time - self.rep_start_time

            # Filter out spurious rapid glitches
            if rep_duration >= self.min_rep_duration_s and len(self.current_rep_features) >= self.min_rep_frames:
                self.total_reps += 1
                summary = self.exercise.evaluate_completed_rep(
                    rep_number=self.total_reps,
                    duration=rep_duration,
                    features_history=self.current_rep_features
                )

                if summary.is_correct:
                    self.correct_reps += 1
                else:
                    self.incorrect_reps += 1

                self.rep_history.append(summary)
                completed_rep = summary
                self.current_score = summary.form_score

            # Reset buffers for next repetition
            self.current_rep_features.clear()
            self.rep_start_time = features.timestamp
            self.exercise.reset_rep_buffers(features.timestamp)

        return phase, phase_label, self.current_score, completed_rep, active_issues

    def get_rep_scores_timeline(self) -> List[float]:
        return [rep.form_score for rep in self.rep_history]

    def reset(self):
        self.total_reps = 0
        self.correct_reps = 0
        self.incorrect_reps = 0
        self.rep_history.clear()
        self.current_rep_features.clear()
        self.rep_start_time = time.time()
        self.current_score = 100.0
        self.exercise.reset_rep_buffers(self.rep_start_time)

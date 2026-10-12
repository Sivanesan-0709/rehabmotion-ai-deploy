"""
AI Exercise Form & Posture Intelligence Platform
Push-up Biomechanical State Machine & Form Analyzer
"""

import uuid
from typing import Dict, List, Tuple
import numpy as np

from cv.landmarks import LandmarkIndex, PoseLandmarks
from exercises.base import BaseExercise, FormIssue, MovementPhase, RepetitionSummary
from features.extractor import KinematicFeatures


class PushUpExercise(BaseExercise):
    """
    Push-up biomechanical analysis engine.
    States: PLANK -> DESCENDING -> BOTTOM -> ASCENDING -> PLANK
    Evaluates: Chest Depth (Elbow angle <= 90 deg), Spinal/Hip Neutrality (No hip sag or hip elevation pike).
    """

    def __init__(
        self,
        plank_threshold: float = 150.0,
        bottom_threshold: float = 90.0,
        max_hip_deviation: float = 20.0
    ):
        super().__init__(name="Push-up")
        self.plank_thresh = plank_threshold
        self.bottom_thresh = bottom_threshold
        self.max_hip_deviation = max_hip_deviation

    def process_frame(
        self,
        features: KinematicFeatures,
        pose: PoseLandmarks
    ) -> Tuple[MovementPhase, str, List[FormIssue]]:
        if not features.is_valid:
            return self.current_phase, self.phase_display_name, []

        elbow_angle = features.angles.get("mean_elbow", 160.0)
        hip_angle = features.angles.get("mean_hip", 175.0)

        self.current_rep_angles.append(features.angles)
        self.current_rep_geometries.append(features.geometry)
        self.current_rep_smoothness.append(features.smoothness_score)

        active_issues: List[FormIssue] = []

        # 1. Real-time form checks: Hip Sag vs Hip Pike
        if hip_angle < 155.0:
            issue = FormIssue(
                code="HIP_PIKE",
                message="Hips elevated too high - maintain straight plank line",
                severity="warning",
                penalty=15.0,
                joint_target=LandmarkIndex.LEFT_HIP
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)
        elif hip_angle > 195.0 or (features.geometry.get("hip_tilt", 0.0) > 25.0):
            issue = FormIssue(
                code="HIP_SAG",
                message="Hips sagging towards floor - engage core/glutes",
                severity="critical",
                penalty=20.0,
                joint_target=LandmarkIndex.LEFT_HIP
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        # 2. Phase State Machine
        self.frame_in_phase += 1

        if self.current_phase == MovementPhase.START:
            self.phase_display_name = "Plank / Top"
            if elbow_angle < self.plank_thresh - 10.0:
                self.current_phase = MovementPhase.TRANSITION_OUT
                self.phase_display_name = "Descending"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_OUT:
            self.phase_display_name = "Descending"
            if elbow_angle <= self.bottom_thresh:
                self.current_phase = MovementPhase.INFLECTION
                self.phase_display_name = "Bottom / Chest Depth"
                self.frame_in_phase = 0
            elif elbow_angle >= self.plank_thresh - 5.0 and self.frame_in_phase > self.min_phase_frames:
                self.current_phase = MovementPhase.START
                self.phase_display_name = "Plank / Top"

        elif self.current_phase == MovementPhase.INFLECTION:
            self.phase_display_name = "Bottom / Chest Depth"
            if elbow_angle > self.bottom_thresh + 10.0:
                self.current_phase = MovementPhase.TRANSITION_BACK
                self.phase_display_name = "Ascending / Pressing"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_BACK:
            self.phase_display_name = "Ascending / Pressing"
            if elbow_angle >= self.plank_thresh - 6.0 and self.frame_in_phase >= self.min_phase_frames:
                self.current_phase = MovementPhase.COMPLETED
                self.phase_display_name = "Completed"
                self.frame_in_phase = 0

        return self.current_phase, self.phase_display_name, active_issues

    def evaluate_completed_rep(
        self,
        rep_number: int,
        duration: float,
        features_history: List[KinematicFeatures]
    ) -> RepetitionSummary:
        min_elbow_angle = 180.0
        hip_deviations = []
        symmetries = []
        smoothness_vals = []

        for f in features_history:
            if f.is_valid:
                e = f.angles.get("mean_elbow", 160.0)
                if e < min_elbow_angle:
                    min_elbow_angle = e
                hip = f.angles.get("mean_hip", 175.0)
                hip_deviations.append(abs(180.0 - hip))
                symmetries.append(f.geometry.get("elbow_symmetry", 100.0))
                smoothness_vals.append(f.smoothness_score)

        if min_elbow_angle <= 90.0:
            rom_score = 100.0
        else:
            rom_score = float(max(20.0, 100.0 - (min_elbow_angle - 90.0) * 3.0))
            self.current_rep_issues.add(f"Insufficient push-up depth (Min elbow: {int(min_elbow_angle)} deg)")

        mean_hip_dev = float(np.mean(hip_deviations)) if hip_deviations else 5.0
        stability_score = float(np.clip(100.0 - mean_hip_dev * 2.5, 30.0, 100.0))
        mean_symmetry = float(np.mean(symmetries)) if symmetries else 90.0
        mean_smoothness = float(np.mean(smoothness_vals)) if smoothness_vals else 85.0
        joint_alignment_score = float(np.clip(100.0 - len(self.current_rep_issues) * 15.0, 30.0, 100.0))

        composite_score = (
            0.25 * joint_alignment_score
            + 0.25 * rom_score
            + 0.20 * stability_score
            + 0.15 * mean_smoothness
            + 0.15 * mean_symmetry
        )
        composite_score = float(np.clip(composite_score, 10.0, 100.0))
        is_correct = composite_score >= 70.0 and rom_score >= 65.0

        return RepetitionSummary(
            rep_id=str(uuid.uuid4())[:8],
            exercise_name="Push-up",
            rep_number=rep_number,
            duration_seconds=duration,
            form_score=round(composite_score, 1),
            score_breakdown={
                "joint_alignment": round(joint_alignment_score, 1),
                "range_of_motion": round(rom_score, 1),
                "posture_stability": round(stability_score, 1),
                "movement_smoothness": round(mean_smoothness, 1),
                "symmetry": round(mean_symmetry, 1),
            },
            range_of_motion=round(rom_score, 1),
            symmetry_score=round(mean_symmetry, 1),
            smoothness_score=round(mean_smoothness, 1),
            torso_stability_score=round(stability_score, 1),
            issues_detected=list(self.current_rep_issues),
            is_correct=is_correct,
            peak_angles={"min_elbow_angle": round(min_elbow_angle, 1), "mean_hip_dev": round(mean_hip_dev, 1)},
            timestamp_start=self.rep_start_time,
            timestamp_end=self.rep_start_time + duration
        )

    def get_primary_joints_to_highlight(self) -> List[LandmarkIndex]:
        return [
            LandmarkIndex.LEFT_SHOULDER,
            LandmarkIndex.RIGHT_SHOULDER,
            LandmarkIndex.LEFT_ELBOW,
            LandmarkIndex.RIGHT_ELBOW,
            LandmarkIndex.LEFT_HIP,
            LandmarkIndex.RIGHT_HIP,
            LandmarkIndex.LEFT_ANKLE,
            LandmarkIndex.RIGHT_ANKLE,
        ]

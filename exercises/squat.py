"""
AI Exercise Form & Posture Intelligence Platform
Squat Biomechanical State Machine & Form Analyzer
"""

import uuid
from typing import Dict, List, Tuple
import numpy as np

from cv.landmarks import LandmarkIndex, PoseLandmarks
from exercises.base import BaseExercise, FormIssue, MovementPhase, RepetitionSummary
from features.extractor import KinematicFeatures


class SquatExercise(BaseExercise):
    """
    Squat biomechanical analysis engine.
    States: STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING
    Evaluates: Depth, Torso Lean, Knee Valgus, Left/Right Symmetry.
    """

    def __init__(
        self,
        standing_threshold: float = 155.0,
        bottom_threshold: float = 95.0,
        max_torso_lean: float = 42.0,
        max_knee_valgus: float = 14.0
    ):
        super().__init__(name="Squat")
        self.standing_thresh = standing_threshold
        self.bottom_thresh = bottom_threshold
        self.max_torso_lean = max_torso_lean
        self.max_knee_valgus = max_knee_valgus

    def process_frame(
        self,
        features: KinematicFeatures,
        pose: PoseLandmarks
    ) -> Tuple[MovementPhase, str, List[FormIssue]]:
        if not features.is_valid:
            return self.current_phase, self.phase_display_name, []

        knee_angle = features.angles.get("mean_knee", 170.0)
        torso_lean = features.angles.get("torso_inclination", 0.0)
        l_valgus = features.geometry.get("left_knee_valgus", 0.0)
        r_valgus = features.geometry.get("right_knee_valgus", 0.0)
        symmetry = features.geometry.get("knee_symmetry", 100.0)

        # Store for rep analysis
        self.current_rep_angles.append(features.angles)
        self.current_rep_geometries.append(features.geometry)
        self.current_rep_smoothness.append(features.smoothness_score)

        active_issues: List[FormIssue] = []

        # 1. Real-time form checks
        if torso_lean > self.max_torso_lean:
            issue = FormIssue(
                code="EXCESSIVE_TORSO_LEAN",
                message=f"Keep chest upright (Torso lean: {int(torso_lean)} deg)",
                severity="warning",
                penalty=15.0,
                joint_target=LandmarkIndex.LEFT_SHOULDER
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        if max(l_valgus, r_valgus) > self.max_knee_valgus:
            issue = FormIssue(
                code="KNEE_VALGUS",
                message="Knees caving inward - push knees outward over toes",
                severity="warning",
                penalty=18.0,
                joint_target=LandmarkIndex.LEFT_KNEE
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        if symmetry < 75.0:
            issue = FormIssue(
                code="ASYMMETRIC_SQUAT",
                message="Uneven weight distribution between legs",
                severity="info",
                penalty=10.0
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        # 2. Phase State Machine
        self.frame_in_phase += 1

        if self.current_phase == MovementPhase.START:
            self.phase_display_name = "Standing"
            if knee_angle < self.standing_thresh - 8.0:
                self.current_phase = MovementPhase.TRANSITION_OUT
                self.phase_display_name = "Descending"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_OUT:
            self.phase_display_name = "Descending"
            if knee_angle <= self.bottom_thresh:
                self.current_phase = MovementPhase.INFLECTION
                self.phase_display_name = "Bottom / Hole"
                self.frame_in_phase = 0
            elif knee_angle >= self.standing_thresh - 5.0 and self.frame_in_phase > self.min_phase_frames:
                # User aborted squat without reaching bottom
                self.current_phase = MovementPhase.START
                self.phase_display_name = "Standing"

        elif self.current_phase == MovementPhase.INFLECTION:
            self.phase_display_name = "Bottom / Hole"
            if knee_angle > self.bottom_thresh + 8.0:
                self.current_phase = MovementPhase.TRANSITION_BACK
                self.phase_display_name = "Ascending"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_BACK:
            self.phase_display_name = "Ascending"
            if knee_angle >= self.standing_thresh - 5.0 and self.frame_in_phase >= self.min_phase_frames:
                self.current_phase = MovementPhase.COMPLETED
                self.phase_display_name = "Completed"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.COMPLETED:
            self.phase_display_name = "Completed"
            if knee_angle >= self.standing_thresh - 8.0 and self.frame_in_phase >= 1:
                self.current_phase = MovementPhase.START
                self.phase_display_name = "Standing"
                self.frame_in_phase = 0

        return self.current_phase, self.phase_display_name, active_issues

    def evaluate_completed_rep(
        self,
        rep_number: int,
        duration: float,
        features_history: List[KinematicFeatures]
    ) -> RepetitionSummary:
        min_knee_angle = 180.0
        max_torso_lean = 0.0
        symmetries = []
        smoothness_vals = []

        for f in features_history:
            if f.is_valid:
                k = f.angles.get("mean_knee", 180.0)
                if k < min_knee_angle:
                    min_knee_angle = k
                t = f.angles.get("torso_inclination", 0.0)
                if t > max_torso_lean:
                    max_torso_lean = t
                symmetries.append(f.geometry.get("knee_symmetry", 100.0))
                smoothness_vals.append(f.smoothness_score)

        # Compute Range of Motion (Ideal bottom <= 90 deg)
        # ROM score: 100 if reached <= 90 deg, scaled down if partial depth
        if min_knee_angle <= 90.0:
            rom_score = 100.0
        elif min_knee_angle <= 105.0:
            rom_score = float(100.0 - (min_knee_angle - 90.0) * 2.0)
        else:
            rom_score = float(max(20.0, 100.0 - (min_knee_angle - 90.0) * 3.5))
            self.current_rep_issues.add(f"Insufficient squat depth (Min knee: {int(min_knee_angle)} deg)")

        # Alignment score based on torso stability and knee valgus
        torso_stability_score = float(np.clip(100.0 - max(0.0, max_torso_lean - 30.0) * 2.5, 30.0, 100.0))
        mean_symmetry = float(np.mean(symmetries)) if symmetries else 90.0
        mean_smoothness = float(np.mean(smoothness_vals)) if smoothness_vals else 85.0
        joint_alignment_score = float(np.clip(100.0 - len(self.current_rep_issues) * 12.0, 30.0, 100.0))

        # Weighted composite score
        # Alignment 25%, ROM 25%, Posture Stability 20%, Smoothness 15%, Symmetry 15%
        composite_score = (
            0.25 * joint_alignment_score
            + 0.25 * rom_score
            + 0.20 * torso_stability_score
            + 0.15 * mean_smoothness
            + 0.15 * mean_symmetry
        )
        composite_score = float(np.clip(composite_score, 10.0, 100.0))

        is_correct = composite_score >= 70.0 and rom_score >= 65.0

        return RepetitionSummary(
            rep_id=str(uuid.uuid4())[:8],
            exercise_name="Squat",
            rep_number=rep_number,
            duration_seconds=duration,
            form_score=round(composite_score, 1),
            score_breakdown={
                "joint_alignment": round(joint_alignment_score, 1),
                "range_of_motion": round(rom_score, 1),
                "posture_stability": round(torso_stability_score, 1),
                "movement_smoothness": round(mean_smoothness, 1),
                "symmetry": round(mean_symmetry, 1),
            },
            range_of_motion=round(rom_score, 1),
            symmetry_score=round(mean_symmetry, 1),
            smoothness_score=round(mean_smoothness, 1),
            torso_stability_score=round(torso_stability_score, 1),
            issues_detected=list(self.current_rep_issues),
            is_correct=is_correct,
            peak_angles={"min_knee_angle": round(min_knee_angle, 1), "max_torso_lean": round(max_torso_lean, 1)},
            timestamp_start=self.rep_start_time,
            timestamp_end=self.rep_start_time + duration
        )

    def get_primary_joints_to_highlight(self) -> List[LandmarkIndex]:
        return [
            LandmarkIndex.LEFT_KNEE,
            LandmarkIndex.RIGHT_KNEE,
            LandmarkIndex.LEFT_HIP,
            LandmarkIndex.RIGHT_HIP,
            LandmarkIndex.LEFT_ANKLE,
            LandmarkIndex.RIGHT_ANKLE,
        ]

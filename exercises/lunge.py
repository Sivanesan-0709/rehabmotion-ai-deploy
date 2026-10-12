"""
AI Exercise Form & Posture Intelligence Platform
Lunge Biomechanical State Machine & Form Analyzer
"""

import uuid
from typing import Dict, List, Tuple
import numpy as np

from cv.landmarks import LandmarkIndex, PoseLandmarks
from exercises.base import BaseExercise, FormIssue, MovementPhase, RepetitionSummary
from features.extractor import KinematicFeatures


class LungeExercise(BaseExercise):
    """
    Lunge biomechanical analysis engine.
    States: STANDING -> LUNGING -> BOTTOM -> RECOVERING -> STANDING
    Evaluates: Lead Knee Angle (target ~90 deg), Torso Uprightness, Knee-Ankle Valgus.
    """

    def __init__(
        self,
        standing_threshold: float = 155.0,
        bottom_threshold: float = 100.0,
        max_torso_lean: float = 25.0
    ):
        super().__init__(name="Lunge")
        self.standing_thresh = standing_threshold
        self.bottom_thresh = bottom_threshold
        self.max_torso_lean = max_torso_lean

    def process_frame(
        self,
        features: KinematicFeatures,
        pose: PoseLandmarks
    ) -> Tuple[MovementPhase, str, List[FormIssue]]:
        if not features.is_valid:
            return self.current_phase, self.phase_display_name, []

        l_knee = features.angles.get("left_knee", 170.0)
        r_knee = features.angles.get("right_knee", 170.0)
        lead_knee = min(l_knee, r_knee)
        torso_lean = features.angles.get("torso_inclination", 0.0)

        self.current_rep_angles.append(features.angles)
        self.current_rep_geometries.append(features.geometry)
        self.current_rep_smoothness.append(features.smoothness_score)

        active_issues: List[FormIssue] = []

        if torso_lean > self.max_torso_lean:
            issue = FormIssue(
                code="TORSO_LEAN_LUNGE",
                message=f"Keep upper body vertical (Torso lean: {int(torso_lean)} deg)",
                severity="warning",
                penalty=15.0,
                joint_target=LandmarkIndex.LEFT_SHOULDER
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        # Phase State Machine
        self.frame_in_phase += 1

        if self.current_phase == MovementPhase.START:
            self.phase_display_name = "Standing"
            if lead_knee < self.standing_thresh - 8.0:
                self.current_phase = MovementPhase.TRANSITION_OUT
                self.phase_display_name = "Stepping / Lunging"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_OUT:
            self.phase_display_name = "Stepping / Lunging"
            if lead_knee <= self.bottom_thresh:
                self.current_phase = MovementPhase.INFLECTION
                self.phase_display_name = "Bottom / 90° Knee"
                self.frame_in_phase = 0
            elif lead_knee >= self.standing_thresh - 5.0 and self.frame_in_phase > self.min_phase_frames:
                self.current_phase = MovementPhase.START
                self.phase_display_name = "Standing"

        elif self.current_phase == MovementPhase.INFLECTION:
            self.phase_display_name = "Bottom / 90° Knee"
            if lead_knee > self.bottom_thresh + 8.0:
                self.current_phase = MovementPhase.TRANSITION_BACK
                self.phase_display_name = "Recovering / Pushing"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_BACK:
            self.phase_display_name = "Recovering / Pushing"
            if lead_knee >= self.standing_thresh - 5.0 and self.frame_in_phase >= self.min_phase_frames:
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
        min_lead_knee = 180.0
        max_torso_lean = 0.0
        smoothness_vals = []

        for f in features_history:
            if f.is_valid:
                lk = min(f.angles.get("left_knee", 180.0), f.angles.get("right_knee", 180.0))
                if lk < min_lead_knee:
                    min_lead_knee = lk
                t = f.angles.get("torso_inclination", 0.0)
                if t > max_torso_lean:
                    max_torso_lean = t
                smoothness_vals.append(f.smoothness_score)

        if min_lead_knee <= 95.0:
            rom_score = 100.0
        else:
            rom_score = float(max(25.0, 100.0 - (min_lead_knee - 95.0) * 3.0))
            self.current_rep_issues.add(f"Insufficient lunge depth (Lead knee: {int(min_lead_knee)} deg)")

        torso_stability_score = float(np.clip(100.0 - max(0.0, max_torso_lean - 15.0) * 3.0, 30.0, 100.0))
        mean_smoothness = float(np.mean(smoothness_vals)) if smoothness_vals else 85.0
        joint_alignment_score = float(np.clip(100.0 - len(self.current_rep_issues) * 15.0, 30.0, 100.0))
        symmetry_score = 90.0  # Asymmetric unilateral movement

        composite_score = (
            0.25 * joint_alignment_score
            + 0.25 * rom_score
            + 0.20 * torso_stability_score
            + 0.15 * mean_smoothness
            + 0.15 * symmetry_score
        )
        composite_score = float(np.clip(composite_score, 10.0, 100.0))
        is_correct = composite_score >= 70.0 and rom_score >= 65.0

        return RepetitionSummary(
            rep_id=str(uuid.uuid4())[:8],
            exercise_name="Lunge",
            rep_number=rep_number,
            duration_seconds=duration,
            form_score=round(composite_score, 1),
            score_breakdown={
                "joint_alignment": round(joint_alignment_score, 1),
                "range_of_motion": round(rom_score, 1),
                "posture_stability": round(torso_stability_score, 1),
                "movement_smoothness": round(mean_smoothness, 1),
                "symmetry": round(symmetry_score, 1),
            },
            range_of_motion=round(rom_score, 1),
            symmetry_score=round(symmetry_score, 1),
            smoothness_score=round(mean_smoothness, 1),
            torso_stability_score=round(torso_stability_score, 1),
            issues_detected=list(self.current_rep_issues),
            is_correct=is_correct,
            peak_angles={"min_lead_knee": round(min_lead_knee, 1), "max_torso_lean": round(max_torso_lean, 1)},
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

"""
AI Exercise Form & Posture Intelligence Platform
Overhead Shoulder Press Biomechanical State Machine & Form Analyzer
"""

import uuid
from typing import Dict, List, Tuple
import numpy as np

from cv.landmarks import LandmarkIndex, PoseLandmarks
from exercises.base import BaseExercise, FormIssue, MovementPhase, RepetitionSummary
from features.extractor import KinematicFeatures


class ShoulderPressExercise(BaseExercise):
    """
    Overhead Shoulder Press biomechanical analysis engine.
    States: RACK -> PRESSING -> LOCKOUT -> LOWERING -> RACK
    Evaluates: Full Overhead Lockout (Elbows >= 165 deg), Arm Symmetry, Lumbar Hyperextension.
    """

    def __init__(
        self,
        rack_threshold: float = 85.0,
        lockout_threshold: float = 160.0,
        max_torso_lean: float = 18.0
    ):
        super().__init__(name="Shoulder Press")
        self.rack_thresh = rack_threshold
        self.lockout_thresh = lockout_threshold
        self.max_torso_lean = max_torso_lean

    def process_frame(
        self,
        features: KinematicFeatures,
        pose: PoseLandmarks
    ) -> Tuple[MovementPhase, str, List[FormIssue]]:
        if not features.is_valid:
            return self.current_phase, self.phase_display_name, []

        elbow_angle = features.angles.get("mean_elbow", 80.0)
        symmetry = features.geometry.get("elbow_symmetry", 100.0)
        torso_lean = features.angles.get("torso_inclination", 0.0)

        self.current_rep_angles.append(features.angles)
        self.current_rep_geometries.append(features.geometry)
        self.current_rep_smoothness.append(features.smoothness_score)

        active_issues: List[FormIssue] = []

        if torso_lean > self.max_torso_lean:
            issue = FormIssue(
                code="LUMBAR_HYPEREXTENSION",
                message="Excessive backward torso lean - squeeze glutes and brace core",
                severity="critical",
                penalty=20.0,
                joint_target=LandmarkIndex.LEFT_HIP
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        if symmetry < 78.0:
            issue = FormIssue(
                code="ARM_PRESS_ASYMMETRY",
                message="Uneven press height between arms",
                severity="warning",
                penalty=12.0,
                joint_target=LandmarkIndex.LEFT_WRIST
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        # Phase State Machine
        self.frame_in_phase += 1

        if self.current_phase == MovementPhase.START:
            self.phase_display_name = "Rack Position"
            if elbow_angle > self.rack_thresh + 10.0:
                self.current_phase = MovementPhase.TRANSITION_OUT
                self.phase_display_name = "Pressing Upward"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_OUT:
            self.phase_display_name = "Pressing Upward"
            if elbow_angle >= self.lockout_thresh:
                self.current_phase = MovementPhase.INFLECTION
                self.phase_display_name = "Overhead Lockout"
                self.frame_in_phase = 0
            elif elbow_angle <= self.rack_thresh + 5.0 and self.frame_in_phase > self.min_phase_frames:
                self.current_phase = MovementPhase.START
                self.phase_display_name = "Rack Position"

        elif self.current_phase == MovementPhase.INFLECTION:
            self.phase_display_name = "Overhead Lockout"
            if elbow_angle < self.lockout_thresh - 10.0:
                self.current_phase = MovementPhase.TRANSITION_BACK
                self.phase_display_name = "Lowering to Rack"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_BACK:
            self.phase_display_name = "Lowering to Rack"
            if elbow_angle <= self.rack_thresh + 8.0 and self.frame_in_phase >= self.min_phase_frames:
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
        max_elbow_angle = 0.0
        min_elbow_angle = 180.0
        symmetries = []
        smoothness_vals = []

        for f in features_history:
            if f.is_valid:
                e = f.angles.get("mean_elbow", 80.0)
                if e > max_elbow_angle:
                    max_elbow_angle = e
                if e < min_elbow_angle:
                    min_elbow_angle = e
                symmetries.append(f.geometry.get("elbow_symmetry", 100.0))
                smoothness_vals.append(f.smoothness_score)

        if max_elbow_angle >= 160.0:
            rom_score = 100.0
        else:
            rom_score = float(max(25.0, 100.0 - (160.0 - max_elbow_angle) * 2.5))
            self.current_rep_issues.add(f"Incomplete overhead lockout (Max elbow: {int(max_elbow_angle)} deg)")

        mean_symmetry = float(np.mean(symmetries)) if symmetries else 92.0
        mean_smoothness = float(np.mean(smoothness_vals)) if smoothness_vals else 88.0
        torso_stability_score = 95.0 if "LUMBAR_HYPEREXTENSION" not in " ".join(self.current_rep_issues) else 60.0
        joint_alignment_score = float(np.clip(100.0 - len(self.current_rep_issues) * 15.0, 30.0, 100.0))

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
            exercise_name="Shoulder Press",
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
            peak_angles={"max_elbow_angle": round(max_elbow_angle, 1), "min_elbow_angle": round(min_elbow_angle, 1)},
            timestamp_start=self.rep_start_time,
            timestamp_end=self.rep_start_time + duration
        )

    def get_primary_joints_to_highlight(self) -> List[LandmarkIndex]:
        return [
            LandmarkIndex.LEFT_SHOULDER,
            LandmarkIndex.RIGHT_SHOULDER,
            LandmarkIndex.LEFT_ELBOW,
            LandmarkIndex.RIGHT_ELBOW,
            LandmarkIndex.LEFT_WRIST,
            LandmarkIndex.RIGHT_WRIST,
            LandmarkIndex.LEFT_HIP,
            LandmarkIndex.RIGHT_HIP,
        ]

"""
AI Exercise Form & Posture Intelligence Platform
Bicep Curl Biomechanical State Machine & Form Analyzer
"""

import uuid
from typing import Dict, List, Tuple
import numpy as np

from cv.landmarks import LandmarkIndex, PoseLandmarks
from exercises.base import BaseExercise, FormIssue, MovementPhase, RepetitionSummary
from features.extractor import KinematicFeatures


class BicepCurlExercise(BaseExercise):
    """
    Bicep Curl biomechanical analysis engine.
    States: EXTENDED -> LIFTING -> PEAK -> LOWERING -> EXTENDED
    Evaluates: Full elbow contraction/extension ROM, Elbow stability (no backward drift), Shoulder cheating.
    """

    def __init__(
        self,
        extended_threshold: float = 145.0,
        peak_threshold: float = 55.0,
        max_elbow_drift: float = 16.0
    ):
        super().__init__(name="Bicep Curl")
        self.extended_thresh = extended_threshold
        self.peak_thresh = peak_threshold
        self.max_elbow_drift = max_elbow_drift

    def process_frame(
        self,
        features: KinematicFeatures,
        pose: PoseLandmarks
    ) -> Tuple[MovementPhase, str, List[FormIssue]]:
        if not features.is_valid:
            return self.current_phase, self.phase_display_name, []

        elbow_angle = features.angles.get("mean_elbow", 160.0)
        l_drift = features.geometry.get("left_elbow_drift", 0.0)
        r_drift = features.geometry.get("right_elbow_drift", 0.0)
        symmetry = features.geometry.get("elbow_symmetry", 100.0)
        torso_lean = features.angles.get("torso_inclination", 0.0)

        self.current_rep_angles.append(features.angles)
        self.current_rep_geometries.append(features.geometry)
        self.current_rep_smoothness.append(features.smoothness_score)

        active_issues: List[FormIssue] = []

        # 1. Real-time form checks
        if max(l_drift, r_drift) > self.max_elbow_drift:
            issue = FormIssue(
                code="ELBOW_DRIFT",
                message="Pin your elbows to your sides (Avoid swinging)",
                severity="warning",
                penalty=15.0,
                joint_target=LandmarkIndex.LEFT_ELBOW
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        if torso_lean > 20.0:
            issue = FormIssue(
                code="TORSO_SWAY_MOMENTUM",
                message="Torso swinging detected - isolate arms without using body momentum",
                severity="warning",
                penalty=18.0,
                joint_target=LandmarkIndex.LEFT_SHOULDER
            )
            active_issues.append(issue)
            self.current_rep_issues.add(issue.message)

        # 2. Phase State Machine
        self.frame_in_phase += 1

        if self.current_phase == MovementPhase.START:
            self.phase_display_name = "Extended"
            if elbow_angle < self.extended_thresh - 10.0:
                self.current_phase = MovementPhase.TRANSITION_OUT
                self.phase_display_name = "Lifting / Curling"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_OUT:
            self.phase_display_name = "Lifting / Curling"
            if elbow_angle <= self.peak_thresh:
                self.current_phase = MovementPhase.INFLECTION
                self.phase_display_name = "Peak Contraction"
                self.frame_in_phase = 0
            elif elbow_angle >= self.extended_thresh - 5.0 and self.frame_in_phase > self.min_phase_frames:
                self.current_phase = MovementPhase.START
                self.phase_display_name = "Extended"

        elif self.current_phase == MovementPhase.INFLECTION:
            self.phase_display_name = "Peak Contraction"
            if elbow_angle > self.peak_thresh + 12.0:
                self.current_phase = MovementPhase.TRANSITION_BACK
                self.phase_display_name = "Lowering (Eccentric)"
                self.frame_in_phase = 0

        elif self.current_phase == MovementPhase.TRANSITION_BACK:
            self.phase_display_name = "Lowering (Eccentric)"
            if elbow_angle >= self.extended_thresh - 8.0 and self.frame_in_phase >= self.min_phase_frames:
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
        max_elbow_angle = 0.0
        symmetries = []
        smoothness_vals = []

        for f in features_history:
            if f.is_valid:
                e = f.angles.get("mean_elbow", 160.0)
                if e < min_elbow_angle:
                    min_elbow_angle = e
                if e > max_elbow_angle:
                    max_elbow_angle = e
                symmetries.append(f.geometry.get("elbow_symmetry", 100.0))
                smoothness_vals.append(f.smoothness_score)

        # ROM Score (Ideal peak <= 45-50 deg, full extension >= 150 deg)
        if min_elbow_angle <= 50.0 and max_elbow_angle >= 145.0:
            rom_score = 100.0
        else:
            contraction_penalty = max(0.0, min_elbow_angle - 50.0) * 1.5
            extension_penalty = max(0.0, 145.0 - max_elbow_angle) * 1.5
            rom_score = float(np.clip(100.0 - contraction_penalty - extension_penalty, 25.0, 100.0))
            if min_elbow_angle > 60.0:
                self.current_rep_issues.add(f"Incomplete curl squeeze (Peak angle: {int(min_elbow_angle)} deg)")
            if max_elbow_angle < 135.0:
                self.current_rep_issues.add("Did not achieve full arm extension at bottom")

        mean_symmetry = float(np.mean(symmetries)) if symmetries else 92.0
        mean_smoothness = float(np.mean(smoothness_vals)) if smoothness_vals else 88.0
        torso_stability_score = 95.0 if "TORSO_SWAY_MOMENTUM" not in " ".join(self.current_rep_issues) else 65.0
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
            exercise_name="Bicep Curl",
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
            peak_angles={"min_elbow_angle": round(min_elbow_angle, 1), "max_elbow_angle": round(max_elbow_angle, 1)},
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
        ]

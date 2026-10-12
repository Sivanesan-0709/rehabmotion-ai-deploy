"""
AI Exercise Form & Posture Intelligence Platform
Explainable Weighted Scoring Engine & Consistency Trend Analyzer
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.config import ScoringWeights


@dataclass
class ScoreExplanation:
    overall_score: float
    status: str
    breakdown: Dict[str, float]
    positive_feedback: List[str]
    improvement_feedback: List[str]
    is_consistency_alert: bool = False
    alert_message: Optional[str] = None


class FormScoringEngine:
    """
    Computes explainable, transparent form scores based on biomechanical weights
    and detects fatigue-related movement degradation trends.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None):
        self.weights = weights or ScoringWeights()

    def evaluate_score(
        self,
        joint_alignment: float,
        range_of_motion: float,
        posture_stability: float,
        movement_smoothness: float,
        symmetry: float,
        issues: List[str],
        recent_scores: Optional[List[float]] = None
    ) -> ScoreExplanation:
        """
        Calculates weighted overall score and generates human-readable explanations.
        """
        # Clamp components
        c_align = float(np.clip(joint_alignment, 0.0, 100.0))
        c_rom = float(np.clip(range_of_motion, 0.0, 100.0))
        c_stab = float(np.clip(posture_stability, 0.0, 100.0))
        c_smooth = float(np.clip(movement_smoothness, 0.0, 100.0))
        c_sym = float(np.clip(symmetry, 0.0, 100.0))

        overall = (
            self.weights.joint_alignment * c_align
            + self.weights.range_of_motion * c_rom
            + self.weights.posture_stability * c_stab
            + self.weights.movement_smoothness * c_smooth
            + self.weights.symmetry * c_sym
        )
        overall = float(np.clip(overall, 10.0, 100.0))

        # Status categorization
        if overall >= 88.0:
            status = "Excellent Form"
        elif overall >= 72.0:
            status = "Good Form"
        elif overall >= 55.0:
            status = "Moderate Form - Attention Needed"
        else:
            status = "Poor Form - Form Correction Required"

        positives = []
        improvements = []

        if c_rom >= 90.0:
            positives.append("Excellent range of motion achieved.")
        elif c_rom < 70.0:
            improvements.append("Increase movement depth / range of motion.")

        if c_sym >= 90.0:
            positives.append("High left/right bilateral symmetry.")
        elif c_sym < 75.0:
            improvements.append("Uneven side-to-side execution detected.")

        if c_stab >= 85.0:
            positives.append("Solid core and torso stability.")
        elif c_stab < 70.0:
            improvements.append("Maintain rigid torso and minimize spine sway.")

        if c_smooth >= 85.0:
            positives.append("Smooth, controlled movement cadence.")
        elif c_smooth < 70.0:
            improvements.append("Movement shows erratic jerk / acceleration.")

        for issue in issues:
            if issue not in improvements:
                improvements.append(issue)

        # Check for Movement Consistency Degradation
        is_alert, alert_msg = self.check_consistency_degradation(recent_scores)

        return ScoreExplanation(
            overall_score=round(overall, 1),
            status=status,
            breakdown={
                "Joint Alignment": round(c_align, 1),
                "Range of Motion": round(c_rom, 1),
                "Posture Stability": round(c_stab, 1),
                "Movement Smoothness": round(c_smooth, 1),
                "Bilateral Symmetry": round(c_sym, 1),
            },
            positive_feedback=positives,
            improvement_feedback=improvements,
            is_consistency_alert=is_alert,
            alert_message=alert_msg
        )

    def check_consistency_degradation(
        self,
        recent_scores: Optional[List[float]],
        window_size: int = 4,
        drop_threshold: float = 14.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Monitors sliding window of rep scores.
        Detects continuous degradation indicative of movement inconsistency / pacing issues.
        (Non-medical fitness recommendation).
        """
        if not recent_scores or len(recent_scores) < window_size:
            return False, None

        window = recent_scores[-window_size:]
        # Check monotonic or strong downward trend
        first_half_avg = np.mean(window[:window_size // 2])
        second_half_avg = np.mean(window[window_size // 2:])

        drop = first_half_avg - second_half_avg

        if drop >= drop_threshold:
            msg = (
                f"Movement consistency is decreasing (Score dropped by {drop:.1f}% over the last "
                f"{window_size} reps). Consider taking a short rest interval."
            )
            return True, msg

        return False, None

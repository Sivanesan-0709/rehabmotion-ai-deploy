"""
AI Exercise Form & Posture Intelligence Platform
Motion DNA & Form Fingerprint Baseline Engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from exercises.base import RepetitionSummary


@dataclass
class MetricDistribution:
    mean: float
    std: float
    median: float
    p25: float
    p75: float
    min_val: float
    max_val: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "median": round(self.median, 2),
            "p25": round(self.p25, 2),
            "p75": round(self.p75, 2),
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
        }


@dataclass
class MotionDNAProfile:
    user_id: str
    exercise_name: str
    total_baseline_reps: int
    created_at: float
    updated_at: float
    metrics: Dict[str, MetricDistribution]
    fingerprint_scores: Dict[str, float]  # Normalized 0-100 values for radar chart
    consistency_index: float              # 0-100% rep-to-rep consistency

    def get_summary_str(self) -> str:
        lines = [
            f"=== USER MOTION DNA: {self.exercise_name.upper()} ===",
            f"Baseline Reps: {self.total_baseline_reps} | Consistency Index: {self.consistency_index:.1f}%",
            "------------------------------------------------"
        ]
        for name, dist in self.metrics.items():
            lines.append(f"{name:<22}: {dist.mean:.1f} +/- {dist.std:.1f} (range {dist.min_val:.1f} - {dist.max_val:.1f})")
        return "\n".join(lines)


@dataclass
class RepetitionDeviation:
    rep_number: int
    overall_deviation_pct: float
    is_significant_deviation: bool
    deviation_breakdown: Dict[str, Dict[str, float]] # metric -> {current, baseline_mean, baseline_std, z_score}
    deviation_message: str


class MotionDNAEngine:
    """
    Constructs personalized movement baselines (Form Fingerprint) from a user's verified repetitions
    and computes statistical deviations for real-time rep comparison.
    """

    CORE_METRICS = [
        "range_of_motion",
        "duration_seconds",
        "symmetry_score",
        "smoothness_score",
        "torso_stability_score",
    ]

    def __init__(self, min_calibration_reps: int = 4):
        self.min_calibration_reps = min_calibration_reps

    def build_profile(
        self,
        user_id: str,
        exercise_name: str,
        reps: List[RepetitionSummary],
        timestamp: float
    ) -> Optional[MotionDNAProfile]:
        """
        Builds a statistical Motion DNA Profile from a list of completed reps.
        """
        valid_reps = [r for r in reps if r.form_score >= 60.0]
        if len(valid_reps) < self.min_calibration_reps:
            return None

        metrics_dist: Dict[str, MetricDistribution] = {}
        fingerprint: Dict[str, float] = {}
        cv_scores = []

        for metric in self.CORE_METRICS:
            values = np.array([getattr(r, metric) for r in valid_reps], dtype=np.float32)
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            median_val = float(np.median(values))
            p25 = float(np.percentile(values, 25))
            p75 = float(np.percentile(values, 75))
            min_v = float(np.min(values))
            max_v = float(np.max(values))

            # Coefficient of Variation for consistency calculation
            cv = (std_val / mean_val) if mean_val > 1e-4 else 0.0
            cv_scores.append(cv)

            metrics_dist[metric] = MetricDistribution(
                mean=mean_val,
                std=std_val,
                median=median_val,
                p25=p25,
                p75=p75,
                min_val=min_v,
                max_val=max_v
            )

            # Fingerprint normalized score for radar visualization
            if metric == "duration_seconds":
                # Normalize typical 1.5s - 4.0s duration to 100
                fingerprint["Cadence Stability"] = float(np.clip(100.0 - cv * 100.0, 30.0, 100.0))
            else:
                fingerprint[metric.replace("_", " ").title()] = float(np.clip(mean_val, 0.0, 100.0))

        # Overall rep consistency: 100 - mean(CV)*100
        mean_cv = float(np.mean(cv_scores)) if cv_scores else 0.1
        consistency_index = float(np.clip(100.0 * (1.0 - mean_cv), 20.0, 99.0))

        return MotionDNAProfile(
            user_id=user_id,
            exercise_name=exercise_name,
            total_baseline_reps=len(valid_reps),
            created_at=timestamp,
            updated_at=timestamp,
            metrics=metrics_dist,
            fingerprint_scores=fingerprint,
            consistency_index=round(consistency_index, 1)
        )

    def calculate_deviation(
        self,
        rep: RepetitionSummary,
        profile: MotionDNAProfile,
        z_threshold: float = 1.8
    ) -> RepetitionDeviation:
        """
        Compares a single repetition against the user's stored Motion DNA baseline.
        """
        breakdown = {}
        z_scores_sq = []
        messages = []

        for metric in self.CORE_METRICS:
            if metric not in profile.metrics:
                continue

            dist = profile.metrics[metric]
            curr_val = float(getattr(rep, metric))
            # Epsilon to avoid divide by zero when variance is near zero
            denom = max(dist.std, 0.05 * dist.mean, 1e-3)
            z_score = (curr_val - dist.mean) / denom

            breakdown[metric] = {
                "current": round(curr_val, 2),
                "baseline_mean": round(dist.mean, 2),
                "baseline_std": round(dist.std, 2),
                "z_score": round(float(z_score), 2)
            }
            z_scores_sq.append(z_score ** 2)

            if abs(z_score) > z_threshold:
                direction = "lower" if z_score < 0 else "higher"
                pct_diff = abs((curr_val - dist.mean) / dist.mean * 100.0) if dist.mean > 1e-4 else 0.0
                metric_clean = metric.replace("_score", "").replace("_", " ").title()
                messages.append(f"{metric_clean} was {pct_diff:.0f}% {direction} than your normal baseline.")

        # Standardized Mahalanobis-like composite distance
        composite_distance = float(np.sqrt(np.mean(z_scores_sq))) if z_scores_sq else 0.0
        # Map distance to percentage deviation [0 - 100]
        dev_pct = float(np.clip(composite_distance * 18.0, 0.0, 100.0))
        is_significant = bool(composite_distance >= z_threshold)

        if is_significant:
            msg = "Your movement differs significantly from your normal baseline pattern: " + " ".join(messages[:2])
        else:
            msg = "Repetition matches your personal baseline Motion DNA."

        return RepetitionDeviation(
            rep_number=rep.rep_number,
            overall_deviation_pct=round(dev_pct, 1),
            is_significant_deviation=is_significant,
            deviation_breakdown=breakdown,
            deviation_message=msg
        )

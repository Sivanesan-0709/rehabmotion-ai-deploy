"""
AI Exercise Form & Posture Intelligence Platform
Dataset Schema Validator & Kinematic Feature Integrity Engine
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Detailed dataset validation audit report."""
    is_valid: bool
    total_samples: int
    total_subjects: int
    total_sessions: int
    unique_exercises: List[str]
    exercise_counts: Dict[str, int]
    missing_value_count: int
    feature_dimension: int
    expected_feature_dimension: int
    has_monotonic_timestamps: bool
    issues: List[str]
    warnings: List[str]

    def summary_markdown(self) -> str:
        status_icon = "✅" if self.is_valid else "❌"
        lines = [
            f"### {status_icon} Dataset Schema Validation Report",
            f"- **Overall Status:** {'VALID' if self.is_valid else 'INVALID / DEFECTIVE'}",
            f"- **Total Frames / Samples:** {self.total_samples}",
            f"- **Subjects:** {self.total_subjects} | **Sessions:** {self.total_sessions}",
            f"- **Feature Dimension:** {self.feature_dimension} / {self.expected_feature_dimension}",
            f"- **Missing / NaN Values:** {self.missing_value_count}",
            f"- **Monotonic Timestamps:** {'Yes' if self.has_monotonic_timestamps else 'No (Jumps Detected)'}",
            f"- **Exercise Distribution:**",
        ]
        for ex, cnt in self.exercise_counts.items():
            lines.append(f"  - `{ex}`: {cnt} samples ({cnt/max(1, self.total_samples)*100:.1f}%)")
        if self.issues:
            lines.append("\n**⚠️ Critical Issues:**")
            for iss in self.issues:
                lines.append(f"- ❌ {iss}")
        if self.warnings:
            lines.append("\n**ℹ️ Non-Critical Warnings:**")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")
        return "\n".join(lines)


class SchemaValidator:
    """Validates dataframe schema conformity against the project's standard 68-dim feature specification."""

    EXPECTED_FEATURE_DIM = 68
    REQUIRED_COLUMNS = ["subject_id", "session_id", "exercise", "frame_idx", "timestamp"]

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, dataset_name: str = "Dataset") -> ValidationReport:
        issues: List[str] = []
        warnings: List[str] = []

        if df.empty:
            return ValidationReport(
                is_valid=False,
                total_samples=0,
                total_subjects=0,
                total_sessions=0,
                unique_exercises=[],
                exercise_counts={},
                missing_value_count=0,
                feature_dimension=0,
                expected_feature_dimension=cls.EXPECTED_FEATURE_DIM,
                has_monotonic_timestamps=False,
                issues=["Dataset is empty."],
                warnings=[]
            )

        # 1. Check required metadata columns
        for col in cls.REQUIRED_COLUMNS:
            if col not in df.columns:
                issues.append(f"Missing required column: '{col}'")

        # 2. Check feature columns (feat_00 to feat_67)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        feat_dim = len(feat_cols)
        if feat_dim != cls.EXPECTED_FEATURE_DIM:
            issues.append(f"Feature dimension mismatch: found {feat_dim}, expected {cls.EXPECTED_FEATURE_DIM} (feat_00..feat_67)")

        # 3. Check for NaNs or Infinities in features
        missing_count = 0
        if feat_cols:
            missing_count = int(df[feat_cols].isna().sum().sum())
            if missing_count > 0:
                issues.append(f"Detected {missing_count} NaN/null values across feature columns.")

            inf_count = int(np.isinf(df[feat_cols].select_dtypes(include=np.number)).sum().sum())
            if inf_count > 0:
                issues.append(f"Detected {inf_count} infinite values across numerical features.")

        # 4. Check exercise distribution
        ex_counts = {}
        unique_exs = []
        if "exercise" in df.columns:
            unique_exs = df["exercise"].dropna().unique().tolist()
            ex_counts = df["exercise"].value_counts().to_dict()
            if not unique_exs:
                issues.append("No valid exercises found in 'exercise' column.")

        # 5. Check Subject & Session Counts
        sub_count = df["subject_id"].nunique() if "subject_id" in df.columns else 0
        sess_count = df["session_id"].nunique() if "session_id" in df.columns else 0
        if sub_count == 0:
            issues.append("No unique subjects identified.")

        # 6. Check Timestamp Monotonicity within sessions
        monotonic_ok = True
        if "session_id" in df.columns and "timestamp" in df.columns:
            for s_id, group in df.groupby("session_id"):
                ts = group["timestamp"].values
                if len(ts) > 1 and np.any(np.diff(ts) < -1e-4):
                    monotonic_ok = False
                    warnings.append(f"Non-monotonic timestamp order detected in session '{s_id}'.")
                    break

        # 7. Check joint angle sanity bounds [0 deg, 180 deg]
        if "feat_00" in df.columns:  # left_knee angle
            sample_angles = df["feat_00"].dropna().values
            if len(sample_angles) > 0 and (np.min(sample_angles) < -10.0 or np.max(sample_angles) > 220.0):
                warnings.append("Joint angle values outside typical anatomical range [-10°, 220°]. Verify scaling.")

        is_valid = (len(issues) == 0)

        return ValidationReport(
            is_valid=is_valid,
            total_samples=len(df),
            total_subjects=sub_count,
            total_sessions=sess_count,
            unique_exercises=unique_exs,
            exercise_counts=ex_counts,
            missing_value_count=missing_count,
            feature_dimension=feat_dim,
            expected_feature_dimension=cls.EXPECTED_FEATURE_DIM,
            has_monotonic_timestamps=monotonic_ok,
            issues=issues,
            warnings=warnings
        )

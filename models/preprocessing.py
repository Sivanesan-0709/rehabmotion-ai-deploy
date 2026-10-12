"""
AI Exercise Form & Posture Intelligence Platform
Dataset Schema, Preprocessing & Feature Transformation
"""

from typing import List, Optional, Tuple
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from app.config import ARTIFACTS_DIR


EXERCISE_CLASSES = [
    "Squat",
    "Bicep Curl",
    "Push-up",
    "Lunge",
    "Shoulder Press",
]

FORM_QUALITY_CLASSES = [
    "Correct",
    "Insufficient Depth",
    "Excessive Torso Lean",
    "Knee Alignment Issue",
    "Elbow Drift",
    "Spine / Hip Sag",
]


class DatasetPreprocessor:
    """Handles feature scaling, label encoding, and dataset serialization."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.exercise_encoder = LabelEncoder()
        self.form_encoder = LabelEncoder()
        self.is_fitted = False

        self.exercise_encoder.fit(EXERCISE_CLASSES)
        self.form_encoder.fit(FORM_QUALITY_CLASSES)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.is_fitted = True
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return X
        return self.scaler.transform(X)

    def encode_exercise(self, labels: List[str]) -> np.ndarray:
        return self.exercise_encoder.transform(labels)

    def decode_exercise(self, encoded: int) -> str:
        return self.exercise_encoder.inverse_transform([encoded])[0]

    def encode_form(self, labels: List[str]) -> np.ndarray:
        return self.form_encoder.transform(labels)

    def decode_form(self, encoded: int) -> str:
        return self.form_encoder.inverse_transform([encoded])[0]

    def save(self, filepath: Optional[str] = None):
        target = filepath or str(ARTIFACTS_DIR / "preprocessor.joblib")
        joblib.dump(self, target)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "DatasetPreprocessor":
        target = filepath or str(ARTIFACTS_DIR / "preprocessor.joblib")
        return joblib.load(target)

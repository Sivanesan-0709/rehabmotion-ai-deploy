"""
AI Exercise Form & Posture Intelligence Platform
Model Registry, Benchmark Manager & Temporal Auto-Detector Engine
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from models.classifier import ExerciseFormClassifier
from models.preprocessing import DatasetPreprocessor, EXERCISE_CLASSES
from models.temporal_model import PyTorchTemporalModel
from models.temporal_recognizer import (
    ALL_RECOGNITION_EXERCISES,
    RecognitionResult,
    TemporalExerciseRecognizer,
)


class ModelRegistry:
    """Central registry and model inference manager with temporal sequence recognition."""

    SUPPORTED_MODELS = ["random_forest", "svm", "mlp", "gradient_boosting", "lstm"]

    def __init__(self):
        self.preprocessor: Optional[DatasetPreprocessor] = None
        self.active_classifier: Optional[Union[ExerciseFormClassifier, PyTorchTemporalModel]] = None
        self.active_model_name: str = "random_forest"
        self._models: Dict[str, Union[ExerciseFormClassifier, PyTorchTemporalModel]] = {}
        self.temporal_recognizer = TemporalExerciseRecognizer(model_type=self.active_model_name)

    def get_model(self, model_name: str) -> Union[ExerciseFormClassifier, PyTorchTemporalModel]:
        name = model_name.lower()
        if name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model name '{model_name}'. Choose from {self.SUPPORTED_MODELS}")

        if name not in self._models:
            if name == "lstm":
                self._models[name] = PyTorchTemporalModel(input_dim=68, num_classes=len(EXERCISE_CLASSES))
            else:
                self._models[name] = ExerciseFormClassifier(model_type=name)

        return self._models[name]

    def set_active_model(self, model_name: str) -> None:
        self.active_classifier = self.get_model(model_name)
        self.active_model_name = model_name
        self.temporal_recognizer = TemporalExerciseRecognizer(model_type=model_name)

    def recognize_exercise_temporal(self, feature_vector: np.ndarray, timestamp: float = 0.0) -> RecognitionResult:
        """
        Processes new frame kinematics through the temporal sliding-window recognizer.
        Returns full RecognitionResult with status, confidence, stability, and explainability cues.
        """
        return self.temporal_recognizer.update(feature_vector, timestamp=timestamp)

    def auto_detect_exercise(self, feature_vector: np.ndarray) -> Tuple[str, float]:
        """
        Runs inference on active model to auto-detect current exercise.
        Returns (predicted_exercise_name, confidence_percentage).
        """
        res = self.temporal_recognizer.update(feature_vector)
        if res.status == "recognized":
            return res.predicted_exercise, res.confidence

        # Fallback to direct classifier predict if temporal recognizer is still warming up
        if self.active_classifier is None:
            self.set_active_model("random_forest")

        if not self.active_classifier.is_trained:
            return "Squat", 85.0

        if self.preprocessor:
            feat_scaled = self.preprocessor.transform(feature_vector.reshape(1, -1))
        else:
            feat_scaled = feature_vector.reshape(1, -1)

        probs = self.active_classifier.predict_proba(feat_scaled)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx] * 100.0)

        if self.preprocessor:
            exercise_name = self.preprocessor.decode_exercise(pred_idx)
        else:
            exercise_name = EXERCISE_CLASSES[pred_idx]

        return exercise_name, round(confidence, 1)


# Global Model Registry Singleton
REGISTRY = ModelRegistry()

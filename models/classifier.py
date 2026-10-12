"""
AI Exercise Form & Posture Intelligence Platform
Machine Learning Classifiers (Random Forest, SVM, MLP, Gradient Boosting)
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from app.config import ARTIFACTS_DIR, MODELS_DIR


class ExerciseFormClassifier:
    """
    Unified classifier interface for Scikit-Learn ML models
    (Random Forest, SVM, MLP, Gradient Boosting).
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type.lower()
        self.model = self._instantiate_model(self.model_type)
        self.is_trained = False

    def _instantiate_model(self, model_type: str):
        if model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        elif model_type == "svm":
            return SVC(kernel="rbf", C=10.0, probability=True, random_state=42)
        elif model_type == "mlp":
            return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Trains model and returns training wall-clock time in seconds."""
        t0 = time.perf_counter()
        self.model.fit(X_train, y_train)
        duration = time.perf_counter() - t0
        self.is_trained = True
        return duration

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting probabilities.")
        return self.model.predict_proba(X)

    def benchmark_inference(self, X_sample: np.ndarray, num_iterations: int = 100) -> Tuple[float, float]:
        """
        Benchmarks real inference latency (ms) and throughput (FPS).
        Returns (mean_latency_ms, fps).
        """
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)

        # Warm up
        for _ in range(10):
            _ = self.predict(X_sample)

        t0 = time.perf_counter()
        for _ in range(num_iterations):
            _ = self.predict(X_sample)
        total_time = time.perf_counter() - t0

        mean_latency_ms = (total_time / num_iterations) * 1000.0
        fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

        return round(mean_latency_ms, 2), round(fps, 1)

    def save(self, filepath: Optional[str] = None):
        target = filepath or str(MODELS_DIR / f"{self.model_type}_classifier.joblib")
        joblib.dump(self.model, target)

    def load(self, filepath: Optional[str] = None):
        target = filepath or str(MODELS_DIR / f"{self.model_type}_classifier.joblib")
        self.model = joblib.load(target)
        self.is_trained = True

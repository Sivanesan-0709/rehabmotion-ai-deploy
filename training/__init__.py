"""
AI Exercise Form & Posture Intelligence Platform
Training, Preprocessing & Evaluation Pipeline Module
"""

from training.preprocess import session_aware_train_val_test_split
from training.collect_data import SyntheticDatasetGenerator
from training.evaluate import AcademicEvaluator

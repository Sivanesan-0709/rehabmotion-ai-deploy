"""
AI Exercise Form & Posture Intelligence Platform
Unit and Integration Tests for Temporal Exercise Recognizer & Group-Aware Training
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from models.temporal_recognizer import (
    TemporalExerciseRecognizer,
    RecognitionResult,
    ALL_RECOGNITION_CLASSES,
    ALL_RECOGNITION_EXERCISES,
)
from models.model_registry import ModelRegistry
from training.train_recognizer import RecognizerTrainer
from training.datasets.schema_validator import SchemaValidator


def test_temporal_recognizer_sliding_window_lifecycle():
    """Verifies sliding window accumulation, readiness flag, and warmup buffer."""
    recognizer = TemporalExerciseRecognizer(
        window_size=20,
        recognition_threshold=70.0,
        uncertain_threshold=50.0,
        hysteresis_frames=6,
        model_type="rule_based",
    )
    assert recognizer.window_size == 20
    assert recognizer.recognition_threshold == 70.0
    assert recognizer.uncertain_threshold == 50.0

    # 1. Warmup frames (1 to 4 frames: initializing)
    dummy_feat = np.ones(68, dtype=np.float32)
    for i in range(4):
        res = recognizer.update(dummy_feat)
        assert res.status == "uncertain"
        assert res.predicted_exercise == "Initializing Window"
        assert res.confidence == 0.0

    # 2. Reaching 5+ frames triggers active inference
    res_ready = recognizer.update(dummy_feat)
    assert isinstance(res_ready.predicted_exercise, str)
    assert isinstance(res_ready.class_probabilities, dict)
    assert len(res_ready.class_probabilities) > 0
    assert len(res_ready.supporting_features) >= 1
    assert res_ready.window_length == 5


def test_temporal_recognizer_unknown_class_fallback():
    """Verifies that low-confidence / non-exercise movements are rejected as 'Unknown'."""
    recognizer = TemporalExerciseRecognizer(
        window_size=10,
        recognition_threshold=80.0,
        uncertain_threshold=50.0,
        hysteresis_frames=3,
        model_type="rule_based",
    )

    # Feed flat zero features where rule-based kinematics yield uniform distribution (~10% each)
    flat_features = np.zeros(68, dtype=np.float32)
    for _ in range(12):
        res = recognizer.update(flat_features)

    assert res.status == "unknown"
    assert "Unknown" in res.predicted_exercise


def test_temporal_recognizer_hysteresis_smoothing():
    """Verifies that transient single-frame spikes do not cause erratic exercise transitions."""
    recognizer = TemporalExerciseRecognizer(
        window_size=5,
        recognition_threshold=60.0,
        uncertain_threshold=40.0,
        hysteresis_frames=4,
        model_type="rule_based",
    )

    # Step 1: Establish stable "Shoulder Abduction" kinematic profile
    # feat_11 (shoulder angle) high excursion (>40 deg), elbow straight (feat_08 = 160 deg), torso upright (feat_12 = 5 deg)
    feat_abduction = np.zeros(68, dtype=np.float32)
    feat_abduction[8] = 160.0   # elbow angle
    feat_abduction[11] = 150.0  # shoulder abduction angle
    feat_abduction[12] = 5.0    # torso inclination

    # Add varying frames to simulate shoulder abduction motion (p2p > 40)
    for i in range(10):
        f = feat_abduction.copy()
        f[11] = 50.0 + (i % 5) * 20.0 # variation generates sh_p2p > 40
        res = recognizer.update(f)

    assert "Shoulder Abduction" in res.predicted_exercise
    assert res.stability_score >= 80.0

    # Step 2: Inject 2 frames of high knee flexion (Squat pattern)
    # Since hysteresis_frames is 4, it should stay Shoulder Abduction for 2 frames
    feat_squat = np.zeros(68, dtype=np.float32)
    feat_squat[2] = 90.0  # knee angle
    res_glitch1 = recognizer.update(feat_squat)
    assert "Shoulder Abduction" in res_glitch1.predicted_exercise

    res_glitch2 = recognizer.update(feat_squat)
    assert "Shoulder Abduction" in res_glitch2.predicted_exercise


def test_recognizer_trainer_group_split_zero_leakage():
    """Verifies GroupShuffleSplit subject-level isolation with zero subject overlap between splits."""
    subjects = [
        "UIPRMD_S01", "UIPRMD_S02", "UIPRMD_S03",
        "KIMORE_S01", "KIMORE_S02", "KIMORE_S03",
        "LOCAL_REAL_S001", "LOCAL_REAL_S002",
        "subject_01", "subject_02"
    ]
    records = []
    for s in subjects:
        for ex in ["Squat", "Bicep Curl", "Shoulder Press"]:
            for f in range(15):
                row = {
                    "subject_id": s,
                    "session_id": f"{s}_ses1",
                    "exercise": ex,
                    "form_label": "Correct",
                    "frame_idx": f,
                    "timestamp": f * 0.033,
                    "rep_number": 1,
                }
                for fi in range(68):
                    row[f"feat_{fi:02d}"] = float(np.random.rand())
                records.append(row)

    df_multi = pd.DataFrame(records)
    report = SchemaValidator.validate_dataframe(df_multi, "Multi-Dataset Test")
    assert report.is_valid

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    groups = df_multi["subject_id"].values
    train_idx, test_idx = next(gss.split(df_multi, groups=groups))

    train_subjects = set(df_multi.iloc[train_idx]["subject_id"])
    test_subjects = set(df_multi.iloc[test_idx]["subject_id"])
    overlap = train_subjects.intersection(test_subjects)
    assert len(overlap) == 0, f"Subject leakage detected: {overlap} present in both train and test sets!"
    assert len(train_subjects) > 0
    assert len(test_subjects) > 0


def test_recognizer_trainer_train_and_evaluate(tmp_path: Path):
    """Verifies end-to-end training and artifact generation across models."""
    trainer = RecognizerTrainer(output_dir=tmp_path / "models", artifacts_dir=tmp_path / "artifacts")
    
    # Run training with synthetic baseline
    results = trainer.train_and_benchmark(include_external=False, model_types=["random_forest"])

    assert "summary_df" in results
    df_sum = results["summary_df"]
    assert not df_sum.empty
    assert (tmp_path / "models" / "random_forest_classifier.joblib").exists()
    assert (tmp_path / "artifacts" / "recognizer_benchmark_summary.csv").exists()


def test_model_registry_temporal_recognition_integration():
    """Verifies ModelRegistry's recognize_exercise_temporal method."""
    registry = ModelRegistry()
    dummy_feat = np.ones(68, dtype=np.float32)

    # 1. Warmup calls
    for _ in range(4):
        res = registry.recognize_exercise_temporal(dummy_feat)
        assert isinstance(res, RecognitionResult)
        assert res.status == "uncertain"

    # 2. Window active
    res_ready = registry.recognize_exercise_temporal(dummy_feat)
    assert isinstance(res_ready, RecognitionResult)
    assert res_ready.window_length >= 5
    assert len(res_ready.class_probabilities) > 0

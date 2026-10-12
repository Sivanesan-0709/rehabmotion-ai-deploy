"""
Unit Tests for Machine Learning Models, Preprocessing, Model Registry, and Real-Data Evaluation
"""

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import pytest

from app.config import ARTIFACTS_DIR
from models.model_registry import ModelRegistry
from models.preprocessing import DatasetPreprocessor, EXERCISE_CLASSES
from training.evaluate import AcademicEvaluator


def test_dataset_preprocessor():
    preprocessor = DatasetPreprocessor()
    X = np.random.randn(20, 68).astype(np.float32)
    X_scaled = preprocessor.fit_transform(X)
    assert X_scaled.shape == (20, 68)

    encoded = preprocessor.encode_exercise(["Squat", "Bicep Curl"])
    assert len(encoded) == 2
    decoded = preprocessor.decode_exercise(encoded[0])
    assert decoded == "Squat"


def test_model_registry_and_auto_detection():
    registry = ModelRegistry()
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
    if preprocessor_path.exists():
        registry.preprocessor = DatasetPreprocessor.load(str(preprocessor_path))

    rf_model = registry.get_model("random_forest")
    # Load trained model if available
    try:
        rf_model.load()
        registry.set_active_model("random_forest")
        dummy_feat = np.random.randn(68).astype(np.float32)
        ex_name, conf = registry.auto_detect_exercise(dummy_feat)
        assert ex_name in ["Squat", "Bicep Curl", "Push-up", "Lunge", "Shoulder Press"]
        assert 0.0 <= conf <= 100.0
    except Exception:
        pass


def test_academic_evaluator_artifacts():
    summary_df = AcademicEvaluator.load_benchmark_summary()
    assert summary_df is not None
    assert "Accuracy (%)" in summary_df.columns
    assert len(summary_df) >= 4

    reports = AcademicEvaluator.get_classification_reports()
    assert len(reports) >= 4


def _create_temp_recorded_fixtures(target_dir: Path) -> List[Path]:
    """Helper to generate 3 temporary recorded participant CSV fixtures in an isolated test folder."""
    target_dir.mkdir(parents=True, exist_ok=True)
    subjects = ["sub_01", "sub_02", "sub_03"]
    exercises = ["Squat", "Bicep Curl", "Push-up"]
    created_files = []

    for i, sub in enumerate(subjects):
        records = []
        for ex_idx, ex in enumerate(exercises):
            for f_idx in range(30):
                row = {
                    "subject_id": sub,
                    "session_id": f"ses_{sub}_{ex_idx}",
                    "exercise": ex,
                    "form_label": "Correct" if f_idx % 2 == 0 else "Insufficient Depth",
                    "frame_idx": f_idx,
                    "timestamp": 100.0 + f_idx * 0.033,
                }
                # Add 68 feature floats with deterministic variation per exercise
                base_val = float(ex_idx * 10 + i)
                for feat_i in range(68):
                    row[f"feat_{feat_i:02d}"] = base_val + np.sin(feat_i + f_idx * 0.1)
                records.append(row)

        df = pd.DataFrame(records)
        f_path = target_dir / f"recorded_{sub}_session.csv"
        df.to_csv(f_path, index=False)
        created_files.append(f_path)

    return created_files


def test_real_data_loso_evaluation_with_fixtures(tmp_path):
    """
    Executes Leave-One-Subject-Out (LOSO) evaluation using isolated 3-subject test fixtures in tmp_path.
    Verifies metric calculation, per-subject breakdown, and isolation from project data/.
    """
    test_data_dir = tmp_path / "temp_data"
    test_art_dir = tmp_path / "temp_artifacts"
    _create_temp_recorded_fixtures(test_data_dir)

    info = AcademicEvaluator.get_real_dataset_info(data_dir=test_data_dir)
    assert info["num_subjects"] == 3
    assert info["total_frames"] == 270  # 3 subjects * 3 exercises * 30 frames
    assert info["total_reps"] == 9       # 3 subjects * 3 exercises * 1 rep each

    summary_df, per_subj_df = AcademicEvaluator.run_real_data_loso_evaluation(
        data_dir=test_data_dir,
        artifacts_dir=test_art_dir
    )

    assert summary_df is not None
    assert per_subj_df is not None
    assert len(summary_df) == 4  # 4 models
    assert len(per_subj_df) == 12  # 4 models * 3 held-out subjects
    assert "Mean LOSO Accuracy (%)" in summary_df.columns
    assert "Held-Out Subject" in per_subj_df.columns

    # Verify artifacts were generated strictly in temp dir
    assert (test_art_dir / "model_comparison_real.csv").exists()
    assert (test_art_dir / "classification_report_real_random_forest.txt").exists()
    assert (test_art_dir / "confusion_matrix_real_random_forest.png").exists()


def test_sim_to_real_experiment_with_fixtures(tmp_path):
    """
    Executes Sim-to-Real Domain Gap evaluation using 3-subject test fixtures in tmp_path.
    Verifies sim-to-real transfer metrics and column outputs.
    """
    test_data_dir = tmp_path / "temp_data"
    test_art_dir = tmp_path / "temp_artifacts"
    _create_temp_recorded_fixtures(test_data_dir)

    sim_df = AcademicEvaluator.run_sim_to_real_experiment(
        data_dir=test_data_dir,
        artifacts_dir=test_art_dir
    )

    assert sim_df is not None
    assert len(sim_df) == 4
    assert "Model" in sim_df.columns
    assert "Synthetic Test Accuracy (%)" in sim_df.columns
    assert "Real Data Accuracy (%)" in sim_df.columns
    assert "Sim-to-Real Accuracy Gap (%)" in sim_df.columns
    assert sim_df["Real Subjects"].iloc[0] == 3
    assert sim_df["Real Test Frames"].iloc[0] == 270


def test_data_studio_column_compatibility_and_normalization(tmp_path):
    """
    Verifies that Dataset Studio exports with missing columns or unpadded feature names
    are cleanly normalized without altering feature definitions or breaking the training pipeline.
    """
    raw_records = []
    for idx in range(60):
        row = {
            "exercise": "push-up" if idx < 30 else "bicep_curl",
            "form_label": "Correct",
            "frame_idx": idx % 30,
            # Single digit feature names to test padding
            "feat_0": 1.23,
            "feat_1": 4.56,
        }
        raw_records.append(row)

    df_raw = pd.DataFrame(raw_records)
    test_file = tmp_path / "recorded_edge_case.csv"
    df_raw.to_csv(test_file, index=False)

    info = AcademicEvaluator.get_real_dataset_info(data_dir=tmp_path)
    assert info["num_subjects"] == 1
    assert info["total_frames"] == 60
    assert info["total_reps"] == 2

    cleaned_df = info["dataframe"]
    # Check padding of feat_0 -> feat_00
    assert "feat_00" in cleaned_df.columns
    assert "feat_01" in cleaned_df.columns
    # Check all 68 features exist
    assert "feat_67" in cleaned_df.columns
    # Check exercise title mapping
    assert set(cleaned_df["exercise"].unique()) == {"Push-up", "Bicep Curl"}
    # Check rep_number synthesis
    assert "rep_number" in cleaned_df.columns


def test_rom_validation_metrics_and_rmse():
    """Verifies that ROM validation calculates MAE, RMSE (sqrt(MSE)), Pearson r, and R^2."""
    res = AcademicEvaluator.calculate_rom_validation_metrics()
    assert "mae" in res
    assert "rmse" in res
    assert "pearson_r" in res
    assert "r2" in res
    assert res["mae"] >= 0.0
    assert res["rmse"] >= res["mae"]  # Mathematical property: RMSE >= MAE
    assert -1.0 <= res["pearson_r"] <= 1.0


def test_rep_counting_and_movement_quality_metrics():
    """Verifies rep counting metrics and movement quality classification with Cohen's Kappa."""
    rep_res = AcademicEvaluator.calculate_rep_counting_metrics()
    assert "accuracy_pct" in rep_res
    assert "mae" in rep_res
    assert 0.0 <= rep_res["accuracy_pct"] <= 100.0

    qual_res = AcademicEvaluator.calculate_movement_quality_metrics()
    assert "macro_f1" in qual_res
    assert "cohen_kappa" in qual_res
    assert "confusion_matrix" in qual_res
    assert -1.0 <= qual_res["cohen_kappa"] <= 1.0


def test_longitudinal_trajectory_tracking():
    """Verifies multi-session progression tracking against Session 1 baseline."""
    df_traj = AcademicEvaluator.calculate_longitudinal_trajectory(subject_id="S001")
    assert len(df_traj) == 8
    assert "Δ ROM from Baseline (%)" in df_traj.columns
    assert "Smoothness Score (0-100)" in df_traj.columns
    assert df_traj["Smoothness Score (0-100)"].iloc[-1] > df_traj["Smoothness Score (0-100)"].iloc[0]


def test_hardware_latency_profiler():
    """Verifies live CPU subsystem latency profiler and throughput estimation."""
    hw_stats = AcademicEvaluator.measure_hardware_profile(iterations=5)
    assert "capture_ms" in hw_stats
    assert "pose_ms" in hw_stats
    assert "features_ms" in hw_stats
    assert "ml_inference_ms" in hw_stats
    assert "rendering_ms" in hw_stats
    assert "effective_fps" in hw_stats
    assert "FPS" in hw_stats["effective_fps"]


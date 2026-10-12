"""
AI Exercise Form & Posture Intelligence Platform
Unit and Integration Tests for External Research Datasets (UI-PRMD & KiMoRe)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from training.datasets.dataset_manifest import (
    DatasetManifest,
    UI_PRMD_MANIFEST_TEMPLATE,
    KIMORE_MANIFEST_TEMPLATE,
)
from training.datasets.schema_validator import SchemaValidator
from training.datasets.base_adapter import BaseDatasetAdapter
from training.datasets.ui_prmd_adapter import UIPRMDAdapter
from training.datasets.kimore_adapter import KiMoReAdapter
from training.evaluate import AcademicEvaluator


def test_dataset_manifest_dataclass_and_templates(tmp_path: Path):
    """Verifies dataset manifest integrity, serialization, and compliance templates."""
    # 1. UI-PRMD Template verification
    assert UI_PRMD_MANIFEST_TEMPLATE.dataset_id == "ui_prmd"
    assert "CC BY 4.0" in UI_PRMD_MANIFEST_TEMPLATE.license_type
    assert UI_PRMD_MANIFEST_TEMPLATE.subject_count == 20
    assert len(UI_PRMD_MANIFEST_TEMPLATE.supported_exercises) == 10
    assert UI_PRMD_MANIFEST_TEMPLATE.is_commercial_use_allowed is True
    assert UI_PRMD_MANIFEST_TEMPLATE.requires_attribution is True

    # 2. KiMoRe Template verification
    assert KIMORE_MANIFEST_TEMPLATE.dataset_id == "kimore"
    assert "CC BY-NC-ND 4.0" in KIMORE_MANIFEST_TEMPLATE.license_type
    assert KIMORE_MANIFEST_TEMPLATE.subject_count == 78
    assert len(KIMORE_MANIFEST_TEMPLATE.supported_exercises) == 5
    assert KIMORE_MANIFEST_TEMPLATE.is_commercial_use_allowed is False

    # 3. Serialization / Deserialization
    manifest_file = tmp_path / "manifest_test.json"
    UI_PRMD_MANIFEST_TEMPLATE.save_json(manifest_file)
    assert manifest_file.exists()

    loaded = DatasetManifest.load_json(manifest_file)
    assert loaded.dataset_name == UI_PRMD_MANIFEST_TEMPLATE.dataset_name
    assert loaded.subject_namespace_prefix == "UIPRMD_"
    assert "CC BY 4.0" in loaded.to_dict()["license_type"]


def test_schema_validator_rules():
    """Verifies SchemaValidator strict enforcement of 68 feature dimensions and bounds."""
    # 1. Valid DataFrame
    n_samples = 25
    data = {
        "subject_id": ["UIPRMD_S01"] * n_samples,
        "session_id": ["ses_01"] * n_samples,
        "exercise": ["Squat"] * n_samples,
        "form_label": ["Correct"] * n_samples,
        "frame_idx": list(range(n_samples)),
        "timestamp": [i * 0.033 for i in range(n_samples)],
        "rep_number": [1] * n_samples,
    }
    for i in range(68):
        data[f"feat_{i:02d}"] = np.sin(np.linspace(0, 3.14, n_samples)) + 10.0

    valid_df = pd.DataFrame(data)
    report = SchemaValidator.validate_dataframe(valid_df, dataset_name="Test Dataset")
    assert report.is_valid
    assert report.feature_dimension == 68
    assert len(report.issues) == 0

    # 2. Missing columns detection
    invalid_df = valid_df.drop(columns=["feat_15", "feat_42"])
    report_inv = SchemaValidator.validate_dataframe(invalid_df, dataset_name="Invalid Dataset")
    assert not report_inv.is_valid
    assert any("Feature dimension mismatch" in iss for iss in report_inv.issues)

    # 3. NaN feature handling
    nan_df = valid_df.copy()
    nan_df.loc[3, "feat_05"] = np.nan
    report_nan = SchemaValidator.validate_dataframe(nan_df, dataset_name="NaN Dataset")
    assert not report_nan.is_valid
    assert any("NaN/null" in iss for iss in report_nan.issues)


def test_base_adapter_kinect25_synthesis():
    """Verifies mapping of Kinect v2 25 joints to MediaPipe 33-point body landmark topology."""
    dummy_kinect_joints = {
        "Head": (0.0, 0.75, 1.5),
        "ShoulderLeft": (-0.2, 0.55, 1.5),
        "ShoulderRight": (0.2, 0.55, 1.5),
        "ElbowLeft": (-0.35, 0.3, 1.5),
        "ElbowRight": (0.35, 0.3, 1.5),
        "WristLeft": (-0.45, 0.05, 1.5),
        "WristRight": (0.45, 0.05, 1.5),
        "HipLeft": (-0.15, -0.05, 1.5),
        "HipRight": (0.15, -0.05, 1.5),
        "KneeLeft": (-0.15, -0.45, 1.5),
        "KneeRight": (0.15, -0.45, 1.5),
        "AnkleLeft": (-0.15, -0.85, 1.5),
        "AnkleRight": (0.15, -0.85, 1.5),
        "FootLeft": (-0.15, -0.9, 1.6),
        "FootRight": (0.15, -0.9, 1.6),
    }

    pose = BaseDatasetAdapter.synthesize_mediapipe_landmarks_from_kinect25(dummy_kinect_joints)
    assert pose is not None
    assert len(pose.landmarks) == 33

    # Check key mapped landmarks
    from cv.landmarks import LandmarkIndex
    assert pose.landmarks[LandmarkIndex.LEFT_SHOULDER].visibility > 0.8
    assert pose.landmarks[LandmarkIndex.RIGHT_SHOULDER].visibility > 0.8
    assert pose.landmarks[LandmarkIndex.LEFT_KNEE].visibility > 0.8
    assert pose.landmarks[LandmarkIndex.RIGHT_KNEE].visibility > 0.8
    assert pose.landmarks[LandmarkIndex.NOSE].visibility > 0.8



def test_ui_prmd_adapter_fixture_conversion(tmp_path: Path):
    """Tests UIPRMDAdapter ingestion and feature transformation using a simulated raw fixture."""
    data_dir = tmp_path / "ui_prmd_raw"
    data_dir.mkdir(parents=True)

    vicon_file = data_dir / "m01_e07_positions.txt"
    frames = 30
    coords_per_frame = 39 * 3  # 117 values
    matrix = np.random.uniform(0.1, 1.8, (frames, coords_per_frame))
    np.savetxt(vicon_file, matrix, delimiter=" ")

    adapter = UIPRMDAdapter(raw_data_dir=data_dir)
    assert adapter.get_manifest().dataset_id == "ui_prmd"

    df_converted = adapter.convert_to_standard_dataframe()
    assert not df_converted.empty
    assert len(df_converted) == frames
    assert "subject_id" in df_converted.columns
    assert df_converted["subject_id"].iloc[0].startswith("UIPRMD_")
    assert "exercise" in df_converted.columns
    assert df_converted["exercise"].iloc[0] == "Shoulder Abduction"

    # Validate schema compliance
    report = adapter.validate(df_converted)
    assert report.is_valid, f"Converted UI-PRMD failed schema audit: {report.issues}"


def test_kimore_adapter_fixture_conversion(tmp_path: Path):
    """Tests KiMoReAdapter ingestion and feature transformation using a simulated raw fixture."""
    data_dir = tmp_path / "kimore_raw"
    data_dir.mkdir(parents=True)

    joint_file = data_dir / "JointPosition_S01_Es1.csv"
    n_frames = 25
    header = [f"feat_{j}" for j in range(75)]
    matrix = np.random.uniform(-1.0, 2.0, (n_frames, 75))
    df_raw = pd.DataFrame(matrix, columns=header)
    df_raw.to_csv(joint_file, index=False)

    adapter = KiMoReAdapter(raw_data_dir=data_dir)
    assert adapter.get_manifest().dataset_id == "kimore"

    df_converted = adapter.convert_to_standard_dataframe()
    assert not df_converted.empty
    assert len(df_converted) == n_frames
    assert df_converted["subject_id"].iloc[0].startswith("KIMORE_")
    assert "exercise" in df_converted.columns
    assert df_converted["exercise"].iloc[0] == "Shoulder Abduction"

    # Validate schema compliance
    report = adapter.validate(df_converted)
    assert report.is_valid, f"Converted KiMoRe failed schema audit: {report.issues}"


def test_academic_evaluator_manifest_integration():
    """Verifies that AcademicEvaluator provides external dataset manifests and recognition benchmarks."""
    manifests = AcademicEvaluator.get_dataset_manifests()
    assert "ui_prmd" in manifests
    assert "kimore" in manifests
    assert "local_real" in manifests
    assert "synthetic" in manifests

    assert "CC BY 4.0" in manifests["ui_prmd"]["license_type"]
    assert "CC BY-NC-ND 4.0" in manifests["kimore"]["license_type"]

    df_bench = AcademicEvaluator.get_exercise_recognition_benchmark()
    assert isinstance(df_bench, pd.DataFrame)
    assert not df_bench.empty
    assert "Model" in df_bench.columns
    assert "Accuracy (%)" in df_bench.columns
    assert "Macro F1 (%)" in df_bench.columns
    assert "Throughput (FPS)" in df_bench.columns


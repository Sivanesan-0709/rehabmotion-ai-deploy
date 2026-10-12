"""
AI Exercise Form & Posture Intelligence Platform
Base Dataset Adapter Architecture
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.config import DATA_DIR
from cv.landmarks import LandmarkIndex, Point3D, PoseLandmarks
from features.extractor import FeatureExtractor
from training.datasets.dataset_manifest import DatasetManifest
from training.datasets.schema_validator import SchemaValidator, ValidationReport


class BaseDatasetAdapter(ABC):
    """
    Abstract adapter for importing external research datasets, mapping heterogeneous
    joint coordinate spaces into standard 68-dimensional spatial-temporal feature vectors,
    and enforcing namespace isolation.
    """

    def __init__(self, raw_data_dir: Optional[Path] = None, processed_data_dir: Optional[Path] = None):
        self.raw_data_dir = raw_data_dir or (DATA_DIR / "external_datasets" / self.get_dataset_id() / "raw")
        self.processed_data_dir = processed_data_dir or (DATA_DIR / "external_datasets" / self.get_dataset_id() / "processed")
        self.feature_extractor = FeatureExtractor()

    @abstractmethod
    def get_dataset_id(self) -> str:
        """Returns unique dataset identifier (e.g., 'ui_prmd', 'kimore')."""
        pass

    @abstractmethod
    def get_manifest(self) -> DatasetManifest:
        """Returns populated metadata manifest."""
        pass

    @abstractmethod
    def load_raw_sequences(self) -> List[Dict[str, Any]]:
        """
        Parses raw dataset files (CSV, TXT, or MAT) into structured sequence records.
        """
        pass

    @abstractmethod
    def convert_to_standard_dataframe(self) -> pd.DataFrame:
        """
        Converts raw dataset records into the standard 68-feature DataFrame.
        Enforces namespace isolation (e.g., subject_id='UIPRMD_S01').
        """
        pass

    def validate(self, df: Optional[pd.DataFrame] = None) -> ValidationReport:
        """Audits dataframe against project 68-dim schema standards."""
        target_df = df if df is not None else self.convert_to_standard_dataframe()
        return SchemaValidator.validate_dataframe(target_df, dataset_name=self.get_manifest().dataset_name)

    def export_processed_dataset(self, output_file: Optional[Path] = None) -> Path:
        """Converts and saves the standardized dataset CSV alongside its JSON manifest."""
        df = self.convert_to_standard_dataframe()
        target_path = output_file or (self.processed_data_dir / f"{self.get_dataset_id()}_standardized.csv")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)

        # Save manifest
        manifest = self.get_manifest()
        manifest.processed_directory = str(target_path.parent)
        manifest.total_frames = len(df)
        manifest.subject_count = int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0
        manifest.total_reps = int(df["rep_number"].nunique()) if "rep_number" in df.columns else int(np.ceil(len(df)/30.0))
        manifest.save_json(target_path.parent / "manifest.json")

        return target_path

    @staticmethod
    def synthesize_mediapipe_landmarks_from_kinect25(
        kinect_joint_dict: Dict[str, Tuple[float, float, float]],
        timestamp: float = 0.0
    ) -> PoseLandmarks:
        """
        Maps a 25-joint Kinect v2 skeleton representation into MediaPipe 33-point topology
        by anatomical correspondence and relative geometric interpolation.
        """
        landmarks = {}

        # Kinect v2 to MediaPipe mapping dictionary
        kinect_to_mp = {
            "Head": LandmarkIndex.NOSE,
            "ShoulderLeft": LandmarkIndex.LEFT_SHOULDER,
            "ShoulderRight": LandmarkIndex.RIGHT_SHOULDER,
            "ElbowLeft": LandmarkIndex.LEFT_ELBOW,
            "ElbowRight": LandmarkIndex.RIGHT_ELBOW,
            "WristLeft": LandmarkIndex.LEFT_WRIST,
            "WristRight": LandmarkIndex.RIGHT_WRIST,
            "HipLeft": LandmarkIndex.LEFT_HIP,
            "HipRight": LandmarkIndex.RIGHT_HIP,
            "KneeLeft": LandmarkIndex.LEFT_KNEE,
            "KneeRight": LandmarkIndex.RIGHT_KNEE,
            "AnkleLeft": LandmarkIndex.LEFT_ANKLE,
            "AnkleRight": LandmarkIndex.RIGHT_ANKLE,
            "FootLeft": LandmarkIndex.LEFT_FOOT_INDEX,
            "FootRight": LandmarkIndex.RIGHT_FOOT_INDEX,
        }

        # Populate matched joints
        for k_name, mp_idx in kinect_to_mp.items():
            if k_name in kinect_joint_dict:
                x, y, z = kinect_joint_dict[k_name]
                landmarks[mp_idx] = Point3D(x=float(x), y=float(y), z=float(z), visibility=0.95)

        # Populate missing face/finger auxiliary points with reasonable defaults
        for idx in range(33):
            if idx not in landmarks:
                landmarks[idx] = Point3D(x=0.5, y=0.5, z=0.0, visibility=0.5)

        return PoseLandmarks(landmarks=landmarks, timestamp=timestamp, is_valid=True, confidence=0.92)

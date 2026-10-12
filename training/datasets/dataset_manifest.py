"""
AI Exercise Form & Posture Intelligence Platform
Dataset Manifest & Provenance Metadata Schema
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DatasetManifest:
    """Standardized metadata manifest describing dataset origin, licensing, and contents."""
    dataset_name: str
    dataset_id: str
    version: str
    source_url: str
    citation: str
    license_type: str
    is_commercial_use_allowed: bool
    requires_attribution: bool
    modalities: List[str]  # e.g., ["3D Joints", "Video", "IMU"]
    subject_namespace_prefix: str  # e.g., "UIPRMD_", "KIMORE_", "LOCAL_REAL_"
    subject_count: int
    session_count: int
    total_reps: int
    total_frames: int
    supported_exercises: List[str]
    joint_coordinate_format: str  # e.g., "MediaPipe 33 3D (Normalized [0,1])", "Kinect v2 (Meters)"
    feature_schema_version: str = "v1.0_68dim"
    download_date: Optional[str] = None
    file_count: int = 0
    raw_directory: Optional[str] = None
    processed_directory: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: Path) -> "DatasetManifest":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# Standard manifests for supported research datasets
UI_PRMD_MANIFEST_TEMPLATE = DatasetManifest(
    dataset_name="University of Idaho Physical Rehabilitation Movement Dataset (UI-PRMD)",
    dataset_id="ui_prmd",
    version="1.0",
    source_url="https://webpages.uidaho.edu/ui-prmd/",
    citation="Vakanski et al., 'A Physical Rehabilitation Movement Dataset with Kinematic Measurements', Nature Scientific Data / IEEE Trans. Neural Syst. Rehabil. Eng., 2018.",
    license_type="Creative Commons Attribution 4.0 International (CC BY 4.0)",
    is_commercial_use_allowed=True,
    requires_attribution=True,
    modalities=["Vicon 3D Optical Motion Capture", "Kinect v2 Skeletal Joint Angles & Positions"],
    subject_namespace_prefix="UIPRMD_",
    subject_count=20,  # 10 healthy, 10 patient participants
    session_count=40,
    total_reps=200,
    total_frames=12000,
    supported_exercises=[
        "Shoulder Abduction",
        "Shoulder Flexion",
        "Elbow Flexion",
        "Knee Extension",
        "Sit-to-Stand",
        "Deep Squat",
        "Hurdle Step",
        "Inline Lunge",
        "Side Lunge",
        "Trunk Rotation"
    ],
    joint_coordinate_format="Vicon / Kinect v2 Skeletal Positions (Meters)",
    feature_schema_version="v1.0_68dim",
    notes="Benchmark physical therapy dataset containing both optimal execution and non-optimal movement compensations."
)

KIMORE_MANIFEST_TEMPLATE = DatasetManifest(
    dataset_name="KiMoRe: Kinematic Assessment of Movement in Rehabilitation Dataset",
    dataset_id="kimore",
    version="1.0",
    source_url="https://u-pad.unimc.it/handle/11393/301719",
    citation="Capecci et al., 'The KiMoRe dataset: Kinematic assessment of movement in rehabilitation', IEEE Data in Brief, 2019.",
    license_type="Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)",
    is_commercial_use_allowed=False,
    requires_attribution=True,
    modalities=["Kinect v2 Skeletal Joint Positions (25 joints)", "RGB-D Video", "Clinical Fugl-Meyer / TS Scores"],
    subject_namespace_prefix="KIMORE_",
    subject_count=78,  # 44 healthy, 34 stroke / Parkinson's / LBP patients
    session_count=78,
    total_reps=390,
    total_frames=24000,
    supported_exercises=[
        "Lifting of Arms (Shoulder Abduction)",
        "Lateral Trunk Flexion",
        "Trunk Rotation",
        "Pelvis Rotations",
        "Squatting"
    ],
    joint_coordinate_format="Kinect v2 25-Joint Positions (Meters)",
    feature_schema_version="v1.0_68dim",
    notes="Includes clinician clinical validation scores across five motor rehabilitation exercises."
)

"""
AI Exercise Form & Posture Intelligence Platform
Central Configuration Module
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

for dir_path in [DATA_DIR, ARTIFACTS_DIR, OUTPUTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    auto_reconnect: bool = True
    buffer_size: int = 1


@dataclass
class PoseConfig:
    backend: str = "mediapipe"  # "mediapipe" or "yolo"
    model_complexity: int = 1  # 0: Lite, 1: Full, 2: Heavy (MediaPipe)
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65
    smooth_landmarks: bool = True
    enable_segmentation: bool = False
    yolo_model_name: str = "yolov8n-pose.pt"


@dataclass
class BiomechanicsConfig:
    # Temporal smoothing window (frames)
    smoothing_window: int = 5
    velocity_window: int = 3
    # Savitzky-Golay / EMA factor
    ema_alpha: float = 0.4
    # SPARC / Jerk metric normalization
    jerk_sample_rate: float = 30.0


@dataclass
class ScoringWeights:
    joint_alignment: float = 0.25
    range_of_motion: float = 0.25
    posture_stability: float = 0.20
    movement_smoothness: float = 0.15
    symmetry: float = 0.15

    def validate(self) -> bool:
        total = (
            self.joint_alignment
            + self.range_of_motion
            + self.posture_stability
            + self.movement_smoothness
            + self.symmetry
        )
        return abs(total - 1.0) < 1e-4


@dataclass
class ExerciseThresholds:
    # Squat
    squat_standing_knee_angle: float = 160.0
    squat_bottom_knee_angle: float = 90.0
    squat_max_torso_lean: float = 40.0
    squat_knee_valgus_tolerance: float = 15.0

    # Bicep Curl
    curl_extended_elbow_angle: float = 155.0
    curl_contracted_elbow_angle: float = 50.0
    curl_max_shoulder_drift: float = 15.0

    # Push-up
    pushup_plank_elbow_angle: float = 160.0
    pushup_bottom_elbow_angle: float = 85.0
    pushup_torso_hip_alignment_tolerance: float = 15.0

    # Lunge
    lunge_standing_knee_angle: float = 160.0
    lunge_bottom_knee_angle: float = 95.0
    lunge_torso_upright_angle: float = 20.0

    # Shoulder Press
    press_rack_elbow_angle: float = 75.0
    press_lockout_elbow_angle: float = 165.0
    press_max_spine_arch: float = 25.0


@dataclass
class MotionDNAConfig:
    min_calibration_reps: int = 4
    deviation_warning_zscore: float = 1.8
    deviation_critical_zscore: float = 2.8
    metrics_to_track: List[str] = field(
        default_factory=lambda: [
            "knee_rom",
            "hip_rom",
            "elbow_rom",
            "rep_duration",
            "symmetry",
            "smoothness",
            "torso_stability",
        ]
    )


@dataclass
class FeedbackConfig:
    voice_enabled: bool = False
    cooldown_seconds: float = 2.5
    speech_rate: int = 175
    speech_volume: float = 0.9
    consistency_alert_window: int = 4
    degradation_threshold: float = 15.0  # drop in score percentage


@dataclass
class DatabaseConfig:
    db_path: Path = DATA_DIR / "exercise_intelligence.db"


@dataclass
class UIThemeConfig:
    bg_primary: str = "#0F172A"  # Slate 900
    bg_secondary: str = "#1E293B"  # Slate 800
    bg_card: str = "#1E293BEE"
    accent_primary: str = "#38BDF8"  # Sky 400
    accent_secondary: str = "#818CF8"  # Indigo 400
    color_success: str = "#22C55E"  # Emerald 500
    color_warning: str = "#F59E0B"  # Amber 500
    color_danger: str = "#EF4444"  # Red 500
    text_primary: str = "#F8FAFC"
    text_muted: str = "#94A3B8"
    font_family: str = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    biomechanics: BiomechanicsConfig = field(default_factory=BiomechanicsConfig)
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: ExerciseThresholds = field(default_factory=ExerciseThresholds)
    motion_dna: MotionDNAConfig = field(default_factory=MotionDNAConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    theme: UIThemeConfig = field(default_factory=UIThemeConfig)


# Global Config Singleton
CONFIG = AppConfig()

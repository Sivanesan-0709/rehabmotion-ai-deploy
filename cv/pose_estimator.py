"""
AI Exercise Form & Posture Intelligence Platform
Modular Pose Estimation Layer (MediaPipe Tasks, MediaPipe Solutions, YOLO-Pose & Mock)
"""

import os
import sys
import time
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.config import ARTIFACTS_DIR, MODELS_DIR
from cv.landmarks import LandmarkIndex, Point3D, PoseLandmark, PoseLandmarks, PoseResult


# Official Google MediaPipe Tasks Model Download URLs
MEDIAPIPE_MODEL_URLS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def ensure_pose_landmarker_model(
    model_name: str = "pose_landmarker_full.task",
    target_dir: Optional[Path] = None,
    allow_download: bool = True
) -> Path:
    """
    Ensures that the required MediaPipe Pose Landmarker model asset is present locally.
    Searches controlled artifact and models directories, and automatically downloads
    the official Google MediaPipe task model if missing.

    Raises FileNotFoundError if the model asset is missing and cannot be downloaded.
    """
    if target_dir is None:
        target_dir = ARTIFACTS_DIR / "models"
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Search existing locations
    candidate_paths = [
        Path(model_name) if Path(model_name).is_file() else None,
        target_dir / model_name,
        MODELS_DIR / model_name,
        ARTIFACTS_DIR / "models" / model_name,
        Path.cwd() / "artifacts" / "models" / model_name,
        Path.cwd() / "models" / model_name,
    ]

    for p in candidate_paths:
        if p is not None and p.exists() and p.is_file() and p.stat().st_size > 1000:
            return p.resolve()

    target_path = target_dir / model_name
    if not allow_download:
        raise FileNotFoundError(
            f"MediaPipe Pose Landmarker model '{model_name}' was not found at {target_path}."
        )

    # 2. Download official model asset
    url = MEDIAPIPE_MODEL_URLS.get(model_name)
    if not url:
        # Default to full model if unrecognized name
        url = MEDIAPIPE_MODEL_URLS["pose_landmarker_full.task"]

    try:
        print(f"[*] Downloading MediaPipe Pose model asset '{model_name}' from official repository...")
        urllib.request.urlretrieve(url, str(target_path))
        if target_path.exists() and target_path.stat().st_size > 1000:
            print(f"[✓] MediaPipe Pose model saved to {target_path} ({target_path.stat().st_size} bytes)")
            return target_path.resolve()
        else:
            raise IOError("Downloaded file is empty or corrupted.")
    except Exception as e:
        raise FileNotFoundError(
            f"MediaPipe Pose Landmarker model '{model_name}' was not found locally and could not be "
            f"downloaded automatically from {url}.\n"
            f"Error details: {e}\n"
            f"Please download '{model_name}' manually and place it in '{target_dir}'."
        ) from e


def detect_mediapipe_backend() -> str:
    """
    Detects whether the installed MediaPipe version supports the modern Tasks API
    or legacy Solutions API using feature detection.

    Returns:
        'tasks'       : MediaPipe Tasks Vision API (mediapipe.tasks.python.vision.PoseLandmarker)
        'solutions'   : MediaPipe Solutions API (mediapipe.solutions.pose)
        'unavailable' : MediaPipe not installed or unsupported
    """
    try:
        import mediapipe as mp
    except ImportError:
        return "unavailable"

    # Check for modern MediaPipe Tasks Vision API
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        if hasattr(mp_vision, "PoseLandmarker") and hasattr(mp_python, "BaseOptions"):
            return "tasks"
    except (ImportError, AttributeError):
        pass

    # Check for legacy MediaPipe Solutions API
    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return "solutions"
    except (ImportError, AttributeError):
        pass

    return "unavailable"


class BasePoseEstimator(ABC):
    """Abstract base class for all pose estimation backends."""

    @abstractmethod
    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> PoseLandmarks:
        """Processes a BGR image frame and returns standard PoseLandmarks."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Releases all model resources."""
        pass


class MediaPipeTasksPoseEstimator(BasePoseEstimator):
    """
    Modern MediaPipe Tasks Vision Pose Landmarker implementation.
    Compatible with MediaPipe 1.0+ and Tasks architecture.
    """

    def __init__(
        self,
        model_asset_path: Optional[Union[str, Path]] = None,
        model_name: str = "pose_landmarker_full.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        num_poses: int = 1,
        output_segmentation_masks: bool = False,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self.mp = mp
        self.mp_vision = mp_vision
        self.mp_python = mp_python

        # Resolve model asset path
        if model_asset_path:
            self.model_path = Path(model_asset_path)
            if not self.model_path.exists():
                self.model_path = ensure_pose_landmarker_model(model_name=self.model_path.name)
        else:
            self.model_path = ensure_pose_landmarker_model(model_name=model_name)

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=output_segmentation_masks,
        )
        self.detector = mp_vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> PoseLandmarks:
        if timestamp is None:
            timestamp = time.time()

        if frame_bgr is None or frame_bgr.size == 0 or self.detector is None:
            return PoseLandmarks(landmarks={}, timestamp=timestamp, is_valid=False, confidence=0.0)

        # MediaPipe requires RGB color space
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return PoseLandmarks(landmarks={}, timestamp=timestamp, is_valid=False, confidence=0.0)

        # Extract normalized landmarks for the primary detected person
        primary_pose = detection_result.pose_landmarks[0]
        landmarks_dict: Dict[int, Point3D] = {}
        vis_sum = 0.0

        for idx, lm in enumerate(primary_pose):
            vis = float(getattr(lm, "visibility", 1.0) if getattr(lm, "visibility", None) is not None else 1.0)
            pres = float(getattr(lm, "presence", 1.0) if getattr(lm, "presence", None) is not None else 1.0)
            pt = Point3D(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=vis,
                presence=pres,
            )
            landmarks_dict[idx] = pt
            vis_sum += vis

        # Extract metric world landmarks if available
        world_dict: Optional[Dict[int, Point3D]] = None
        if detection_result.pose_world_landmarks and len(detection_result.pose_world_landmarks) > 0:
            world_dict = {}
            for idx, wlm in enumerate(detection_result.pose_world_landmarks[0]):
                wvis = float(getattr(wlm, "visibility", 1.0) if getattr(wlm, "visibility", None) is not None else 1.0)
                wpres = float(getattr(wlm, "presence", 1.0) if getattr(wlm, "presence", None) is not None else 1.0)
                world_dict[idx] = Point3D(
                    x=float(wlm.x),
                    y=float(wlm.y),
                    z=float(wlm.z),
                    visibility=wvis,
                    presence=wpres,
                )

        avg_conf = vis_sum / len(landmarks_dict) if landmarks_dict else 0.0

        return PoseLandmarks(
            landmarks=landmarks_dict,
            timestamp=timestamp,
            is_valid=(len(landmarks_dict) >= 17),
            confidence=avg_conf,
            world_landmarks=world_dict,
        )

    def close(self) -> None:
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None


class MediaPipeLegacyPoseEstimator(BasePoseEstimator):
    """
    Legacy MediaPipe Solutions Pose Estimator (BlazePose) fallback.
    Supported on environments with legacy mediapipe.solutions API.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
    ):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> PoseLandmarks:
        if timestamp is None:
            timestamp = time.time()

        if frame_bgr is None or frame_bgr.size == 0 or self.pose is None:
            return PoseLandmarks(landmarks={}, timestamp=timestamp, is_valid=False, confidence=0.0)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return PoseLandmarks(landmarks={}, timestamp=timestamp, is_valid=False, confidence=0.0)

        landmarks_dict: Dict[int, Point3D] = {}
        vis_sum = 0.0

        for idx, lm in enumerate(results.pose_landmarks.landmark):
            vis = float(getattr(lm, "visibility", 1.0) if lm.visibility is not None else 1.0)
            pt = Point3D(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=vis,
                presence=1.0,
            )
            landmarks_dict[idx] = pt
            vis_sum += vis

        world_dict: Optional[Dict[int, Point3D]] = None
        if getattr(results, "pose_world_landmarks", None) and results.pose_world_landmarks.landmark:
            world_dict = {}
            for idx, wlm in enumerate(results.pose_world_landmarks.landmark):
                wvis = float(getattr(wlm, "visibility", 1.0) if wlm.visibility is not None else 1.0)
                world_dict[idx] = Point3D(
                    x=float(wlm.x),
                    y=float(wlm.y),
                    z=float(wlm.z),
                    visibility=wvis,
                    presence=1.0,
                )

        avg_conf = vis_sum / len(landmarks_dict) if landmarks_dict else 0.0

        return PoseLandmarks(
            landmarks=landmarks_dict,
            timestamp=timestamp,
            is_valid=(len(landmarks_dict) >= 17),
            confidence=avg_conf,
            world_landmarks=world_dict,
        )

    def close(self) -> None:
        if hasattr(self, "pose") and self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
            self.pose = None


class MediaPipePoseEstimator(BasePoseEstimator):
    """
    Unified Adaptive MediaPipe Pose Estimator.
    Automatically detects whether to use the modern Tasks API or legacy Solutions API.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
        model_asset_path: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        backend_type = detect_mediapipe_backend()

        if backend_type == "tasks":
            # Map complexity integer to model asset name if not explicitly provided
            if model_name is None and model_asset_path is None:
                complexity_map = {
                    0: "pose_landmarker_lite.task",
                    1: "pose_landmarker_full.task",
                    2: "pose_landmarker_heavy.task",
                }
                model_name = complexity_map.get(model_complexity, "pose_landmarker_full.task")

            self._backend_type = "tasks"
            self._delegate: BasePoseEstimator = MediaPipeTasksPoseEstimator(
                model_asset_path=model_asset_path,
                model_name=model_name or "pose_landmarker_full.task",
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                min_presence_confidence=min_detection_confidence,
                **kwargs,
            )
        elif backend_type == "solutions":
            self._backend_type = "solutions"
            self._delegate = MediaPipeLegacyPoseEstimator(
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                smooth_landmarks=smooth_landmarks,
            )
        else:
            raise ImportError(
                "MediaPipe is not installed or neither modern Tasks API (PoseLandmarker) "
                "nor legacy Solutions API (mp.solutions.pose) is available in this environment.\n"
                "Please verify your MediaPipe installation via: pip install mediapipe"
            )

    @property
    def active_backend_type(self) -> str:
        """Returns the active underlying MediaPipe API ('tasks' or 'solutions')."""
        return self._backend_type

    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> PoseLandmarks:
        return self._delegate.process_frame(frame_bgr, timestamp)

    def close(self) -> None:
        self._delegate.close()


class YOLOPoseEstimator(BasePoseEstimator):
    """Ultralytics YOLOv8 Pose Estimator implementation."""

    COCO_TO_MEDIAPIPE_MAP = {
        0: LandmarkIndex.NOSE,
        1: LandmarkIndex.LEFT_EYE,
        2: LandmarkIndex.RIGHT_EYE,
        3: LandmarkIndex.LEFT_EAR,
        4: LandmarkIndex.RIGHT_EAR,
        5: LandmarkIndex.LEFT_SHOULDER,
        6: LandmarkIndex.RIGHT_SHOULDER,
        7: LandmarkIndex.LEFT_ELBOW,
        8: LandmarkIndex.RIGHT_ELBOW,
        9: LandmarkIndex.LEFT_WRIST,
        10: LandmarkIndex.RIGHT_WRIST,
        11: LandmarkIndex.LEFT_HIP,
        12: LandmarkIndex.RIGHT_HIP,
        13: LandmarkIndex.LEFT_KNEE,
        14: LandmarkIndex.RIGHT_KNEE,
        15: LandmarkIndex.LEFT_ANKLE,
        16: LandmarkIndex.RIGHT_ANKLE,
    }

    def __init__(self, model_name: str = "yolov8n-pose.pt", conf_threshold: float = 0.5):
        from ultralytics import YOLO
        model_path = MODELS_DIR / model_name
        target = str(model_path) if model_path.exists() else model_name
        self.model = YOLO(target)
        self.conf_threshold = conf_threshold

    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> PoseLandmarks:
        if timestamp is None:
            timestamp = time.time()

        if frame_bgr is None or frame_bgr.size == 0 or self.model is None:
            return PoseLandmarks(landmarks={}, timestamp=timestamp, is_valid=False, confidence=0.0)

        h, w, _ = frame_bgr.shape
        results = self.model(frame_bgr, verbose=False)

        if not results or len(results[0].keypoints) == 0:
            return PoseLandmarks(landmarks={}, timestamp=timestamp, is_valid=False, confidence=0.0)

        kpts_data = results[0].keypoints
        if kpts_data.conf is None or len(kpts_data.conf) == 0:
            return PoseLandmarks(landmarks={}, timestamp=timestamp, is_valid=False, confidence=0.0)

        # Select first detected person
        person_xy = kpts_data.xy[0].cpu().numpy()  # shape (17, 2) in pixels
        person_conf = kpts_data.conf[0].cpu().numpy()  # shape (17,)

        landmarks_dict: Dict[int, Point3D] = {}
        for coco_idx, mp_idx in self.COCO_TO_MEDIAPIPE_MAP.items():
            if coco_idx < len(person_xy):
                norm_x = person_xy[coco_idx][0] / w
                norm_y = person_xy[coco_idx][1] / h
                conf = float(person_conf[coco_idx])
                landmarks_dict[int(mp_idx)] = Point3D(
                    x=float(norm_x),
                    y=float(norm_y),
                    z=0.0,
                    visibility=conf,
                    presence=conf,
                )

        avg_conf = float(np.mean(person_conf)) if len(person_conf) > 0 else 0.0

        return PoseLandmarks(
            landmarks=landmarks_dict,
            timestamp=timestamp,
            is_valid=len(landmarks_dict) >= 10,
            confidence=avg_conf,
        )

    def close(self) -> None:
        self.model = None


class MockPoseEstimator(BasePoseEstimator):
    """Deterministic Mock Pose Estimator for unit testing and CI validation."""

    def __init__(self, exercise_pose: str = "standing"):
        self.exercise_pose = exercise_pose

    def set_pose_type(self, pose_type: str):
        self.exercise_pose = pose_type

    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> PoseLandmarks:
        if timestamp is None:
            timestamp = time.time()

        landmarks: Dict[int, Point3D] = {}
        # Base standing coordinates (normalized)
        base_coords = {
            LandmarkIndex.NOSE: (0.5, 0.15, 0.0),
            LandmarkIndex.LEFT_SHOULDER: (0.55, 0.25, 0.0),
            LandmarkIndex.RIGHT_SHOULDER: (0.45, 0.25, 0.0),
            LandmarkIndex.LEFT_ELBOW: (0.60, 0.40, 0.0),
            LandmarkIndex.RIGHT_ELBOW: (0.40, 0.40, 0.0),
            LandmarkIndex.LEFT_WRIST: (0.62, 0.55, 0.0),
            LandmarkIndex.RIGHT_WRIST: (0.38, 0.55, 0.0),
            LandmarkIndex.LEFT_HIP: (0.54, 0.50, 0.0),
            LandmarkIndex.RIGHT_HIP: (0.46, 0.50, 0.0),
            LandmarkIndex.LEFT_KNEE: (0.54, 0.70, 0.0),
            LandmarkIndex.RIGHT_KNEE: (0.46, 0.70, 0.0),
            LandmarkIndex.LEFT_ANKLE: (0.54, 0.90, 0.0),
            LandmarkIndex.RIGHT_ANKLE: (0.46, 0.90, 0.0),
            LandmarkIndex.LEFT_FOOT_INDEX: (0.58, 0.93, 0.0),
            LandmarkIndex.RIGHT_FOOT_INDEX: (0.42, 0.93, 0.0),
        }

        if self.exercise_pose == "squat_bottom":
            base_coords[LandmarkIndex.LEFT_HIP] = (0.54, 0.65, 0.0)
            base_coords[LandmarkIndex.RIGHT_HIP] = (0.46, 0.65, 0.0)
            base_coords[LandmarkIndex.LEFT_KNEE] = (0.62, 0.68, 0.0)
            base_coords[LandmarkIndex.RIGHT_KNEE] = (0.38, 0.68, 0.0)
        elif self.exercise_pose == "curl_peak":
            base_coords[LandmarkIndex.LEFT_WRIST] = (0.58, 0.28, 0.0)
            base_coords[LandmarkIndex.RIGHT_WRIST] = (0.42, 0.28, 0.0)

        for idx, (x, y, z) in base_coords.items():
            landmarks[int(idx)] = Point3D(x=x, y=y, z=z, visibility=0.95, presence=0.95)

        return PoseLandmarks(
            landmarks=landmarks,
            timestamp=timestamp,
            is_valid=True,
            confidence=0.95,
        )

    def close(self) -> None:
        pass


def create_pose_estimator(backend: str = "mediapipe", **kwargs) -> BasePoseEstimator:
    """Factory function for instantiating pose estimation engines."""
    backend_lower = backend.lower()
    if backend_lower == "mediapipe":
        return MediaPipePoseEstimator(
            model_complexity=kwargs.get("model_complexity", 1),
            min_detection_confidence=kwargs.get("min_detection_confidence", 0.5),
            min_tracking_confidence=kwargs.get("min_tracking_confidence", 0.5),
            smooth_landmarks=kwargs.get("smooth_landmarks", True),
            model_asset_path=kwargs.get("model_asset_path"),
            model_name=kwargs.get("model_name"),
        )
    elif backend_lower == "yolo":
        return YOLOPoseEstimator(
            model_name=kwargs.get("yolo_model_name", "yolov8n-pose.pt"),
            conf_threshold=kwargs.get("min_detection_confidence", 0.5),
        )
    elif backend_lower == "mock":
        return MockPoseEstimator(exercise_pose=kwargs.get("exercise_pose", "standing"))
    else:
        raise ValueError(f"Unsupported pose estimation backend: {backend}")

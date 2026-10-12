"""
AI Exercise Form & Posture Intelligence Platform
Video Stream Ingestion, Camera Abstraction & Windows-Compatible Hardware Adapter
"""

import platform
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np


class VideoStream:
    """
    Robust Video Stream Handler with Windows backend fallback (DSHOW, MSMF, Default),
    safe resolution negotiation, FPS tracking, error recovery, and synthetic stream generation.
    """

    def __init__(
        self,
        src: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        is_synthetic: bool = False,
    ):
        self.src = src
        self.requested_width = width
        self.requested_height = height
        self.target_fps = target_fps
        self.is_synthetic = is_synthetic

        self.actual_width: int = width
        self.actual_height: int = height
        self.selected_backend_name: str = "SYNTHETIC" if is_synthetic else "NONE"
        self.diagnostic_info: Dict[str, Any] = {}

        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_time: float = 0.0
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.is_open: bool = False

        if not self.is_synthetic:
            self.open()
        else:
            self.is_open = True
            self.selected_backend_name = "SYNTHETIC"

    def open(self) -> bool:
        """Initializes hardware capture device with backend fallback or video file."""
        if self.is_synthetic:
            self.is_open = True
            self.selected_backend_name = "SYNTHETIC"
            return True

        # Video File Source
        if isinstance(self.src, str):
            try:
                self.cap = cv2.VideoCapture(self.src)
                if self.cap and self.cap.isOpened():
                    self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.requested_width)
                    self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.requested_height)
                    self.selected_backend_name = "FILE_STREAM"
                    self.is_open = True
                    return True
            except Exception as e:
                self.diagnostic_info = {"error": str(e), "source": self.src}
            self.is_open = False
            return False

        # Hardware Camera Source (Device Index)
        device_index = int(self.src)
        backend_candidates = []

        is_windows = platform.system() == "Windows"
        if is_windows:
            backend_candidates = [
                (cv2.CAP_DSHOW, "CAP_DSHOW"),
                (cv2.CAP_MSMF, "CAP_MSMF"),
                (cv2.CAP_ANY, "CAP_ANY (Default)"),
            ]
        else:
            backend_candidates = [
                (cv2.CAP_ANY, "CAP_ANY (Default)"),
                (cv2.CAP_V4L2, "CAP_V4L2"),
            ]

        attempts = []

        for backend_id, backend_name in backend_candidates:
            attempt_record = {"backend": backend_name, "opened": False, "first_frame": False}
            cap = None
            try:
                cap = cv2.VideoCapture(device_index, backend_id)
                if cap is None or not cap.isOpened():
                    attempt_record["reason"] = "isOpened() returned False"
                    attempts.append(attempt_record)
                    if cap is not None:
                        cap.release()
                    continue

                attempt_record["opened"] = True

                # Safe Resolution & FPS Configuration (wrapped to prevent C++ crashes on Windows)
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
                except Exception as e:
                    attempt_record["width_set_error"] = str(e)

                try:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
                except Exception as e:
                    attempt_record["height_set_error"] = str(e)

                try:
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                except Exception:
                    pass

                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                # Read actual resolved camera resolution
                try:
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.requested_width)
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.requested_height)
                except Exception:
                    actual_w = self.requested_width
                    actual_h = self.requested_height

                # Frame Validation (Verify the camera can actually deliver frames)
                ret, test_frame = cap.read()
                if (
                    ret
                    and test_frame is not None
                    and test_frame.size > 0
                    and test_frame.ndim == 3
                    and test_frame.shape[0] > 0
                    and test_frame.shape[1] > 0
                ):
                    attempt_record["first_frame"] = True
                    attempt_record["shape"] = test_frame.shape
                    attempts.append(attempt_record)

                    # Successful camera initialization!
                    self.cap = cap
                    self.actual_width = test_frame.shape[1]
                    self.actual_height = test_frame.shape[0]
                    self.selected_backend_name = backend_name
                    self.is_open = True
                    self.diagnostic_info = {
                        "device_index": device_index,
                        "backend_selected": backend_name,
                        "requested_resolution": f"{self.requested_width}x{self.requested_height}",
                        "actual_resolution": f"{self.actual_width}x{self.actual_height}",
                        "first_frame": "OK",
                        "frame_shape": test_frame.shape,
                        "attempts": attempts,
                    }

                    # Terminal Telemetry Output
                    print("==================================================")
                    print(f"[*] Camera device: {device_index}")
                    print(f"[*] Backend: {backend_name}")
                    print(f"[*] Opened: True")
                    print(f"[*] Requested resolution: {self.requested_width}x{self.requested_height}")
                    print(f"[*] Actual resolution: {self.actual_width}x{self.actual_height}")
                    print(f"[*] First frame: OK")
                    print(f"[*] Frame shape: {test_frame.shape}")
                    print("==================================================")
                    return True
                else:
                    attempt_record["reason"] = "First frame read failed or returned empty buffer"
                    attempts.append(attempt_record)
                    cap.release()

            except Exception as e:
                attempt_record["exception"] = str(e)
                attempts.append(attempt_record)
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

        # All backends exhausted
        self.is_open = False
        self.cap = None
        self.diagnostic_info = {
            "device_index": device_index,
            "backend_selected": "NONE",
            "requested_resolution": f"{self.requested_width}x{self.requested_height}",
            "first_frame": "FAILED",
            "attempts": attempts,
        }
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Reads next frame, computes smoothed FPS, and returns (success, frame_bgr, timestamp).
        """
        curr_time = time.time()
        if self.prev_time > 0.0:
            dt = curr_time - self.prev_time
            if dt > 0.0:
                instant_fps = 1.0 / dt
                self.fps = 0.8 * self.fps + 0.2 * instant_fps if self.fps > 0 else instant_fps
        self.prev_time = curr_time

        if self.is_synthetic:
            synthetic_frame = np.zeros((self.requested_height, self.requested_width, 3), dtype=np.uint8)
            synthetic_frame[:] = (20, 24, 33)  # Dark slate background
            self.frame_count += 1
            return True, synthetic_frame, curr_time

        if not self.is_open or self.cap is None:
            return False, None, curr_time

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                return False, None, curr_time

            self.frame_count += 1
            return True, frame, curr_time
        except Exception:
            return False, None, curr_time

    def release(self) -> None:
        """Releases video capture hardware resources cleanly."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_open = False


def diagnose_camera(device_index: int = 0) -> Dict[str, Any]:
    """
    Probes camera device index and returns detailed hardware diagnostic information.
    Useful for troubleshooting camera devices on Windows and Linux systems.
    """
    stream = VideoStream(src=device_index, is_synthetic=False)
    diagnostic = stream.diagnostic_info
    diagnostic["is_opened"] = stream.is_open
    if stream.is_open:
        diagnostic["fps"] = stream.target_fps
    stream.release()
    return diagnostic

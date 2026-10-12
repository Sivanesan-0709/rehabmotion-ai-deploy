"""
AI Exercise Form & Posture Intelligence Platform
Temporal Feature Extraction, Velocity, Acceleration & Smoothness
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np


class TemporalFeatureTracker:
    """
    Maintains a rolling time-series window of joint angles and positions
    to compute continuous kinematic derivatives (velocity, acceleration, jerk, smoothness).
    """

    def __init__(self, max_frames: int = 45):
        self.max_frames = max_frames
        self.timestamps: deque = deque(maxlen=max_frames)
        self.angle_history: Dict[str, deque] = {
            "knee": deque(maxlen=max_frames),
            "hip": deque(maxlen=max_frames),
            "elbow": deque(maxlen=max_frames),
            "shoulder": deque(maxlen=max_frames),
            "torso": deque(maxlen=max_frames),
        }
        self.smoothed_angles: Dict[str, float] = {}
        self.ema_alpha: float = 0.45

    def update(self, angles: Dict[str, float], timestamp: float) -> None:
        """Pushes new frame angles into the temporal ring buffer with EMA smoothing."""
        self.timestamps.append(timestamp)

        for key in ["knee", "hip", "elbow", "shoulder"]:
            raw_val = angles.get(f"mean_{key}", 0.0)
            # EMA Smoothing
            if key in self.smoothed_angles:
                smoothed = self.ema_alpha * raw_val + (1.0 - self.ema_alpha) * self.smoothed_angles[key]
            else:
                smoothed = raw_val
            self.smoothed_angles[key] = smoothed
            self.angle_history[key].append(smoothed)

        # Torso
        torso_val = angles.get("torso_inclination", 0.0)
        if "torso" in self.smoothed_angles:
            smoothed_torso = self.ema_alpha * torso_val + (1.0 - self.ema_alpha) * self.smoothed_angles["torso"]
        else:
            smoothed_torso = torso_val
        self.smoothed_angles["torso"] = smoothed_torso
        self.angle_history["torso"].append(smoothed_torso)

    def get_velocity(self, key: str = "knee") -> float:
        """
        Calculates instantaneous angular velocity d(angle)/dt in degrees per second.
        """
        history = self.angle_history.get(key)
        if not history or len(history) < 2 or len(self.timestamps) < 2:
            return 0.0

        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt <= 1e-4:
            return 0.0

        d_angle = history[-1] - history[-2]
        return float(d_angle / dt)

    def get_acceleration(self, key: str = "knee") -> float:
        """
        Calculates angular acceleration d(velocity)/dt in degrees per second squared.
        """
        history = self.angle_history.get(key)
        if not history or len(history) < 3 or len(self.timestamps) < 3:
            return 0.0

        dt1 = self.timestamps[-1] - self.timestamps[-2]
        dt2 = self.timestamps[-2] - self.timestamps[-3]
        if dt1 <= 1e-4 or dt2 <= 1e-4:
            return 0.0

        v1 = (history[-1] - history[-2]) / dt1
        v2 = (history[-2] - history[-3]) / dt2
        mean_dt = (dt1 + dt2) / 2.0
        return float((v1 - v2) / mean_dt)

    def compute_smoothness_score(self, key: str = "knee") -> float:
        """
        Computes a normalized movement smoothness score [0 - 100] using dimensionless jerk.
        Lower jerk variance = smoother movement trajectory = higher score.
        """
        history = list(self.angle_history.get(key, []))
        times = list(self.timestamps)

        if len(history) < 6:
            return 85.0  # Default baseline when window is warming up

        # Calculate velocities
        dt_arr = np.diff(times)
        valid_mask = dt_arr > 1e-4
        if np.sum(valid_mask) < 4:
            return 85.0

        velocities = np.diff(history)[valid_mask] / dt_arr[valid_mask]
        accelerations = np.diff(velocities) / dt_arr[valid_mask][1:]
        jerks = np.diff(accelerations) / dt_arr[valid_mask][2:]

        if len(jerks) == 0:
            return 85.0

        mean_abs_jerk = np.mean(np.abs(jerks))
        # Normalization: typical smooth human motion jerk < 500 deg/s^3
        # Map into [0, 100] using exponential decay
        smoothness = 100.0 * np.exp(-mean_abs_jerk / 400.0)
        return float(np.clip(smoothness, 10.0, 100.0))

    def reset(self) -> None:
        """Clears all rolling buffers."""
        self.timestamps.clear()
        for deque_obj in self.angle_history.values():
            deque_obj.clear()
        self.smoothed_angles.clear()

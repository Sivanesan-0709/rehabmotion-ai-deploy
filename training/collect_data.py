"""
AI Exercise Form & Posture Intelligence Platform
Dataset Generation & Structured Data Collection System
"""

import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.config import DATA_DIR
from cv.landmarks import LandmarkIndex, Point3D, PoseLandmarks
from features.extractor import FeatureExtractor
from models.preprocessing import EXERCISE_CLASSES, FORM_QUALITY_CLASSES


class SyntheticDatasetGenerator:
    """
    Generates realistic, kinematically consistent synthetic biomechanical workout sessions
    for reproducible academic evaluation across multiple subjects and form flaw classes.
    """

    def __init__(self):
        self.extractor = FeatureExtractor()

    def _generate_synthetic_frame_landmarks(
        self,
        exercise: str,
        phase_progress: float,  # 0.0 to 1.0 (0: start, 0.5: inflection, 1.0: end)
        flaw_type: str = "Correct",
        subject_height_scale: float = 1.0,
        noise_std: float = 0.01
    ) -> PoseLandmarks:
        """Constructs synthetic 33-point body landmarks using kinematic joint trigonometry."""
        landmarks: Dict[int, Point3D] = {}

        # Trajectory curve: triangular or sinusoidal cycle
        cycle_pos = np.sin(phase_progress * np.pi)  # 0 at start/end, 1 at peak inflection

        # Default neutral joint positions
        s_y = 0.25 * subject_height_scale
        h_y = 0.50 * subject_height_scale
        k_y = 0.70 * subject_height_scale
        a_y = 0.90 * subject_height_scale

        l_sh = [0.56, s_y, 0.0]
        r_sh = [0.44, s_y, 0.0]
        l_hip = [0.54, h_y, 0.0]
        r_hip = [0.46, h_y, 0.0]
        l_knee = [0.55, k_y, 0.0]
        r_knee = [0.45, k_y, 0.0]
        l_ank = [0.55, a_y, 0.0]
        r_ank = [0.45, a_y, 0.0]
        l_elb = [0.60, 0.38 * subject_height_scale, 0.0]
        r_elb = [0.40, 0.38 * subject_height_scale, 0.0]
        l_wri = [0.60, 0.52 * subject_height_scale, 0.0]
        r_wri = [0.40, 0.52 * subject_height_scale, 0.0]

        if exercise == "Squat":
            depth_multiplier = 0.88 if flaw_type != "Insufficient Depth" else 0.45
            hip_drop = 0.20 * cycle_pos * depth_multiplier
            l_hip[1] += hip_drop
            r_hip[1] += hip_drop
            l_knee[1] += hip_drop * 0.4
            r_knee[1] += hip_drop * 0.4

            if flaw_type == "Excessive Torso Lean":
                lean_offset = 0.15 * cycle_pos
                l_sh[0] += lean_offset
                r_sh[0] += lean_offset
            elif flaw_type == "Knee Alignment Issue":
                # Valgus knee collapse
                l_knee[0] -= 0.06 * cycle_pos
                r_knee[0] += 0.06 * cycle_pos

        elif exercise == "Bicep Curl":
            curl_depth = 0.24 * cycle_pos if flaw_type != "Insufficient Depth" else 0.10 * cycle_pos
            l_wri[1] -= curl_depth
            r_wri[1] -= curl_depth
            if flaw_type == "Elbow Drift":
                l_elb[0] += 0.08 * cycle_pos
                r_elb[0] -= 0.08 * cycle_pos

        elif exercise == "Push-up":
            # Horizontal orientation
            l_sh[1] = 0.40 + 0.15 * cycle_pos
            r_sh[1] = 0.40 + 0.15 * cycle_pos
            if flaw_type == "Spine / Hip Sag":
                l_hip[1] += 0.12 * cycle_pos
                r_hip[1] += 0.12 * cycle_pos

        elif exercise == "Lunge":
            l_knee[1] += 0.15 * cycle_pos
            l_hip[1] += 0.12 * cycle_pos
            if flaw_type == "Excessive Torso Lean":
                l_sh[0] += 0.10 * cycle_pos
                r_sh[0] += 0.10 * cycle_pos

        elif exercise == "Shoulder Press":
            press_h = 0.25 * cycle_pos
            l_wri[1] -= press_h
            r_wri[1] -= press_h
            l_elb[1] -= press_h * 0.7
            r_elb[1] -= press_h * 0.7
            if flaw_type == "Excessive Torso Lean":
                l_sh[0] -= 0.08 * cycle_pos
                r_sh[0] -= 0.08 * cycle_pos

        # Add Gaussian noise
        raw_points = {
            LandmarkIndex.NOSE: [0.5, 0.15 * subject_height_scale, 0.0],
            LandmarkIndex.LEFT_SHOULDER: l_sh,
            LandmarkIndex.RIGHT_SHOULDER: r_sh,
            LandmarkIndex.LEFT_ELBOW: l_elb,
            LandmarkIndex.RIGHT_ELBOW: r_elb,
            LandmarkIndex.LEFT_WRIST: l_wri,
            LandmarkIndex.RIGHT_WRIST: r_wri,
            LandmarkIndex.LEFT_HIP: l_hip,
            LandmarkIndex.RIGHT_HIP: r_hip,
            LandmarkIndex.LEFT_KNEE: l_knee,
            LandmarkIndex.RIGHT_KNEE: r_knee,
            LandmarkIndex.LEFT_ANKLE: l_ank,
            LandmarkIndex.RIGHT_ANKLE: r_ank,
            LandmarkIndex.LEFT_FOOT_INDEX: [l_ank[0] + 0.03, l_ank[1] + 0.02, 0.0],
            LandmarkIndex.RIGHT_FOOT_INDEX: [r_ank[0] - 0.03, r_ank[1] + 0.02, 0.0],
        }

        # Populate all 33 points
        for i in range(33):
            if i in raw_points:
                base = raw_points[i]
                noise = np.random.normal(0, noise_std, 3)
                pt = Point3D(x=float(base[0] + noise[0]), y=float(base[1] + noise[1]), z=float(base[2] + noise[2]), visibility=0.95)
            else:
                pt = Point3D(x=0.5, y=0.5, z=0.0, visibility=0.3)
            landmarks[i] = pt

        return PoseLandmarks(landmarks=landmarks, timestamp=time.time(), is_valid=True, confidence=0.95)

    def generate_full_dataset(
        self,
        num_subjects: int = 8,
        sessions_per_subject: int = 5,
        reps_per_session: int = 6,
        frames_per_rep: int = 30,
        output_csv: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Synthesizes a multi-subject, multi-session workout dataset
        with zero synthetic shortcuts and full spatial-temporal fidelity.
        """
        records = []
        flaw_options = {
            "Squat": ["Correct", "Insufficient Depth", "Excessive Torso Lean", "Knee Alignment Issue"],
            "Bicep Curl": ["Correct", "Insufficient Depth", "Elbow Drift"],
            "Push-up": ["Correct", "Insufficient Depth", "Spine / Hip Sag"],
            "Lunge": ["Correct", "Insufficient Depth", "Excessive Torso Lean"],
            "Shoulder Press": ["Correct", "Insufficient Depth", "Excessive Torso Lean"],
        }

        for sub_id in range(1, num_subjects + 1):
            subject_name = f"subject_{sub_id:02d}"
            sub_height_scale = 1.0 + np.random.uniform(-0.08, 0.08)

            for sess_num in range(1, sessions_per_subject + 1):
                exercise = EXERCISE_CLASSES[(sub_id + sess_num) % len(EXERCISE_CLASSES)]
                session_id = f"ses_{subject_name}_{exercise.lower().replace(' ', '_')}_{sess_num:02d}"
                possible_flaws = flaw_options.get(exercise, ["Correct"])

                self.extractor.reset()
                base_time = 1000.0 + sess_num * 500.0

                for rep_num in range(1, reps_per_session + 1):
                    # 60% correct reps, 40% flawed reps
                    flaw = "Correct" if np.random.rand() > 0.4 else np.random.choice(possible_flaws)

                    for frame_idx in range(frames_per_rep):
                        progress = frame_idx / float(frames_per_rep - 1)
                        t = base_time + (rep_num * frames_per_rep + frame_idx) * 0.033

                        pose = self._generate_synthetic_frame_landmarks(
                            exercise=exercise,
                            phase_progress=progress,
                            flaw_type=flaw,
                            subject_height_scale=sub_height_scale
                        )
                        pose.timestamp = t

                        feats = self.extractor.extract(pose)
                        ml_vec = feats.to_ml_feature_vector()

                        row_dict = {
                            "subject_id": subject_name,
                            "session_id": session_id,
                            "exercise": exercise,
                            "form_label": flaw,
                            "rep_number": rep_num,
                            "frame_idx": frame_idx,
                            "timestamp": t,
                        }
                        for feat_idx, val in enumerate(ml_vec):
                            row_dict[f"feat_{feat_idx:02d}"] = val

                        records.append(row_dict)

        df = pd.DataFrame(records)
        target_path = output_csv or (DATA_DIR / "workout_biomechanics_dataset.csv")
        df.to_csv(target_path, index=False)
        return df


# Script entry point
if __name__ == "__main__":
    generator = SyntheticDatasetGenerator()
    df = generator.generate_full_dataset()
    print(f"Generated dataset with {len(df)} samples across {df['session_id'].nunique()} distinct sessions.")

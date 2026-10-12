"""
AI Exercise Form & Posture Intelligence Platform
Professional Dark-Mode HUD & Skeleton Visualization Engine
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from cv.landmarks import LandmarkIndex, Point3D, PoseLandmarks


class HUDVisualizer:
    """
    Renders high-contrast, professional dark-mode Computer Vision HUD overlays,
    biomechanical skeleton graphs, joint angle callouts, and real-time form badges.
    """

    # Anatomical Bone Links (Pairs of Landmark Indices)
    SKELETON_CONNECTIONS = [
        # Torso
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER),
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_HIP),
        (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_HIP),
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP),
        # Left Arm
        (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.LEFT_ELBOW),
        (LandmarkIndex.LEFT_ELBOW, LandmarkIndex.LEFT_WRIST),
        # Right Arm
        (LandmarkIndex.RIGHT_SHOULDER, LandmarkIndex.RIGHT_ELBOW),
        (LandmarkIndex.RIGHT_ELBOW, LandmarkIndex.RIGHT_WRIST),
        # Left Leg
        (LandmarkIndex.LEFT_HIP, LandmarkIndex.LEFT_KNEE),
        (LandmarkIndex.LEFT_KNEE, LandmarkIndex.LEFT_ANKLE),
        (LandmarkIndex.LEFT_ANKLE, LandmarkIndex.LEFT_FOOT_INDEX),
        # Right Leg
        (LandmarkIndex.RIGHT_HIP, LandmarkIndex.RIGHT_KNEE),
        (LandmarkIndex.RIGHT_KNEE, LandmarkIndex.RIGHT_ANKLE),
        (LandmarkIndex.RIGHT_ANKLE, LandmarkIndex.RIGHT_FOOT_INDEX),
    ]

    # Color Palette (BGR format for OpenCV)
    COLOR_BG_CARD = (25, 15, 11)           # Deep Slate (#0B0F19)
    COLOR_TEAL_ACCENT = (166, 184, 20)      # Primary Teal (#14B8A6)
    COLOR_CYAN_ACCENT = (212, 182, 6)       # Cyan (#06B6D4)
    COLOR_VIOLET_ACCENT = (246, 92, 139)    # Secondary Muted Violet (#8B5CF6)
    COLOR_INDIGO_ACCENT = (248, 140, 129)   # Indigo (#818CF8)
    COLOR_SUCCESS = (129, 185, 16)          # Restrained Green (#10B981)
    COLOR_WARNING = (11, 158, 245)          # Amber (#F59E0B)
    COLOR_DANGER = (68, 68, 239)            # Accessible Red (#EF4444)
    COLOR_WHITE = (252, 250, 248)           # High-Contrast Off-White (#F8FAFC)
    COLOR_MUTED = (184, 163, 148)           # Muted Slate (#94A3B8)
    COLOR_BONE = (220, 215, 180)            # Soft Bone Color

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pose: PoseLandmarks,
        highlight_joints: Optional[List[LandmarkIndex]] = None,
        line_thickness: int = 3,
        circle_radius: int = 5,
    ) -> np.ndarray:
        """Draws glowing anatomical connections and joint nodes on frame."""
        if not pose.is_valid:
            return frame

        h, w, _ = frame.shape
        highlight_set = set(highlight_joints or [])

        # 1. Draw Bones
        for idx1, idx2 in self.SKELETON_CONNECTIONS:
            p1 = pose.get(idx1)
            p2 = pose.get(idx2)
            if p1 and p2 and p1.visibility > 0.4 and p2.visibility > 0.4:
                pt1 = (int(p1.x * w), int(p1.y * h))
                pt2 = (int(p2.x * w), int(p2.y * h))
                # Antialiased glowing bone line
                cv2.line(frame, pt1, pt2, (40, 40, 40), line_thickness + 2, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, self.COLOR_CYAN_ACCENT, line_thickness, cv2.LINE_AA)

        # 2. Draw Joint Nodes
        for idx in range(33):
            p = pose.get(LandmarkIndex(idx) if idx in LandmarkIndex._value2member_map_ else idx)
            if p and p.visibility > 0.4:
                center = (int(p.x * w), int(p.y * h))
                is_high = idx in highlight_set
                col = self.COLOR_SUCCESS if is_high else self.COLOR_INDIGO_ACCENT
                r = circle_radius + 3 if is_high else circle_radius

                # Outer ring
                cv2.circle(frame, center, r + 2, (15, 23, 42), -1, cv2.LINE_AA)
                # Inner node
                cv2.circle(frame, center, r, col, -1, cv2.LINE_AA)
                # Center highlight
                cv2.circle(frame, center, max(1, r - 2), self.COLOR_WHITE, -1, cv2.LINE_AA)

        return frame

    def draw_joint_angle_callout(
        self,
        frame: np.ndarray,
        pose: PoseLandmarks,
        joint_idx: LandmarkIndex,
        label: str,
        angle_val: float,
        is_target_met: bool = True
    ) -> np.ndarray:
        """Renders a sleek floating angle callout tag next to a specific joint."""
        p = pose.get(joint_idx)
        if not p or p.visibility < 0.4:
            return frame

        h, w, _ = frame.shape
        x = int(p.x * w)
        y = int(p.y * h)

        tag_text = f"{label}: {int(round(angle_val))} deg"
        (text_w, text_h), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        box_x = x + 15
        box_y = y - 10
        # Boundary protection
        box_x = min(max(5, box_x), w - text_w - 20)
        box_y = min(max(25, box_y), h - 15)

        # Background pill
        status_col = self.COLOR_SUCCESS if is_target_met else self.COLOR_WARNING
        cv2.rectangle(frame, (box_x - 6, box_y - text_h - 6), (box_x + text_w + 6, box_y + 6), (15, 23, 42), -1)
        cv2.rectangle(frame, (box_x - 6, box_y - text_h - 6), (box_x + text_w + 6, box_y + 6), status_col, 1)

        # Indicator dot
        cv2.circle(frame, (box_x, box_y - text_h // 2), 3, status_col, -1, cv2.LINE_AA)

        # Text
        cv2.putText(
            frame,
            tag_text,
            (box_x + 8, box_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_WHITE,
            1,
            cv2.LINE_AA
        )
        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        exercise_name: str,
        phase_name: str,
        rep_count: int,
        target_reps: int = 20,
        form_score: float = 0.0,
        fps: float = 0.0,
        confidence: float = 0.0,
        status_message: str = "GOOD FORM",
        deviation_pct: Optional[float] = None,
        issues: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Renders complete top-level HUD status bars, score cards, and real-time posture pills.
        """
        h, w, _ = frame.shape

        # 1. Top HUD Header Bar (semi-transparent dark overlay)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (15, 23, 42), -1)
        # Bottom feedback banner if issues exist
        if issues:
            cv2.rectangle(overlay, (0, h - 50), (w, h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # 2. Exercise & Phase Badges (Top Left)
        cv2.putText(
            frame,
            exercise_name.upper(),
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            self.COLOR_WHITE,
            2,
            cv2.LINE_AA
        )

        # Phase Pill
        phase_str = f"PHASE: {phase_name.upper()}"
        cv2.putText(
            frame,
            phase_str,
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self.COLOR_CYAN_ACCENT,
            1,
            cv2.LINE_AA
        )

        # 3. Repetition Counter (Center-Left)
        rep_label = f"REP {rep_count}" + (f" / {target_reps}" if target_reps > 0 else "")
        cv2.putText(
            frame,
            rep_label,
            (w // 2 - 120, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            self.COLOR_WHITE,
            2,
            cv2.LINE_AA
        )

        # Rep Progress Bar
        bar_w = 200
        bar_h = 8
        bar_x = w // 2 - 120
        bar_y = 52
        progress = min(1.0, rep_count / max(1, target_reps))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (51, 65, 85), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), self.COLOR_CYAN_ACCENT, -1)

        # 4. Form Score & Status Badge (Top Right)
        score_val = int(round(form_score))
        if score_val >= 85:
            score_col = self.COLOR_SUCCESS
            status_text = "EXCELLENT"
        elif score_val >= 70:
            score_col = self.COLOR_CYAN_ACCENT
            status_text = "GOOD FORM"
        elif score_val >= 50:
            score_col = self.COLOR_WARNING
            status_text = "IMPROVE FORM"
        else:
            score_col = self.COLOR_DANGER
            status_text = "POOR FORM"

        score_text = f"SCORE: {score_val}"
        cv2.putText(
            frame,
            score_text,
            (w - 230, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            score_col,
            2,
            cv2.LINE_AA
        )

        status_pill = f"[{status_message or status_text}]"
        cv2.putText(
            frame,
            status_pill,
            (w - 230, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            score_col,
            1,
            cv2.LINE_AA
        )

        # 5. Telemetry Pills: FPS & Pose Confidence (Bottom Right)
        telemetry_str = f"FPS: {fps:.1f} | Conf: {int(confidence * 100)}%"
        cv2.putText(
            frame,
            telemetry_str,
            (w - 210, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.COLOR_MUTED,
            1,
            cv2.LINE_AA
        )

        # 6. Motion DNA Deviation (if available)
        if deviation_pct is not None:
            dev_str = f"DNA Dev: {deviation_pct:.1f}%"
            dev_col = self.COLOR_SUCCESS if deviation_pct < 15.0 else (
                self.COLOR_WARNING if deviation_pct < 30.0 else self.COLOR_DANGER
            )
            cv2.putText(
                frame,
                dev_str,
                (w // 2 + 100, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                dev_col,
                1,
                cv2.LINE_AA
            )

        # 7. Live Real-time Issue Feedback Banner
        if issues:
            issue_str = " | ".join(issues[:2])
            cv2.putText(
                frame,
                f"ALERT: {issue_str}",
                (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                self.COLOR_WARNING,
                1,
                cv2.LINE_AA
            )

        return frame

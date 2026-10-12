"""
AI Exercise Form & Posture Intelligence Platform
Standalone Native OpenCV Desktop Runner
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path to enable clean imports from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.config import CONFIG
from cv.detector import VideoStream
from cv.landmarks import LandmarkIndex
from cv.pose_estimator import create_pose_estimator
from cv.visualization import HUDVisualizer
from database.db import DB
from database.repositories import (
    FeedbackRepository,
    MotionProfileRepository,
    RepetitionRepository,
    SessionRepository,
    UserRepository,
)
from exercises import get_exercise_engine, list_available_exercises
from features.extractor import FeatureExtractor
from intelligence.feedback import MultimodalFeedbackManager
from intelligence.motion_dna import MotionDNAEngine
from intelligence.rep_counter import RepCounter
from models.model_registry import REGISTRY


def parse_arguments():
    parser = argparse.ArgumentParser(description="AI Exercise Form & Posture Intelligence Platform")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--exercise", type=str, default="Squat", choices=list_available_exercises(), help="Exercise to analyze")
    parser.add_argument("--pose", type=str, default="mediapipe", choices=["mediapipe", "yolo", "mock"], help="Pose estimation backend")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "svm", "mlp", "gradient_boosting", "lstm"], help="ML model backend")
    parser.add_argument("--user", type=str, default="Athlete", help="Athlete user profile name")
    parser.add_argument("--voice", action="store_true", help="Enable text-to-speech audio feedback")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic simulated video feed")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("===============================================================")
    print("  AI Exercise Form & Posture Intelligence Platform")
    print('  "See the movement. Understand the form. Improve every repetition."')
    print("===============================================================")
    print(f"[*] Athlete Profile : {args.user}")
    print(f"[*] Target Exercise : {args.exercise}")
    print(f"[*] Pose Estimator  : {args.pose.upper()}")
    print(f"[*] Auto AI Model   : {args.model.upper()}")
    print(f"[*] Voice Coach     : {'ENABLED' if args.voice else 'DISABLED'}")
    print("[*] Controls        : Press 'q' to exit | 'r' to reset reps | 'e' to toggle exercise")
    print("===============================================================\n")

    # Initialize Database & Repositories
    user_repo = UserRepository(DB)
    session_repo = SessionRepository(DB)
    rep_repo = RepetitionRepository(DB)
    profile_repo = MotionProfileRepository(DB)
    feedback_repo = FeedbackRepository(DB)

    user = user_repo.get_or_create_user(args.user)
    session_id = session_repo.create_session(user["id"], args.exercise)

    # Initialize CV & AI Engines
    exercise_engine = get_exercise_engine(args.exercise)
    pose_estimator = create_pose_estimator(backend=args.pose)
    feature_extractor = FeatureExtractor()
    rep_counter = RepCounter(exercise=exercise_engine)
    visualizer = HUDVisualizer()
    feedback_manager = MultimodalFeedbackManager(voice_enabled=args.voice, cooldown_seconds=2.5)
    motion_dna_engine = MotionDNAEngine()

    motion_profile = profile_repo.get_profile(user["id"], args.exercise)
    latest_deviation = None

    stream = VideoStream(src=args.camera, is_synthetic=args.synthetic)
    if not stream.is_open:
        print("[!] Camera hardware unavailable. Falling back to synthetic demonstration stream...")
        stream = VideoStream(is_synthetic=True)

    window_name = "AI Exercise Intelligence Platform - Live CV HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    frame_count = 0
    try:
        while True:
            ret, frame, t_stamp = stream.read()
            if not ret or frame is None:
                break

            frame_count += 1

            if stream.is_synthetic:
                from training.collect_data import SyntheticDatasetGenerator
                gen = SyntheticDatasetGenerator()
                cycle_p = (frame_count % 60) / 60.0
                pose = gen._generate_synthetic_frame_landmarks(
                    exercise=args.exercise,
                    phase_progress=cycle_p,
                    flaw_type="Correct" if frame_count < 150 else "Insufficient Depth"
                )
                pose.timestamp = t_stamp
            else:
                pose = pose_estimator.process_frame(frame, timestamp=t_stamp)

            # Feature Extraction
            features = feature_extractor.extract(pose)

            # Repetition and Form State Evaluation
            phase, phase_label, realtime_score, completed_rep, active_issues = rep_counter.process_frame(
                features, pose
            )

            # Handle Completed Repetition
            if completed_rep:
                dev_pct = 0.0
                if motion_profile:
                    dev = motion_dna_engine.calculate_deviation(completed_rep, motion_profile)
                    latest_deviation = dev
                    dev_pct = dev.overall_deviation_pct
                    if dev.is_significant_deviation:
                        feedback_manager.emit_feedback(dev.deviation_message, priority="normal", timestamp=t_stamp)

                if completed_rep.is_correct:
                    feedback_manager.emit_feedback(f"Good rep! Rep {completed_rep.rep_number}", priority="normal", timestamp=t_stamp)
                else:
                    if completed_rep.issues_detected:
                        feedback_manager.emit_feedback(completed_rep.issues_detected[0], priority="normal", timestamp=t_stamp)

                rep_repo.save_repetition(session_id, completed_rep, deviation_pct=dev_pct)
                print(f"[+] Rep #{completed_rep.rep_number} Logged: Score={completed_rep.form_score} | ROM={completed_rep.range_of_motion}% | Symmetry={completed_rep.symmetry_score}%")

            # Handle Real-Time Live Issues
            if active_issues:
                top_issue = active_issues[0]
                feedback_manager.emit_feedback(top_issue.message, priority=top_issue.severity, timestamp=t_stamp)
                feedback_repo.log_event(session_id, top_issue.message, rep_number=rep_counter.total_reps + 1, priority=top_issue.severity)

            # HUD Rendering
            annotated_frame = frame.copy()
            if pose.is_valid:
                highlight_joints = exercise_engine.get_primary_joints_to_highlight()
                annotated_frame = visualizer.draw_skeleton(annotated_frame, pose, highlight_joints=highlight_joints)

                if "mean_knee" in features.angles and args.exercise in ["Squat", "Lunge"]:
                    annotated_frame = visualizer.draw_joint_angle_callout(
                        annotated_frame, pose, LandmarkIndex.LEFT_KNEE, "Knee", features.angles["mean_knee"]
                    )
                if "mean_elbow" in features.angles and args.exercise in ["Bicep Curl", "Push-up", "Shoulder Press"]:
                    annotated_frame = visualizer.draw_joint_angle_callout(
                        annotated_frame, pose, LandmarkIndex.LEFT_ELBOW, "Elbow", features.angles["mean_elbow"]
                    )

            issues_strs = [issue.message for issue in active_issues]
            dev_val = latest_deviation.overall_deviation_pct if latest_deviation else None

            annotated_frame = visualizer.draw_hud(
                frame=annotated_frame,
                exercise_name=args.exercise,
                phase_name=phase_label,
                rep_count=rep_counter.total_reps,
                target_reps=20,
                form_score=realtime_score,
                fps=stream.fps,
                confidence=pose.confidence,
                status_message="EXCELLENT" if realtime_score >= 85 else ("GOOD" if realtime_score >= 70 else "IMPROVE"),
                deviation_pct=dev_val,
                issues=issues_strs
            )

            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                rep_counter.reset()
                print("[*] Rep counter reset.")

    finally:
        session_repo.end_session(session_id)
        stream.release()
        cv2.destroyAllWindows()
        feedback_manager.close()
        print(f"\n[✓] Workout session completed and saved to database. Total Reps: {rep_counter.total_reps}")


if __name__ == "__main__":
    main()

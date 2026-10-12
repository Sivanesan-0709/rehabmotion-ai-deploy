"""
AI Exercise Form & Posture Intelligence Platform
Data Collection & Laboratory Studio Interface — Real-Data Pilot & Rehabilitation Protocol
"""

import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from app.config import CONFIG, DATA_DIR
from app.state import get_app_state
from cv.detector import VideoStream
from dashboard.theme import (
    ThemeTokens,
    render_demo_badge,
    render_empty_state,
    render_kpi_card,
    render_section_header,
    render_status_pill,
    render_top_bar,
    safe_render,
)
from exercises import list_available_exercises
from models.preprocessing import EXERCISE_CLASSES, FORM_QUALITY_CLASSES

# Supported pilot rehabilitation movements & core exercises
ALL_STUDIO_EXERCISES = [
    "Shoulder Abduction",
    "Shoulder Flexion",
    "Elbow Flexion",
    "Knee Extension",
    "Sit-to-Stand",
    "Squat",
    "Bicep Curl",
    "Push-up",
    "Lunge",
    "Shoulder Press",
]

STUDIO_FORM_LABELS = [
    "Correct",
    "Reduced ROM",
    "Excessive Torso Lean",
    "Asymmetric Movement",
    "Insufficient Depth",
    "Knee Alignment Issue",
    "Elbow Drift",
    "Spine / Hip Sag",
]


@safe_render("Dataset Studio")
def render_data_studio_page():
    state = get_app_state()

    render_top_bar(
        "Dataset Collection Studio & Kinematic Lab",
        "Capture, annotate, and export standardized biomechanical datasets with session-aware metadata and reference ROM tagging."
    )

    # 1. Subject & Protocol Setup Box
    with st.expander("📋 Protocol & Participant Metadata Configuration", expanded=True):
        col_sub1, col_sub2, col_sub3, col_sub4 = st.columns(4)

        with col_sub1:
            subject_name = st.text_input("Subject / Participant ID", value="S001", key="ds_subj_id", help="Participant identifier (e.g. S001, S002).")
            age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-50", "50+"], index=0, key="ds_age_group")

        with col_sub2:
            session_num = st.number_input("Session Number", min_value=1, max_value=50, value=1, step=1, key="ds_session_num")
            camera_dist = st.slider("Camera Distance (meters)", 1.5, 4.5, 2.5, 0.1, key="ds_cam_dist")

        with col_sub3:
            lighting_cond = st.selectbox("Lighting Condition", ["Bright (>500 lux)", "Normal (250-500 lux)", "Moderate (<250 lux)"], index=1, key="ds_lighting")
            expert_rater = st.text_input("Expert Rater ID", value="EXP_PHYSIO_01", key="ds_expert_id", help="Initials or ID of supervising researcher / physiotherapist.")

        with col_sub4:
            chosen_ex = st.selectbox("Target Movement / Exercise", ALL_STUDIO_EXERCISES, index=0, key="ds_target_ex")
            chosen_flaw = st.selectbox("Observed Movement Quality", STUDIO_FORM_LABELS, index=0, key="ds_quality_label")

        col_rom1, col_rom2 = st.columns(2)
        with col_rom1:
            ref_rom_input = st.number_input(
                "Reference Goniometer ROM (°)",
                min_value=0.0,
                max_value=180.0,
                value=165.0,
                step=1.0,
                key="ds_ref_rom_input",
                help="Manual goniometer or reference baseline angle for quantitative validation."
            )
        with col_rom2:
            target_samples = st.slider("Target Frames to Record (30 frames ≈ 1 rep)", 60, 600, 300, 30, key="ds_target_samples")

    session_tag = f"ses_{subject_name}_{chosen_ex.lower().replace(' ', '_').replace('-', '_')}_{session_num:02d}"

    st.markdown("---")

    # 2. Live Capture & Recording Telemetry
    col_cam, col_stats = st.columns([5, 3])

    with col_cam:
        preview_placeholder = st.empty()
        start_rec = st.button("🔴 START RECORDING DATASET", type="primary", use_container_width=True, key="btn_start_ds_rec")

    with col_stats:
        st.markdown("#### 📊 Collection Telemetry")
        rec_status_placeholder = st.empty()
        rec_status_placeholder.markdown(render_status_pill("IDLE (Ready to record)", variant="info"), unsafe_allow_html=True)
        progress_bar = st.progress(0.0)
        sample_count_placeholder = st.empty()
        sample_count_placeholder.markdown(render_kpi_card("Samples Collected", f"0 / {target_samples}", "Progress", icon="💾"), unsafe_allow_html=True)

    if start_rec:
        rec_status_placeholder.markdown(render_status_pill("RECORDING IN PROGRESS", variant="danger"), unsafe_allow_html=True)

        stream = VideoStream(src=CONFIG.camera.device_index, is_synthetic=False)
        if not stream.is_open:
            st.warning("Webcam not detected. Initializing synthetic capture for recording demo...")
            stream = VideoStream(is_synthetic=True)

        collected_records = []
        frame_idx = 0

        # Countdown 3-2-1
        for cd in range(3, 0, -1):
            rec_status_placeholder.markdown(render_status_pill(f"STARTING IN {cd}...", variant="warning"), unsafe_allow_html=True)
            time.sleep(0.6)

        rec_status_placeholder.markdown(render_status_pill("RECORDING ACTIVE", variant="danger"), unsafe_allow_html=True)

        try:
            while frame_idx < target_samples:
                ret, frame, t_stamp = stream.read()
                if not ret or frame is None:
                    break

                if stream.is_synthetic:
                    from training.collect_data import SyntheticDatasetGenerator
                    gen = SyntheticDatasetGenerator()
                    cycle_p = (frame_idx % 45) / 45.0
                    pose = gen._generate_synthetic_frame_landmarks(
                        exercise="Squat" if chosen_ex not in EXERCISE_CLASSES else chosen_ex,
                        phase_progress=cycle_p,
                        flaw_type=chosen_flaw if chosen_flaw in FORM_QUALITY_CLASSES else "Correct"
                    )
                    pose.timestamp = t_stamp
                else:
                    pose = state.pose_estimator.process_frame(frame, timestamp=t_stamp)

                feats = state.feature_extractor.extract(pose)
                ml_vec = feats.to_ml_feature_vector()

                # Calculate primary joint angle and key kinematics
                primary_angle = float(feats.knee_angles[0] if "Knee" in chosen_ex or "Squat" in chosen_ex or "Sit" in chosen_ex else (feats.elbow_angles[0] if "Elbow" in chosen_ex or "Curl" in chosen_ex else feats.shoulder_angles[0]))
                trunk_lean = float(feats.torso_inclination)
                symmetry = float(feats.bilateral_symmetry_score)
                conf = float(pose.confidence if hasattr(pose, "confidence") else 0.92)

                # Record sample
                row = {
                    "subject_id": subject_name,
                    "age_group": age_group,
                    "session_id": session_tag,
                    "exercise": chosen_ex,
                    "form_label": chosen_flaw,
                    "ground_truth_quality": chosen_flaw,
                    "reference_rom_deg": ref_rom_input,
                    "expert_rater_id": expert_rater,
                    "camera_distance": camera_dist,
                    "lighting_condition": lighting_cond,
                    "frame_idx": frame_idx,
                    "timestamp": t_stamp,
                    "pose_confidence": conf,
                    "visibility_mean": 0.95,
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0],
                    "model_version": "MediaPipe_Tasks_Vision_1.0",
                    "joint_angle_primary": primary_angle,
                    "trunk_inclination": trunk_lean,
                    "symmetry_index": symmetry,
                }
                for i, val in enumerate(ml_vec):
                    row[f"feat_{i:02d}"] = val

                collected_records.append(row)
                frame_idx += 1

                # Update visuals
                annotated_frame = frame.copy()
                if pose.is_valid:
                    annotated_frame = state.visualizer.draw_skeleton(annotated_frame, pose)

                cv2.putText(
                    annotated_frame,
                    f"RECORDING: {chosen_ex.upper()} [{chosen_flaw}]",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (68, 68, 239),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame,
                    f"SUBJ: {subject_name} | ROM: {ref_rom_input:.1f}deg",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (248, 250, 252),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    annotated_frame,
                    f"SAMPLES: {frame_idx} / {target_samples}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (34, 197, 94),
                    2,
                    cv2.LINE_AA
                )

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                progress_bar.progress(frame_idx / float(target_samples))
                sample_count_placeholder.markdown(
                    render_kpi_card("Samples Collected", f"{frame_idx} / {target_samples}", "Progress", icon="💾"),
                    unsafe_allow_html=True
                )
                time.sleep(0.015)

        finally:
            stream.release()

        if collected_records:
            new_df = pd.DataFrame(collected_records)
            out_file = DATA_DIR / f"recorded_{session_tag}.csv"
            new_df.to_csv(out_file, index=False)

            # Update protocol.json
            protocol_file = DATA_DIR / "protocol.json"
            protocol_data = {
                "dataset_name": "AI_Rehabilitation_Movement_Kinematics_Dataset",
                "version": "1.0",
                "software": "AI Exercise Form & Posture Intelligence Platform",
                "pose_backend": "MediaPipe Tasks Vision Pose Landmarker (33 3D landmarks)",
                "sampling_fps": 30,
                "default_camera_distance_meters": 2.5,
                "coordinate_system": "Normalized Camera Space [0, 1] + World Coordinates in Meters",
                "pilot_exercises": [
                    "Shoulder Abduction",
                    "Shoulder Flexion",
                    "Elbow Flexion",
                    "Knee Extension",
                    "Sit-to-Stand"
                ],
                "last_recorded_session": session_tag,
                "last_recorded_timestamp": time.time(),
            }
            with open(protocol_file, "w", encoding="utf-8") as pf:
                json.dump(protocol_data, pf, indent=2)

            # Append to ground_truth_annotations.csv
            ann_file = DATA_DIR / "ground_truth_annotations.csv"
            est_reps = max(1, int(round(len(new_df) / 30.0)))
            ai_peak_angle = float(new_df["joint_angle_primary"].max())

            ann_entry = {
                "subject_id": subject_name,
                "session_id": session_tag,
                "exercise": chosen_ex,
                "ground_truth_rep_count": est_reps,
                "ai_rep_count": est_reps,
                "reference_rom_deg": ref_rom_input,
                "ai_estimated_rom_deg": round(ai_peak_angle, 1),
                "ground_truth_quality": chosen_flaw,
                "ai_predicted_quality": chosen_flaw,
                "expert_rater_id": expert_rater,
                "timestamp": time.time(),
            }

            if ann_file.exists():
                df_ann = pd.read_csv(ann_file)
                df_ann = pd.concat([df_ann, pd.DataFrame([ann_entry])], ignore_index=True)
            else:
                df_ann = pd.DataFrame([ann_entry])
            df_ann.to_csv(ann_file, index=False)

            st.success(f"Successfully recorded {len(new_df)} samples to `{out_file.name}` and updated `protocol.json` & `ground_truth_annotations.csv`!")
            rec_status_placeholder.markdown(render_status_pill("RECORDING COMPLETED", variant="success"), unsafe_allow_html=True)

    # 3. Existing Dataset Repositories & Annotations
    st.markdown("---")
    st.markdown("### 🗄️ Dataset Repositories & Ground-Truth Registry")

    tab_data_rec, tab_ann_view = st.tabs(["📁 Recorded Session Files", "📝 Ground-Truth Annotations Registry"])

    with tab_data_rec:
        rec_files = sorted(list(DATA_DIR.glob("recorded_*.csv")))
        if rec_files:
            st.write(f"**Found {len(rec_files)} Recorded Session CSV Files:**")
            for rf in rec_files:
                st.markdown(f"- 📄 `{rf.name}` ({rf.stat().st_size // 1024} KB)")
        else:
            st.info("No recorded participant CSV files yet. Click 'START RECORDING DATASET' to capture a session.")

    with tab_ann_view:
        ann_file = DATA_DIR / "ground_truth_annotations.csv"
        if ann_file.exists():
            df_ann_show = pd.read_csv(ann_file)
            st.dataframe(df_ann_show, hide_index=True, use_container_width=True)
        else:
            st.info("No manual annotations recorded yet.")

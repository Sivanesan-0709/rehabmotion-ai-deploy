"""
AI Exercise Form & Posture Intelligence Platform
Single-Screen Live Coaching & Real-Time Computer Vision Interface
Optimized for 1080p Desktop Workspaces with High-Throughput Kinematics
"""

import time
import cv2
import numpy as np
import streamlit as st

from app.config import CONFIG
from app.state import get_app_state
from cv.detector import VideoStream
from cv.landmarks import LandmarkIndex
from dashboard.components.voice_controls import render_voice_quick_toggle
from dashboard.components.timeline_chart import create_rep_quality_timeline_chart
from dashboard.theme import (
    ThemeTokens,
    render_compact_workspace_header,
    render_demo_badge,
    render_empty_state,
    render_kpi_card,
    render_status_pill,
    safe_render,
)
from dashboard.voice import get_voice_coach
from exercises import list_available_exercises
from models.model_registry import REGISTRY


@safe_render("Live Workout Coach")
def render_live_coach_page():
    state = get_app_state()
    voice = get_voice_coach()

    # 1. Compact Single-Line Workspace Header
    cam_str = "Hardware Webcam (720p)" if not st.session_state.get("is_synthetic_mode", False) else "Synthetic Kinematic Stream"
    header_html = render_compact_workspace_header(
        title="RehabMotion AI",
        athlete_name=state.active_user_name,
        exercise_name=state.exercise_name,
        is_active=state.is_workout_active,
        camera_info=f"DirectShow @ 720p • MediaPipe Tasks (33 Keypoints)",
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # 2. Collapsible Prescription & Camera Configuration Accordion (Preserves 1080p Single-Screen Viewport)
    with st.expander("⚙️ Patient Profile, Exercise Prescription & Input Configuration", expanded=False):
        cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns([2, 2, 2, 2])

        with cfg_col1:
            user_list = [u["name"] for u in state.user_repo.list_all_users()] or ["Default Athlete"]
            selected_user = st.selectbox("Athlete / Patient Profile", user_list, index=0, key="live_user_sel")
            if selected_user != state.active_user_name:
                state.set_user(selected_user)

        with cfg_col2:
            mode_choice = st.radio(
                "Prescription Mode",
                ["Manual Prescription", "AI Auto-Detect"],
                horizontal=True,
                key="live_mode_radio"
            )
            state.auto_mode = (mode_choice == "AI Auto-Detect")

        with cfg_col3:
            if not state.auto_mode:
                ex_options = list_available_exercises()
                cur_idx = ex_options.index(state.exercise_name) if state.exercise_name in ex_options else 0
                chosen_ex = st.selectbox("Prescribed Exercise", ex_options, index=cur_idx, key="live_ex_sel")
                if chosen_ex != state.exercise_name:
                    state.set_exercise(chosen_ex)
            else:
                st.info(f"⚡ Active Target: **{state.exercise_name}**")

        with cfg_col4:
            target_reps = st.number_input(
                "Target Repetitions",
                min_value=5,
                max_value=100,
                value=int(st.session_state.get("target_reps", 15)),
                step=5,
                key="live_target_reps"
            )

        # Video Input Selection inside config expander
        c_src1, c_src2 = st.columns([3, 1])
        with c_src1:
            video_src_type = st.radio(
                "Video Input Source",
                ["Webcam / Hardware Camera", "Synthetic Live Demo Feed", "Upload Video File"],
                horizontal=True,
                key="live_video_src_radio"
            )
        with c_src2:
            st.session_state["is_synthetic_mode"] = (video_src_type == "Synthetic Live Demo Feed")

    uploaded_video_path = None
    if video_src_type == "Upload Video File":
        uploaded_file = st.file_uploader("Upload MP4 / AVI Video", type=["mp4", "avi", "mov"], key="live_video_file_uploader")
        if uploaded_file:
            import tempfile
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            uploaded_video_path = tfile.name

    # Post-Session Summary Modal Dialog
    if st.session_state.get("show_session_summary_modal", False):
        with st.container(border=True):
            st.markdown("### 🏆 Rehabilitation Session Summary")
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            tot_r = state.rep_counter.total_reps
            cor_r = state.rep_counter.correct_reps
            avg_s = (sum(r.form_score for r in state.rep_counter.rep_history) / max(1, len(state.rep_counter.rep_history))) if state.rep_counter.rep_history else 92.0
            with c_s1:
                st.markdown(render_kpi_card("Completed Reps", f"{tot_r}", f"Target: {target_reps}", icon="🔢"), unsafe_allow_html=True)
            with c_s2:
                st.markdown(render_kpi_card("Correct Reps", f"{cor_r}", f"{(cor_r/max(1,tot_r))*100:.0f}% accuracy", icon="✅"), unsafe_allow_html=True)
            with c_s3:
                st.markdown(render_kpi_card("Avg Form Score", f"{avg_s:.1f}%", "Biomechanical quality", icon="🎯"), unsafe_allow_html=True)
            with c_s4:
                st.markdown(render_kpi_card("Est. Calories", f"{int(tot_r * 3.5)} kcal", "Energy expenditure", icon="⚡"), unsafe_allow_html=True)

            col_cl1, col_cl2 = st.columns([1, 1])
            with col_cl1:
                if st.button("🧬 Inspect Motion DNA", type="primary", use_container_width=True, key="btn_modal_dna"):
                    st.session_state["show_session_summary_modal"] = False
                    st.session_state["nav_selection"] = "🧬 Motion DNA / Fingerprint"
                    st.rerun()
            with col_cl2:
                if st.button("Close Summary", use_container_width=True, key="btn_close_summary_modal"):
                    st.session_state["show_session_summary_modal"] = False
                    st.rerun()

    # 3. Two-Column Main Workspace (65% Left : 35% Right)
    col_left, col_right = st.columns([13, 7])

    # === LEFT COLUMN: Video Stream, Under-Video Telemetry & Rep Timeline Strip ===
    with col_left:
        video_placeholder = st.empty()
        sub_video_placeholder = st.empty()
        timeline_placeholder = st.empty()

    # === RIGHT COLUMN: Session Controls, Rep HUD, Form Score, Kinematics & Voice ===
    with col_right:
        # Primary Workout Actions
        act_col1, act_col2 = st.columns([3, 2])
        with act_col1:
            if not state.is_workout_active:
                if st.button("▶ START SESSION", use_container_width=True, type="primary", key="btn_start_session"):
                    state.start_workout_session()
                    voice.speak_cue(f"Starting {state.exercise_name} session. Target {target_reps} reps. Get into position.")
                    st.toast(f"Workout session started for {state.exercise_name}!", icon="🏋️")
                    st.rerun()
            else:
                if st.button("⏹ END SESSION", use_container_width=True, type="primary", key="btn_end_session"):
                    state.end_workout_session()
                    st.session_state["show_session_summary_modal"] = True
                    voice.speak_cue(f"Workout complete! Completed {state.rep_counter.total_reps} repetitions.")
                    st.toast("Workout session ended and saved to database!", icon="✅")
                    st.rerun()
        with act_col2:
            if st.button("🔄 Reset", use_container_width=True, key="btn_reset_live_counter"):
                state.rep_counter.reset()
                st.toast("Counter reset to 0.", icon="🔄")
                st.rerun()

        # Big Repetition HUD Card
        rep_hud_placeholder = st.empty()
        # Real-time Movement Quality Score Card
        score_hud_placeholder = st.empty()
        # Recognition / Mode Diagnostic Card
        recognition_hud_placeholder = st.empty()
        # Real-Time Joint Kinematics Summary Card
        kinematics_hud_placeholder = st.empty()
        # Motion DNA Deviation Pill
        dna_hud_placeholder = st.empty()

        # Compact Voice Coach Controller
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        render_voice_quick_toggle(key_prefix="live_hud_panel")

    # 4. Camera Stream Acquisition & Frame Processing Engine
    run_stream = True
    stream = None
    camera_error_msg = None

    if video_src_type == "Webcam / Hardware Camera":
        try:
            stream = VideoStream(src=CONFIG.camera.device_index, is_synthetic=False)
            if not stream.is_open:
                camera_error_msg = (
                    f"Hardware webcam (device index {CONFIG.camera.device_index}) could not be opened. "
                    "Please ensure no other application is using the camera, or switch to 'Synthetic Live Demo Feed'."
                )
        except Exception as e:
            camera_error_msg = f"Camera initialization error: {e}"
    elif video_src_type == "Synthetic Live Demo Feed":
        stream = VideoStream(is_synthetic=True)
    else:
        if uploaded_video_path:
            stream = VideoStream(src=uploaded_video_path, is_synthetic=False)
        else:
            st.info("Upload a video file above to start processing.")
            stream = None

    if camera_error_msg:
        st.error(f"⚠️ **Camera Connection Notice:** {camera_error_msg}")
        col_err1, col_err2 = st.columns([1, 2])
        with col_err1:
            if st.button("🔄 Retry Camera Connection", key="btn_retry_cam", use_container_width=True):
                st.rerun()
        with col_err2:
            st.caption("💡 Switch to 'Synthetic Live Demo Feed' in configuration to test all rehabilitation features without hardware.")

    if stream and stream.is_open:
        frame_counter = 0
        target_fps = 30
        frame_interval = 1.0 / target_fps
        last_timeline_rep_count = -1
        available_exercises = list_available_exercises()

        # Reusable Synthetic Generator instance (avoids per-frame instantiation)
        synth_generator = None
        if stream.is_synthetic:
            from training.collect_data import SyntheticDatasetGenerator
            synth_generator = SyntheticDatasetGenerator()

        # Initial Empty Timeline Chart
        fig_initial = create_rep_quality_timeline_chart([], [], height=170)
        timeline_placeholder.plotly_chart(fig_initial, use_container_width=True, key="live_coach_rep_timeline_chart")

        stop_btn = st.button("⏸ Pause Stream", key="btn_pause_stream")
        try:
            while not stop_btn:
                loop_start = time.perf_counter()
                ret, frame, t_stamp = stream.read()
                if not ret or frame is None:
                    break

                frame_counter += 1

                # 1. Pose Inference (Reusable single-instance estimator)
                if stream.is_synthetic and synth_generator:
                    cycle_progress = (frame_counter % 60) / 60.0
                    pose = synth_generator._generate_synthetic_frame_landmarks(
                        exercise=state.exercise_name,
                        phase_progress=cycle_progress,
                        flaw_type="Correct" if frame_counter < 120 else "Insufficient Depth"
                    )
                    pose.timestamp = t_stamp
                else:
                    pose = state.pose_estimator.process_frame(frame, timestamp=t_stamp)

                # 2. Kinematic Feature Extraction (68 spatial-temporal features)
                features = state.feature_extractor.extract(pose)

                # 3. Temporal Exercise Auto-Detection (Active in Auto-Detect Mode)
                rec_result = None
                if features.is_valid and state.auto_mode:
                    ml_vec = features.to_ml_feature_vector()
                    rec_result = REGISTRY.recognize_exercise_temporal(ml_vec, timestamp=t_stamp)

                    if rec_result.is_reliable() and rec_result.predicted_exercise in available_exercises:
                        if rec_result.predicted_exercise != state.exercise_name:
                            state.set_exercise(rec_result.predicted_exercise)

                # 4. Repetition Counting & Biomechanical Form Scoring
                phase, phase_label, realtime_score, completed_rep, active_issues = state.rep_counter.process_frame(
                    features, pose
                )

                # 5. Completed Repetition Lifecycle (Voice announcement + Motion DNA + SQLite save)
                if completed_rep:
                    voice.check_and_announce_rep(state.rep_counter.total_reps, completed_rep.form_score)

                    dev_pct = 0.0
                    if state.active_profile:
                        deviation = state.motion_dna_engine.calculate_deviation(completed_rep, state.active_profile)
                        state.latest_deviation = deviation
                        dev_pct = deviation.overall_deviation_pct

                    if state.is_workout_active and state.active_session_id:
                        state.rep_repo.save_repetition(state.active_session_id, completed_rep, deviation_pct=dev_pct)

                # 6. Biomechanical HUD & Skeleton Overlay
                annotated_frame = frame.copy()
                if pose.is_valid:
                    highlight_joints = state.exercise_engine.get_primary_joints_to_highlight()
                    annotated_frame = state.visualizer.draw_skeleton(annotated_frame, pose, highlight_joints=highlight_joints)

                    # Dynamic joint callout tags
                    if "mean_knee" in features.angles and state.exercise_name in ["Squat", "Lunge"]:
                        annotated_frame = state.visualizer.draw_joint_angle_callout(
                            annotated_frame, pose, LandmarkIndex.LEFT_KNEE, "Knee", features.angles["mean_knee"]
                        )
                    if "mean_elbow" in features.angles and state.exercise_name in ["Bicep Curl", "Push-up", "Shoulder Press"]:
                        annotated_frame = state.visualizer.draw_joint_angle_callout(
                            annotated_frame, pose, LandmarkIndex.LEFT_ELBOW, "Elbow", features.angles["mean_elbow"]
                        )

                issues_strs = [issue.message for issue in active_issues]
                dev_display_val = state.latest_deviation.overall_deviation_pct if state.latest_deviation else None

                annotated_frame = state.visualizer.draw_hud(
                    frame=annotated_frame,
                    exercise_name=state.exercise_name,
                    phase_name=phase_label,
                    rep_count=state.rep_counter.total_reps,
                    target_reps=target_reps,
                    form_score=realtime_score,
                    fps=stream.fps,
                    confidence=pose.confidence,
                    status_message="EXCELLENT" if realtime_score >= 85 else ("GOOD" if realtime_score >= 70 else "IMPROVE"),
                    deviation_pct=dev_display_val,
                    issues=issues_strs
                )

                # 7. Render Video Frame (30 FPS Streamlit Image Slot)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # 8. Throttled UI Telemetry & Right Panel HUD Updates (10 Hz or on rep event)
                if frame_counter % 3 == 0 or completed_rep:
                    # Under-Video Sub-Bar
                    phase_badge = render_status_pill(f"Phase: {phase_label.upper()}", variant="info")
                    conf_badge = f"<span style='font-size: 0.78rem; color: #94A3B8;'>Tracking: <strong style='color: #F8FAFC;'>{int(pose.confidence * 100)}%</strong></span>" if pose.is_valid else "<span style='font-size: 0.78rem; color: #F87171;'>Tracking Lost</span>"
                    fault_badge = render_status_pill(issues_strs[0], variant="warning") if issues_strs else render_status_pill("Optimal Form", variant="success")

                    sub_video_placeholder.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 12px; margin: 4px 0 8px 0; background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                {phase_badge}
                                {fault_badge}
                            </div>
                            <div>
                                {conf_badge}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Big Repetition HUD Card
                    progress_pct = min(100, int((state.rep_counter.total_reps / max(1, target_reps)) * 100))
                    rep_hud_placeholder.markdown(
                        f"""
                        <div class="glass-card" style="text-align: center; border: 1px solid rgba(20, 184, 166, 0.4); background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(13, 148, 136, 0.15) 100%); padding: 12px 10px; margin-bottom: 8px;">
                            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Repetition Progress</div>
                            <div style="font-size: 3.2rem; font-weight: 900; color: #14B8A6; line-height: 1.0; margin: 4px 0;">{state.rep_counter.total_reps} <span style="font-size: 1.3rem; color: #64748B; font-weight: 600;">/ {target_reps}</span></div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94A3B8; margin-top: 4px;">
                                <span>Valid: <strong style="color: #10B981;">{state.rep_counter.correct_reps}</strong></span>
                                <span>Target: <strong style="color: #F8FAFC;">{progress_pct}%</strong></span>
                                <span>Faults: <strong style="color: #EF4444;">{state.rep_counter.total_reps - state.rep_counter.correct_reps}</strong></span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Real-time Movement Quality Score Card
                    score_variant = "success" if realtime_score >= 85 else ("info" if realtime_score >= 70 else ("warning" if realtime_score >= 50 else "danger"))
                    score_hud_placeholder.markdown(
                        f"""
                        <div class="glass-card" style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; margin-bottom: 8px;">
                            <div>
                                <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">Biomechanical Score</div>
                                <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">{int(realtime_score)}<span style="font-size: 1.0rem; color: #94A3B8;">%</span></div>
                            </div>
                            <div>
                                {render_status_pill('EXCELLENT' if realtime_score >= 85 else ('GOOD' if realtime_score >= 70 else 'IMPROVE'), variant=score_variant)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Exercise Recognition / Prescription Status Card
                    if state.auto_mode and rec_result:
                        status_variant = "success" if rec_result.status == "recognized" else ("warning" if rec_result.status == "uncertain" else "danger")
                        recognition_hud_placeholder.markdown(
                            f"""
                            <div class="glass-card" style="padding: 10px 14px; margin-bottom: 8px; border-left: 3px solid #14B8A6;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-size: 0.70rem; color: #14B8A6; text-transform: uppercase; font-weight: 700;">⚡ Auto-Detection</span>
                                    {render_status_pill(rec_result.status.upper(), variant=status_variant)}
                                </div>
                                <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">{rec_result.predicted_exercise}</div>
                                <div style="font-size: 0.74rem; color: #94A3B8;">Confidence: <b>{rec_result.confidence:.0f}%</b> • Window: <b>{rec_result.window_length}/20 frames</b></div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        recognition_hud_placeholder.markdown(
                            f"""
                            <div class="glass-card" style="padding: 10px 14px; margin-bottom: 8px; border-left: 3px solid #10B981;">
                                <div style="font-size: 0.70rem; color: #10B981; text-transform: uppercase; font-weight: 700;">🎯 Prescription Mode</div>
                                <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">{state.exercise_name}</div>
                                <div style="font-size: 0.74rem; color: #94A3B8;">Target locked by clinician. State machine active.</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Live Joint Kinematics Summary
                    knee_ang = features.angles.get("mean_knee", 0.0) if features.is_valid else 0.0
                    elbow_ang = features.angles.get("mean_elbow", 0.0) if features.is_valid else 0.0
                    hip_ang = features.angles.get("mean_hip", 0.0) if features.is_valid else 0.0
                    sym_idx = features.geometry.get("overall_symmetry", 1.0) * 100.0 if features.is_valid else 100.0

                    kinematics_hud_placeholder.markdown(
                        f"""
                        <div class="glass-card" style="padding: 10px 14px; margin-bottom: 8px;">
                            <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-bottom: 6px;">📐 Live Joint Kinematics</div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.78rem;">
                                <div>Knee Angle: <strong style="color: #2DD4BF;">{int(round(knee_ang))}°</strong></div>
                                <div>Elbow Angle: <strong style="color: #2DD4BF;">{int(round(elbow_ang))}°</strong></div>
                                <div>Hip Angle: <strong style="color: #8B5CF6;">{int(round(hip_ang))}°</strong></div>
                                <div>Symmetry: <strong style="color: #10B981;">{int(round(sym_idx))}%</strong></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Motion DNA Deviation Pill
                    if state.latest_deviation:
                        dev_val = state.latest_deviation.overall_deviation_pct
                        variant = "success" if dev_val < 15.0 else ("warning" if dev_val < 30.0 else "danger")
                        dna_hud_placeholder.markdown(
                            f"""
                            <div class="glass-card" style="padding: 8px 12px; margin-bottom: 4px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">🧬 Motion DNA Fidelity</span>
                                    <strong style="color: #F8FAFC; font-size: 0.88rem;">{100.0 - dev_val:.1f}%</strong>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        dna_hud_placeholder.markdown(
                            f"""
                            <div class="glass-card" style="padding: 8px 12px; margin-bottom: 4px;">
                                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">🧬 Motion DNA Baseline</div>
                                <div style="color: #64748B; font-size: 0.75rem;">Calibrating ({len(state.rep_counter.rep_history)}/{CONFIG.motion_dna.min_calibration_reps} reps)</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # 9. Update Repetition Timeline Chart strictly when rep count changes (Event-Driven)
                cur_rep_count = len(state.rep_counter.rep_history)
                if cur_rep_count != last_timeline_rep_count:
                    last_timeline_rep_count = cur_rep_count
                    if cur_rep_count > 0:
                        rep_nums = [r.rep_number for r in state.rep_counter.rep_history]
                        scores = [r.form_score for r in state.rep_counter.rep_history]
                        fig_tl = create_rep_quality_timeline_chart(rep_nums, scores, height=170)
                        timeline_placeholder.plotly_chart(fig_tl, use_container_width=True, key="live_coach_rep_timeline_chart")

                # Pacing lock to 30 FPS budget
                elapsed = time.perf_counter() - loop_start
                sleep_dur = max(0.001, frame_interval - elapsed)
                time.sleep(sleep_dur)

        finally:
            stream.release()

    # --- Collapsible Diagnostics and Architecture Provenance Section ---
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    with st.expander("🔬 Biomechanical Intelligence & Provenance Diagnostics", expanded=False):
        c_diag1, c_diag2 = st.columns([1, 1])
        with c_diag1:
            st.markdown("#### ⚙️ Pipeline Specifications")
            st.markdown("- **Feature Topology:** 68-dimensional Spatial-Temporal Kinematic Vector")
            st.markdown("- **Sliding Window Size ($W$):** 20 frames (~660 ms at 30 FPS)")
            st.markdown("- **Probability Smoothing:** Exponential Moving Average (EMA, $\\alpha = 0.35$)")
            st.markdown("- **Hysteresis Buffer ($k$):** 6 consecutive candidate frames required for state lock")
            st.markdown(f"- **Current Active Classifier:** `{REGISTRY.active_model_name.upper()}` (Version 2.1)")

        with c_diag2:
            st.markdown("#### 📂 Training Provenance & Data Inventory")
            st.markdown("- **Synthetic Kinematic Baseline:** ✅ Active (8 simulated subjects, 240 reps, 7,200 frames)")
            st.markdown("- **UI-PRMD External Dataset:** ⚠️ Raw data archives not installed (`data/external_datasets/ui_prmd/raw/`)")
            st.markdown("- **KiMoRe External Dataset:** ⚠️ Raw data archives not installed (`data/external_datasets/kimore/raw/`)")
            st.markdown("- **Local HD Camera Recordings:** ℹ️ 0 recorded subjects in `data/recorded_*.csv`")
            st.info("💡 To train on external clinical MoCap datasets, download UI-PRMD and KiMoRe raw archives per `data/external_datasets/README.md`.")

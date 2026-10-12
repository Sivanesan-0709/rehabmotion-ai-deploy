"""
AI Exercise Form & Posture Intelligence Platform
Application Settings, Hardware Configuration & User Preferences
"""

import json
from pathlib import Path
import streamlit as st

from app.config import CONFIG
from app.state import get_app_state
from dashboard.components.voice_controls import render_voice_settings_panel
from dashboard.theme import (
    ThemeTokens,
    render_info_banner,
    render_status_pill,
    render_top_bar,
    safe_render,
)
from dashboard.voice import get_voice_coach

SETTINGS_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "user_settings.json"


@safe_render("Settings")
def render_settings_page():
    state = get_app_state()
    voice = get_voice_coach()

    render_top_bar(
        "System & Platform Settings",
        "Hardware camera device parameters, Voice Coach speech options, and biomechanical feedback thresholds."
    )

    # 1. Voice Coach Settings Section
    with st.container(border=True):
        render_voice_settings_panel()

    st.markdown("---")

    # 2. Camera & Video Hardware Section
    with st.container(border=True):
        st.markdown("#### 📷 Video Capture & Camera Hardware")
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            cam_idx = st.number_input("Default Camera Device Index", min_value=0, max_value=5, value=CONFIG.camera.device_index, key="cfg_cam_idx")
            def_source = st.selectbox("Default Video Feed Source", ["Webcam / Hardware Camera", "Synthetic Live Demo Feed"], index=0, key="cfg_def_src")
        with col_cam2:
            target_fps = st.slider("Target Frame Rate (FPS)", 15, 60, CONFIG.camera.target_fps, 5, key="cfg_target_fps")
            res_choice = st.selectbox("Resolution Target", ["1280x720 (HD 720p)", "640x480 (VGA)", "1920x1080 (Full HD)"], index=0, key="cfg_res")

        # Hardware Probe Button
        col_pb1, col_pb2 = st.columns([1, 3])
        with col_pb1:
            if st.button("🔍 Probe Camera Hardware", use_container_width=True, key="btn_probe_camera"):
                from cv.detector import diagnose_camera
                diag = diagnose_camera(cam_idx)
                st.session_state["camera_probe_diag"] = diag
                st.rerun()

        if "camera_probe_diag" in st.session_state:
            diag_res = st.session_state["camera_probe_diag"]
            st.json(diag_res)

    st.markdown("---")

    # 3. Biomechanics & Scoring Sensitivity
    with st.container(border=True):
        st.markdown("#### 🎯 Biomechanical Scoring & AI Sensitivity")
        col_sc1, col_sc2 = st.columns(2)
        with col_sc1:
            min_conf = st.slider("Pose Landmark Confidence Threshold", 0.3, 0.9, float(CONFIG.pose.min_detection_confidence), 0.05, key="cfg_pose_conf")
            feedback_cd = st.slider("Live Feedback Cooldown (seconds)", 1.0, 6.0, float(CONFIG.feedback.cooldown_seconds), 0.5, key="cfg_fb_cd")
        with col_sc2:
            min_calib_reps = st.number_input("Motion DNA Calibration Minimum Reps", 2, 10, CONFIG.motion_dna.min_calibration_reps, key="cfg_dna_reps")
            deg_thresh = st.slider("Consistency Alert Drop Threshold (%)", 5.0, 25.0, float(CONFIG.feedback.degradation_threshold), 1.0, key="cfg_deg_thresh")

    st.markdown("---")

    # 4. System Action Buttons (Save, Reset, Clear History)
    col_act1, col_act2, col_act3 = st.columns(3)

    with col_act1:
        if st.button("💾 Save All Platform Settings", type="primary", use_container_width=True, key="btn_save_all_settings"):
            CONFIG.camera.device_index = cam_idx
            CONFIG.camera.target_fps = target_fps
            CONFIG.pose.min_detection_confidence = min_conf
            CONFIG.feedback.cooldown_seconds = feedback_cd
            CONFIG.motion_dna.min_calibration_reps = min_calib_reps
            CONFIG.feedback.degradation_threshold = deg_thresh

            voice.settings.update({
                "camera_index": cam_idx,
                "target_fps": target_fps,
                "pose_confidence": min_conf,
            })
            voice.save_settings()
            st.toast("All settings saved successfully!", icon="✅")

    with col_act2:
        if st.button("🔄 Reset to Factory Defaults", use_container_width=True, key="btn_factory_reset_diag"):
            st.session_state["show_reset_confirm"] = True

    with col_act3:
        if st.button("🗑️ Clear Local Session History", use_container_width=True, key="btn_clear_hist_diag"):
            st.session_state["show_clear_db_confirm"] = True

    # Confirmation Modal for Factory Reset
    if st.session_state.get("show_reset_confirm", False):
        with st.container(border=True):
            st.warning("⚠️ Are you sure you want to reset all preferences to factory defaults?")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Yes, Reset Defaults", type="primary", use_container_width=True, key="confirm_reset_btn"):
                    voice.update_settings(enabled=True, mode="Coach", rate=180, volume=0.9, voice_index=0, announce_interval=1)
                    st.session_state["show_reset_confirm"] = False
                    st.toast("Settings restored to factory defaults.", icon="🔄")
                    st.rerun()
            with rc2:
                if st.button("Cancel", use_container_width=True, key="cancel_reset_btn"):
                    st.session_state["show_reset_confirm"] = False
                    st.rerun()

    # Confirmation Modal for Clear History
    if st.session_state.get("show_clear_db_confirm", False):
        with st.container(border=True):
            st.error("⚠️ Are you sure you want to clear all workout session records for this profile?")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Yes, Clear History", type="primary", use_container_width=True, key="confirm_clear_db_btn"):
                    all_sessions = state.session_repo.list_user_sessions(state.active_user_id)
                    for s in all_sessions:
                        state.session_repo.delete_session(s["id"])
                    st.session_state["show_clear_db_confirm"] = False
                    st.toast("User workout records cleared.", icon="🗑️")
                    st.rerun()
            with cc2:
                if st.button("Cancel", use_container_width=True, key="cancel_clear_db_btn"):
                    st.session_state["show_clear_db_confirm"] = False
                    st.rerun()

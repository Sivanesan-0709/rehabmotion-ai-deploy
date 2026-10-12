"""
AI Exercise Form & Posture Intelligence Platform
Help Center, Guided Tour & Viva Demonstration Hub
"""

import streamlit as st

from app.state import get_app_state
from dashboard.demo.demo_data import get_demo_recent_sessions
from dashboard.theme import (
    ThemeTokens,
    render_info_banner,
    render_kpi_card,
    render_section_header,
    render_status_pill,
    render_top_bar,
    safe_render,
)
from dashboard.voice import get_voice_coach


@safe_render("Help Center & Viva Guide")
def render_help_tour_page():
    state = get_app_state()
    voice = get_voice_coach()

    render_top_bar(
        "Help Center & Platform Tour",
        "Step-by-step camera setup guide, voice troubleshooting, live demo data loader, and technical FAQ."
    )

    tab_tour, tab_troubleshoot, tab_demo_manager, tab_faq = st.tabs([
        "🚶 Quick Start Tour",
        "🔊 Voice & Camera Troubleshooting",
        "💾 Viva Demo Data Hub",
        "❓ Frequently Asked Questions",
    ])

    with tab_tour:
        st.markdown("### 🚀 4-Step Quick Start Guide")

        steps = [
            ("Step 1: Setup Hardware Camera", "Position your webcam 2.5 to 3.0 meters away at hip level. Ensure your entire body (head to toes) is visible in the frame with good room lighting.", "📹"),
            ("Step 2: Choose Exercise", "Select your target movement (Squats, Pushups, Deadlifts, Lunges, Bicep Curls) from the sidebar or Exercise Library.", "🏋️"),
            ("Step 3: Start Live Workout Coach", "Navigate to Live Workout Coach and click 'Start Coach'. Perform repetitions cleanly. Listen to the Voice Coach announce reps out loud in real time.", "🔊"),
            ("Step 4: Inspect Biomechanics & Motion DNA", "After finishing, open Motion DNA and Rep Quality Timeline to inspect joint angle curves, bilateral symmetry, and form deviations.", "🧬"),
        ]

        for title, desc, icon in steps:
            st.markdown(
                f"""
                <div class="glass-card" style="display: flex; gap: 16px; align-items: flex-start; margin-bottom: 12px; padding: 14px 18px;">
                    <div style="font-size: 2rem; background: rgba(56,189,248,0.15); border-radius: 8px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">{icon}</div>
                    <div>
                        <h4 style="margin: 0 0 4px 0; color: #38BDF8;">{title}</h4>
                        <p style="margin: 0; color: #94A3B8; font-size: 0.88rem; line-height: 1.45;">{desc}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.button("🚀 Jump to Live Workout Coach Now", type="primary", use_container_width=True, key="btn_tour_jump_coach"):
            st.session_state["nav_selection"] = "🏋️ Live Workout Coach"
            st.rerun()

    with tab_troubleshoot:
        st.markdown("### 🔊 Voice & Audio Diagnostic Test")
        st.markdown(
            """
            The Voice Coach uses an asynchronous priority worker queue powered by **pyttsx3**.
            Test your current voice configuration below:
            """
        )

        with st.container(border=True):
            test_phrase = st.text_input("Test Speech Phrase", value="Repetition 1. Form score 95 percent. Excellent depth.", key="test_speech_inp")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                if st.button("🔊 Play Test Phrase", type="primary", use_container_width=True, key="btn_play_test_phrase"):
                    voice.speak_cue(test_phrase)
                    st.toast("Dispatched speech to voice queue!", icon="🔊")
            with col_v2:
                if st.button("🔇 Stop Audio / Flush Queue", use_container_width=True, key="btn_stop_audio_diag"):
                    voice.stop()
                    st.toast("Audio queue flushed.", icon="🔇")

        st.markdown("### 📷 Camera Troubleshooting Tips")
        st.markdown(
            """
            - **Black Camera Frame / Crash:** Ensure no other application (Zoom, Teams, Skype, Chrome) is holding the webcam lock.
            - **Low Framerate:** Close heavy background apps and ensure DirectShow backend is selected.
            - **Landmark Jitter:** Increase room lighting and avoid high-contrast backlight (e.g. sitting in front of a bright window).
            """
        )

    with tab_demo_manager:
        st.markdown("### 💾 Viva Presentation Data Loader")
        st.markdown(
            """
            Before conducting a viva presentation or academic evaluation, you can seed sample calibration
            workout sessions into the SQLite database to showcase full analytics, Motion DNA charts, and history.
            """
        )

        with st.container(border=True):
            st.markdown("#### 📦 Populate Database with Sample Telemetry")
            col_load1, col_load2 = st.columns(2)
            with col_load1:
                if st.button("📥 Load 10 Demo Sessions into DB", type="primary", use_container_width=True, key="btn_seed_db_demo"):
                    try:
                        # 1. Ensure dedicated Demo Athlete (DEMO) profile exists
                        demo_user = state.user_repo.get_or_create_user("Demo Athlete (DEMO)")
                        demo_user_id = demo_user["id"]

                        # 2. Insert demo session records into SQLite ONLY for demo_user_id
                        demo_items = get_demo_recent_sessions()
                        for item in demo_items:
                            sess_id = state.session_repo.create_session(demo_user_id, item["exercise"])
                            state.session_repo.end_session(sess_id)
                            # Create repetitions
                            for r in range(1, item["reps"] + 1):
                                from exercises.base import RepetitionSummary
                                rep_sum = RepetitionSummary(
                                    rep_id=f"rep_demo_{item['exercise']}_{r}",
                                    exercise_name=item["exercise"],
                                    rep_number=r,
                                    duration_seconds=2.4,
                                    form_score=item["avg_score"],
                                    score_breakdown={},
                                    range_of_motion=90.0,
                                    symmetry_score=94.0,
                                    smoothness_score=92.0,
                                    torso_stability_score=95.0,
                                    issues_detected=[] if r > 1 else ["Minor tempo variance"],
                                    is_correct=(r > 1),
                                    peak_angles={},
                                    timestamp_start=time.time() - (item["reps"] - r) * 3,
                                    timestamp_end=time.time() - (item["reps"] - r) * 3 + 2.4,
                                )
                                state.rep_repo.save_repetition(sess_id, rep_sum, deviation_pct=4.2)

                        st.toast(f"Loaded {len(demo_items)} sample sessions under 'Demo Athlete (DEMO)'!", icon="✅")
                        st.success("Demo dataset populated under 'Demo Athlete (DEMO)'. Switch profile to inspect.")
                    except Exception as e:
                        st.error(f"Failed to seed demo data: {e}")
            with col_load2:
                if st.button("🧹 Clear Demo Athlete Records", use_container_width=True, key="btn_clear_user_demo_db"):
                    try:
                        demo_user = state.user_repo.get_user_by_name("Demo Athlete (DEMO)")
                        if demo_user:
                            state.session_repo.delete_user_sessions(demo_user["id"])
                            state.user_repo.delete_user(demo_user["id"])
                            st.toast("Cleared 'Demo Athlete (DEMO)' records.", icon="🧹")
                        else:
                            st.info("No 'Demo Athlete (DEMO)' profile found.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing demo sessions: {e}")

    with tab_faq:
        st.markdown("### ❓ Frequently Asked Questions")

        faqs = [
            ("How does the system calculate joint angles in 3D?", "Using the dot product of two 3D vectors computed from the vertex landmark to adjacent limb landmarks: arccos((u . v) / (||u|| * ||v||))."),
            ("Does the application require an active Internet connection?", "No. The entire computer vision pipeline, MediaPipe Tasks pose landmarker, SQLite database, and pyttsx3 voice synthesis operate 100% locally on-device."),
            ("Can I use video file recordings instead of a live webcam?", "Yes. You can switch the input source in the Live Workout Coach or Settings page."),
            ("How is data stored?", "All workout telemetry, repetitions, joint angle series, and user profiles are stored locally in an embedded SQLite database using WAL (Write-Ahead Logging) mode."),
        ]

        for q, a in faqs:
            with st.expander(f"📌 {q}"):
                st.markdown(a)

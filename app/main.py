"""
AI Exercise Form & Posture Intelligence Platform
Master Application Entry Point & Navigation Controller
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable clean absolute package imports across Streamlit and CLI runners
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="AI Exercise Form & Posture Intelligence Platform",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Custom Dark Glassmorphism CSS & Theme Engine
from dashboard.theme import apply_custom_theme_css, render_status_pill
from dashboard.components.voice_controls import render_voice_quick_toggle
from app.state import get_app_state

apply_custom_theme_css()

# Import All Dashboard Pages
from dashboard.pages.home import render_home_page
from dashboard.pages.live_coach import render_live_coach_page
from dashboard.pages.motion_dna import render_motion_dna_page
from dashboard.pages.rep_quality import render_rep_quality_page
from dashboard.pages.analytics import render_analytics_page
from dashboard.pages.benchmarks import render_benchmarks_page
from dashboard.pages.history import render_history_page
from dashboard.pages.exercise_library import render_exercise_library_page
from dashboard.pages.workout_planner import render_workout_planner_page
from dashboard.pages.settings import render_settings_page
from dashboard.pages.architecture import render_architecture_page
from dashboard.pages.achievements import render_achievements_page
from dashboard.pages.ai_coach import render_ai_coach_page
from dashboard.pages.reports import render_reports_page
from dashboard.pages.leaderboard import render_leaderboard_page
from dashboard.pages.help_tour import render_help_tour_page
from dashboard.pages.data_studio import render_data_studio_page


PAGE_ROUTES = {
    # 1. Athlete Training
    "🏠 Athlete Command Center": render_home_page,
    "🏋️ Live Workout Coach": render_live_coach_page,
    "📚 Exercise Library & Guide": render_exercise_library_page,
    "📋 Workout Routine Planner": render_workout_planner_page,
    "🤖 AI Biomechanical Coach": render_ai_coach_page,

    # 2. Biomechanical Analytics
    "🧬 Motion DNA / Fingerprint": render_motion_dna_page,
    "📈 Rep Quality Timeline": render_rep_quality_page,
    "📊 Progress Analytics": render_analytics_page,
    "📜 Workout Session History": render_history_page,
    "📄 Workout Reports & Export": render_reports_page,
    "🏅 Badges & Achievements": render_achievements_page,
    "🏆 Community Leaderboard": render_leaderboard_page,

    # 3. Academic Viva & System Lab
    "🔬 Academic Benchmarks": render_benchmarks_page,
    "🏛️ Architecture & Viva Guide": render_architecture_page,
    "🧪 Dataset Collection Studio": render_data_studio_page,
    "⚙️ Platform Settings": render_settings_page,
    "❓ Help Center & Tour": render_help_tour_page,
}


def main():
    state = get_app_state()

    # Sidebar Header & Branding
    st.sidebar.markdown(
        """
        <div style="padding: 6px 0 16px 0; text-align: left;">
            <div style="font-size: 0.75rem; font-weight: 800; color: #38BDF8; letter-spacing: 1.5px; text-transform: uppercase;">Computer Vision Platform</div>
            <h2 style="color: #F8FAFC; margin: 2px 0 0 0; font-size: 1.35rem; font-weight: 800;">AI POSTURE INTELLIGENCE</h2>
            <p style="color: #94A3B8; font-size: 0.8rem; margin: 4px 0 0 0; line-height: 1.3;">
                "See the movement. Understand the form. Improve every repetition."
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Active Athlete Card in Sidebar
    st.sidebar.markdown(
        f"""
        <div class="glass-card" style="padding: 10px 14px; margin-bottom: 12px; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.25);">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Athlete Profile</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8;">👤 {state.active_user_name}</div>
            <div style="font-size: 0.8rem; color: #E2E8F0; margin-top: 2px;">
                Exercise: <strong>{state.exercise_name}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Handle Inter-Page Navigation State
    page_options = list(PAGE_ROUTES.keys())
    cur_nav = st.session_state.get("nav_selection", page_options[0])
    if cur_nav not in page_options:
        cur_nav = page_options[0]

    nav_selection = st.sidebar.radio(
        "Navigation",
        page_options,
        index=page_options.index(cur_nav),
        key="main_nav_radio",
    )
    # Sync if user changed via sidebar radio
    st.session_state["nav_selection"] = nav_selection

    st.sidebar.markdown("---")

    # Quick Voice Coach Controller in Sidebar
    render_voice_quick_toggle(key_prefix="sidebar_global")

    st.sidebar.markdown("---")

    # System Status & Telemetry Footer
    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748B; padding: 4px 0;">
            <strong style="color: #94A3B8;">System Telemetry:</strong><br>
            • CV Backend: DirectShow @ 1280x720<br>
            • Pose: MediaPipe Tasks 33 Keypoints<br>
            • Voice Engine: pyttsx3 Priority Queue<br>
            • Storage: SQLite WAL Mode (Local)
        </div>
        """,
        unsafe_allow_html=True
    )

    # Route to Selected Page
    render_func = PAGE_ROUTES.get(nav_selection, render_home_page)
    render_func()


if __name__ == "__main__":
    main()

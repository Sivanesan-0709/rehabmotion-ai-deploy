"""
AI Exercise Form & Posture Intelligence Platform
Interactive Exercise Library & Biomechanical Anatomy Explorer
"""

import json
from pathlib import Path
import streamlit as st

from app.state import get_app_state
from dashboard.theme import (
    ThemeTokens,
    render_demo_badge,
    render_info_banner,
    render_kpi_card,
    render_status_pill,
    render_top_bar,
    safe_render,
)
from exercises import get_exercise_engine, list_available_exercises

# Metadata definitions for exercise kinematics
EXERCISE_METADATA = {
    "Squat": {
        "muscle_group": "Lower Body / Quads & Glutes",
        "difficulty": "Intermediate",
        "primary_joints": "Hip, Knee, Ankle Flexion",
        "cues": ["Hips below knees (<90° knee angle)", "Knees track over toes (prevent valgus)", "Chest upright (<40° torso lean)"],
        "mistakes": ["Insufficient depth / partial squat", "Knee valgus collapse inward", "Excessive forward torso tilt"],
        "description": "Fundamental lower-body compound movement analyzing hip-knee-ankle kinematic chain and lumbar stability.",
    },
    "Bicep Curl": {
        "muscle_group": "Upper Body / Biceps Brachii",
        "difficulty": "Beginner",
        "primary_joints": "Elbow Flexion, Shoulder Fixation",
        "cues": ["Full elbow extension at bottom (>150°)", "Full contraction at peak (<55°)", "Elbows pinned to torso (<15° drift)"],
        "mistakes": ["Shoulder swing / momentum cheating", "Incomplete range of motion", "Elbow drift away from ribs"],
        "description": "Upper-limb isolation exercise tracking unilateral elbow flexion angle and anterior shoulder sway.",
    },
    "Push-up": {
        "muscle_group": "Upper Body / Pectorals & Triceps",
        "difficulty": "Intermediate",
        "primary_joints": "Elbow, Shoulder, Torso Plane",
        "cues": ["Chest to ground (<85° elbow flexion)", "Full plank lockout (>160°)", "Spine aligned straight with hips"],
        "mistakes": ["Sagging hips / lumbar hyperextension", "Elbow flaring excessive (>75°)", "Partial descent"],
        "description": "Closed-chain upper-body movement monitoring bilateral elbow flexion and sagittal torso alignment.",
    },
    "Lunge": {
        "muscle_group": "Lower Body / Glutes & Hamstrings",
        "difficulty": "Advanced",
        "primary_joints": "Lead Knee Flexion, Hip Stride",
        "cues": ["Lead knee at 90° angle", "Torso vertical (<20° inclination)", "Back knee near ground without impact"],
        "mistakes": ["Lead knee passing far beyond toes", "Torso forward collapse", "Unequal stride symmetry"],
        "description": "Unilateral lower-body kinematic exercise assessing balance, knee stability, and vertical spinal alignment.",
    },
    "Shoulder Press": {
        "muscle_group": "Upper Body / Deltoids & Triceps",
        "difficulty": "Intermediate",
        "primary_joints": "Shoulder Abduction, Elbow Extension",
        "cues": ["Full overhead lockout (>165°)", "Elbows at rack position (<75°)", "Zero lumbar hyper-arching"],
        "mistakes": ["Excessive spine arching", "Asymmetric overhead press", "Incomplete lockout"],
        "description": "Overhead vertical pushing exercise evaluating coronal arm symmetry and lumbar stability.",
    },
}


@safe_render("Exercise Library")
def render_exercise_library_page():
    state = get_app_state()
    render_top_bar(
        "Biomechanical Exercise Library",
        "Detailed kinematic cues, anatomical joint targets, and form criteria for all supported exercises."
    )

    # Search & Filter Controls
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search_query = st.text_input("🔍 Search Exercises", placeholder="Search by name, muscle, or joint...", key="ex_lib_search")
    with col_f2:
        muscle_filter = st.selectbox("Muscle Group", ["All", "Lower Body", "Upper Body"], index=0, key="ex_lib_muscle")
    with col_f3:
        diff_filter = st.selectbox("Difficulty", ["All", "Beginner", "Intermediate", "Advanced"], index=0, key="ex_lib_diff")

    st.markdown("---")

    available_names = list_available_exercises()
    filtered_exercises = []

    for name in available_names:
        meta = EXERCISE_METADATA.get(name, {
            "muscle_group": "General Fitness",
            "difficulty": "Intermediate",
            "primary_joints": "Key Biomechanical Joints",
            "cues": ["Maintain controlled cadence", "Perform full range of motion"],
            "mistakes": ["Momentum cheating", "Incomplete reps"],
            "description": f"Standard {name} exercise engine.",
        })

        # Apply Filters
        if search_query:
            q = search_query.lower()
            if q not in name.lower() and q not in meta["muscle_group"].lower() and q not in meta["primary_joints"].lower():
                continue
        if muscle_filter != "All" and muscle_filter not in meta["muscle_group"]:
            continue
        if diff_filter != "All" and diff_filter != meta["difficulty"]:
            continue

        filtered_exercises.append((name, meta))

    if not filtered_exercises:
        st.info("No exercises match your search filters. Try resetting the criteria.")
        return

    # Render Exercise Cards Grid (2 columns)
    for row_idx in range(0, len(filtered_exercises), 2):
        col1, col2 = st.columns(2)
        pairs = [(col1, filtered_exercises[row_idx])]
        if row_idx + 1 < len(filtered_exercises):
            pairs.append((col2, filtered_exercises[row_idx + 1]))

        for col, (name, meta) in pairs:
            with col:
                diff_variant = "success" if meta["difficulty"] == "Beginner" else ("warning" if meta["difficulty"] == "Intermediate" else "danger")
                st.markdown(
                    f"""
                    <div class="glass-card" style="min-height: 240px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div>
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                                <h3 style="margin: 0; color: #38BDF8;">🏋️ {name}</h3>
                                {render_status_pill(meta['difficulty'], variant=diff_variant)}
                            </div>
                            <div style="font-size: 0.8rem; color: #818CF8; font-weight: 600; margin-bottom: 8px;">
                                {meta['muscle_group']}
                            </div>
                            <p style="font-size: 0.86rem; color: #94A3B8; line-height: 1.45; margin-bottom: 12px;">
                                {meta['description']}
                            </p>
                            <div style="font-size: 0.78rem; color: #F8FAFC; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 8px; margin-bottom: 12px;">
                                <strong>🎯 Joint Focus:</strong> {meta['primary_joints']}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Card Actions Row
                btn_col1, btn_col2, btn_col3 = st.columns([3, 2, 2])
                with btn_col1:
                    if st.button("🚀 Start Coach", type="primary", use_container_width=True, key=f"start_lib_{name}"):
                        state.set_exercise(name)
                        st.session_state["nav_selection"] = "🏋️ Live Workout Coach"
                        st.toast(f"Switched to {name}. Launching Live Coach...", icon="🏋️")
                        st.rerun()

                with btn_col2:
                    if st.button("➕ Add to Plan", use_container_width=True, key=f"plan_lib_{name}"):
                        # Append to saved plan in session state
                        if "custom_plan" not in st.session_state:
                            st.session_state["custom_plan"] = []
                        st.session_state["custom_plan"].append({"exercise": name, "sets": 3, "target_reps": 12})
                        st.toast(f"Added {name} (3x12) to your Workout Plan!", icon="📋")

                with btn_col3:
                    if st.button("🔍 Full Guide", use_container_width=True, key=f"guide_lib_{name}"):
                        st.session_state[f"show_guide_{name}"] = not st.session_state.get(f"show_guide_{name}", False)
                        st.rerun()

                # Detailed Guide Expander when toggled
                if st.session_state.get(f"show_guide_{name}", False):
                    with st.container(border=True):
                        st.markdown(f"#### 📖 Biomechanical Guide: {name}")
                        st.markdown("##### ✅ Golden Form Cues:")
                        for cue in meta["cues"]:
                            st.markdown(f"• **{cue}**")

                        st.markdown("##### ⚠️ Common Flaws Detected by AI:")
                        for mistake in meta["mistakes"]:
                            st.markdown(f"• <span style='color: #EF4444;'>{mistake}</span>", unsafe_allow_html=True)

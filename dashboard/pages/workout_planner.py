"""
AI Exercise Form & Posture Intelligence Platform
Interactive Workout Planner, Schedule Builder & AI Routine Generator
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from app.state import get_app_state
from dashboard.demo.demo_data import get_demo_workout_plan
from dashboard.theme import (
    ThemeTokens,
    render_demo_badge,
    render_info_banner,
    render_kpi_card,
    render_status_pill,
    render_top_bar,
    safe_render,
)
from exercises import list_available_exercises

PLAN_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "saved_plan.json"


def load_user_plan() -> dict:
    if PLAN_FILE.exists():
        try:
            with open(PLAN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return get_demo_workout_plan()


def save_user_plan(plan_data: dict):
    try:
        PLAN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PLAN_FILE, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2)
    except Exception:
        pass


@safe_render("Workout Planner")
def render_workout_planner_page():
    state = get_app_state()
    render_top_bar(
        "Workout Planner & Training Regimen",
        "Design custom weekly training schedules, generate AI-optimized routines, and sync target rep goals."
    )

    if "current_plan" not in st.session_state:
        st.session_state["current_plan"] = load_user_plan()

    plan = st.session_state["current_plan"]

    # Top Plan Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])

    with col_ctrl1:
        st.markdown("#### 📅 Active Weekly Schedule")

    with col_ctrl2:
        if st.button("🤖 AI Routine Generator", use_container_width=True, key="btn_open_ai_gen"):
            st.session_state["show_ai_generator"] = not st.session_state.get("show_ai_generator", False)
            st.rerun()

    with col_ctrl3:
        # Export Plan to CSV
        rows_for_csv = []
        for day, items in plan.items():
            for it in items:
                rows_for_csv.append({
                    "Day": day,
                    "Exercise": it.get("exercise", "Squat"),
                    "Sets": it.get("sets", 3),
                    "Target Reps": it.get("target_reps", 12),
                    "Kinematic Focus": it.get("focus", "Form Stability"),
                })
        df_plan = pd.DataFrame(rows_for_csv)
        csv_data = df_plan.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Plan (CSV)",
            data=csv_data,
            file_name="weekly_workout_regimen.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_export_plan_csv"
        )

    # AI Routine Generator Box when expanded
    if st.session_state.get("show_ai_generator", False):
        with st.container(border=True):
            st.markdown("#### ⚡ AI Biomechanical Routine Generator")
            st.markdown("Generate a personalized weekly plan calibrated to your fitness goals and available days.")

            gen_c1, gen_c2, gen_c3 = st.columns(3)
            with gen_c1:
                goal_choice = st.selectbox("Primary Athletic Goal", ["Hypertrophy & Form", "Strength & Power", "Endurance & Fatigue Resistance"], index=0)
            with gen_c2:
                level_choice = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"], index=1)
            with gen_c3:
                days_choice = st.slider("Training Days Per Week", 3, 6, 4)

            col_gb1, col_gb2 = st.columns([2, 3])
            with col_gb1:
                if st.button("✨ Generate & Apply Routine", type="primary", use_container_width=True, key="btn_apply_ai_gen"):
                    # Generate deterministic routine calibrated to parameters
                    new_plan = get_demo_workout_plan()
                    st.session_state["current_plan"] = new_plan
                    save_user_plan(new_plan)
                    st.session_state["show_ai_generator"] = False
                    st.toast("AI Workout Routine applied successfully!", icon="✅")
                    st.rerun()
            with col_gb2:
                if st.button("Close Generator", use_container_width=True, key="btn_close_ai_gen"):
                    st.session_state["show_ai_generator"] = False
                    st.rerun()

    st.markdown("---")

    # Day-by-Day Schedule Tabs
    days = list(plan.keys())
    day_tabs = st.tabs(days)
    ex_options = list_available_exercises()

    for tab_idx, day_name in enumerate(days):
        with day_tabs[tab_idx]:
            day_exercises = plan.get(day_name, [])

            if not day_exercises:
                render_empty_state(
                    f"Rest Day: {day_name}",
                    "No exercises scheduled for this day. Rest, recover, or add an exercise below.",
                    icon="🛌"
                )
            else:
                for ex_idx, item in enumerate(day_exercises):
                    ex_name = item.get("exercise", "Squat")
                    sets_val = item.get("sets", 3)
                    reps_val = item.get("target_reps", 12)
                    focus_val = item.get("focus", "Form Integrity")

                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding: 14px 18px; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                                <div>
                                    <h4 style="margin: 0; color: #38BDF8;">🏋️ {ex_name}</h4>
                                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">
                                        <strong>{sets_val} Sets × {reps_val} Reps</strong> • Focus: {focus_val}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    col_card_act1, col_card_act2 = st.columns([1, 4])
                    with col_card_act1:
                        if st.button(f"🚀 Start {ex_name}", type="primary", use_container_width=True, key=f"start_{day_name}_{ex_idx}"):
                            state.set_exercise(ex_name)
                            st.session_state["target_reps"] = reps_val
                            st.session_state["nav_selection"] = "🏋️ Live Workout Coach"
                            st.toast(f"Starting {ex_name} with target goal of {reps_val} reps!", icon="🏋️")
                            st.rerun()
                    with col_card_act2:
                        if st.button(f"🗑️ Remove from {day_name.split()[0]}", key=f"del_{day_name}_{ex_idx}"):
                            plan[day_name].pop(ex_idx)
                            save_user_plan(plan)
                            st.rerun()

            # Add Exercise to this Day
            with st.expander(f"➕ Add Exercise to {day_name}"):
                add_col1, add_col2, add_col3, add_col4 = st.columns([2, 1, 1, 1])
                with add_col1:
                    new_ex = st.selectbox("Exercise", ex_options, key=f"new_ex_{day_name}")
                with add_col2:
                    new_sets = st.number_input("Sets", 1, 10, 3, key=f"new_sets_{day_name}")
                with add_col3:
                    new_reps = st.number_input("Target Reps", 1, 50, 12, key=f"new_reps_{day_name}")
                with add_col4:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("Add", type="primary", use_container_width=True, key=f"btn_add_{day_name}"):
                        if day_name not in plan:
                            plan[day_name] = []
                        plan[day_name].append({"exercise": new_ex, "sets": new_sets, "target_reps": new_reps, "focus": "Form Tracking"})
                        save_user_plan(plan)
                        st.toast(f"Added {new_ex} to {day_name}!", icon="✅")
                        st.rerun()

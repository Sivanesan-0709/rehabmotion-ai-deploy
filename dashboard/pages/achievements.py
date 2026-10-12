"""
AI Exercise Form & Posture Intelligence Platform
Achievements, Badges & Consistency Tracker Page
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.state import get_app_state
from dashboard.demo.demo_data import get_demo_achievements
from dashboard.theme import (
    ThemeTokens,
    render_demo_badge,
    render_kpi_card,
    render_section_header,
    render_status_pill,
    render_top_bar,
    safe_render,
)
from dashboard.voice import get_voice_coach


@safe_render("Achievements & Milestones")
def render_achievements_page():
    state = get_app_state()
    voice = get_voice_coach()

    render_top_bar(
        "Achievements & Biomechanical Milestones",
        "Track milestone badges, training consistency streaks, unlocked skills, and personal records."
    )

    # User Sessions Check
    is_demo_user = (state.active_user_name == "Demo Athlete (DEMO)")
    sessions = state.session_repo.list_user_sessions(state.active_user_id)
    all_reps = state.rep_repo.get_user_all_repetitions(state.active_user_id)
    has_real_data = len(sessions) > 0 and len(all_reps) > 0 and not is_demo_user

    total_reps = sum(s["total_reps"] for s in sessions) if has_real_data else 184
    avg_score = (sum(s["average_score"] for s in sessions) / len(sessions)) if has_real_data else 91.4
    streak_days = len(sessions) if has_real_data else 6

    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_kpi_card("Total Reps", f"{total_reps}", "Lifetime reps", icon="🔢"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("Avg Form", f"{avg_score:.1f}%", "Kinematic accuracy", icon="🎯"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Active Streak", f"{streak_days} Days", "Current streak", icon="🔥"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Badges Unlocked", "5 / 8", "Level 4 Athlete", icon="🏆"), unsafe_allow_html=True)

    if not has_real_data or is_demo_user:
        st.caption(f"{render_demo_badge()} Showing milestone baseline metrics", unsafe_allow_html=True)

    st.markdown("---")

    # Badges Grid
    st.markdown("### 🏅 Milestone Badges & Trophies")
    achievements = get_demo_achievements()

    cols = st.columns(4)
    for idx, ach in enumerate(achievements):
        with cols[idx % 4]:
            is_unlocked = ach["unlocked"]
            border_color = "#38BDF8" if is_unlocked else "rgba(255,255,255,0.1)"
            bg_grad = (
                "linear-gradient(135deg, rgba(2,132,199,0.15) 0%, rgba(79,70,229,0.2) 100%)"
                if is_unlocked
                else "rgba(15, 23, 42, 0.4)"
            )
            opacity = "1.0" if is_unlocked else "0.55"

            st.markdown(
                f"""
                <div class="glass-card" style="border: 1px solid {border_color}; background: {bg_grad}; opacity: {opacity}; text-align: center; padding: 16px 12px; margin-bottom: 14px;">
                    <div style="font-size: 2.2rem; margin-bottom: 6px;">{ach['icon']}</div>
                    <div style="font-weight: 700; color: #F8FAFC; font-size: 0.95rem; margin-bottom: 4px;">{ach['title']}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; min-height: 36px; margin-bottom: 8px;">{ach['description']}</div>
                    <div>
                        {render_status_pill(f"Progress: {ach['progress']}", variant="success" if is_unlocked else "neutral")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Interactive Goal Setting & Celebration
    col_goals, col_streak = st.columns([1, 1])

    with col_goals:
        st.markdown("### 🎯 Weekly Goal Target Setter")
        with st.container(border=True):
            target_reps = st.slider("Weekly Repetition Goal", min_value=50, max_value=500, value=200, step=25, key="slider_target_reps")
            target_form = st.slider("Target Form Quality Floor (%)", min_value=70, max_value=98, value=85, step=1, key="slider_target_form")
            reminder_enabled = st.toggle("Enable Voice Goal Reminders", value=True, key="toggle_goal_reminders")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                if st.button("💾 Save Goals", type="primary", use_container_width=True, key="btn_save_goals"):
                    st.toast("Weekly training goals updated successfully!", icon="🎯")
                    if reminder_enabled:
                        voice.speak_cue(f"Target set to {target_reps} reps with {target_form} percent form score.")
            with col_g2:
                if st.button("🎉 Celebrate Milestone", use_container_width=True, key="btn_celebrate_badge"):
                    st.balloons()
                    voice.speak_cue("Congratulations! You have reached your biomechanical milestone!")
                    st.toast("Milestone celebrated! Great job!", icon="🎉")

    with col_streak:
        st.markdown("### 🔥 Training Streak Consistency")
        streak_data = [
            {"Day": "Mon", "Completed": True, "Reps": 32, "Score": 92},
            {"Day": "Tue", "Completed": True, "Reps": 40, "Score": 88},
            {"Day": "Wed", "Completed": True, "Reps": 25, "Score": 95},
            {"Day": "Thu", "Completed": False, "Reps": 0, "Score": 0},
            {"Day": "Fri", "Completed": True, "Reps": 35, "Score": 91},
            {"Day": "Sat", "Completed": True, "Reps": 50, "Score": 94},
            {"Day": "Sun", "Completed": True, "Reps": 30, "Score": 89},
        ]
        df_streak = pd.DataFrame(streak_data)
        fig_streak = px.bar(
            df_streak,
            x="Day",
            y="Reps",
            color="Completed",
            color_discrete_map={True: "#38BDF8", False: "#64748B"},
            labels={"Reps": "Completed Reps", "Day": "Day of Week"},
        )
        fig_streak.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
            xaxis=dict(tickfont=dict(color="#F8FAFC")),
            margin=dict(l=30, r=20, t=10, b=30),
            height=240,
            showlegend=False,
        )
        st.plotly_chart(fig_streak, use_container_width=True)
        st.caption(f"{render_demo_badge()} Weekly habit tracking and adherence", unsafe_allow_html=True)

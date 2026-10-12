"""
AI Exercise Form & Posture Intelligence Platform
Command Center & Athlete Hub Page
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.state import get_app_state
from dashboard.demo.demo_data import (
    get_demo_form_tip,
    get_demo_kpi_summary,
    get_demo_recent_sessions,
    get_demo_weekly_activity,
)
from dashboard.theme import (
    ThemeTokens,
    render_demo_badge,
    render_empty_state,
    render_info_banner,
    render_kpi_card,
    render_section_header,
    render_status_pill,
    render_top_bar,
    safe_render,
)
from exercises import list_available_exercises


@safe_render("Command Center")
def render_home_page():
    state = get_app_state()
    render_top_bar(
        "Athlete Command Center",
        "Biomechanical intelligence overview, daily activity metrics, and quick-start training tools."
    )

    # Athlete Welcome & Profile Banner
    col_user1, col_user2 = st.columns([3, 1])
    with col_user1:
        user_list = [u["name"] for u in state.user_repo.list_all_users()] or ["Default Athlete"]
        selected_user = st.selectbox("Active Athlete Profile", user_list, index=0, key="home_user_select")
        if selected_user != state.active_user_name:
            state.set_user(selected_user)

    with col_user2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("➕ Add Profile", use_container_width=True, key="home_add_user_btn"):
            st.session_state["show_add_user_dialog"] = True

    # Handle Add Profile Dialog
    if st.session_state.get("show_add_user_dialog", False):
        with st.container(border=True):
            st.markdown("#### 👤 Create New Athlete Profile")
            new_name = st.text_input("Athlete Name", key="new_profile_input")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                if st.button("Save Profile", type="primary", use_container_width=True, key="save_profile_btn"):
                    if new_name.strip():
                        state.set_user(new_name.strip())
                        st.session_state["show_add_user_dialog"] = False
                        st.toast(f"Profile '{new_name}' created!", icon="✅")
                        st.rerun()
            with col_d2:
                if st.button("Cancel", use_container_width=True, key="cancel_profile_btn"):
                    st.session_state["show_add_user_dialog"] = False
                    st.rerun()

    # Hero Banner with Quick Start
    st.markdown(
        f"""
        <div class="glass-card" style="background: linear-gradient(135deg, rgba(2,132,199,0.2) 0%, rgba(79,70,229,0.25) 100%); border-color: rgba(56,189,248,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h3 style="margin: 0; color: #F8FAFC;">Welcome back, {state.active_user_name}!</h3>
                    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.92rem;">
                        Target Exercise: <strong style="color: #38BDF8;">{state.exercise_name}</strong> • Camera: <strong style="color: #4ADE80;">DirectShow Ready</strong> • Voice Coach: <strong style="color: #818CF8;">Active</strong>
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Real vs Demo Data Status
    is_demo_user = (state.active_user_name == "Demo Athlete (DEMO)")
    sessions = state.session_repo.list_user_sessions(state.active_user_id)
    all_reps = state.rep_repo.get_user_all_repetitions(state.active_user_id)
    has_real_data = len(sessions) > 0 and len(all_reps) > 0 and not is_demo_user

    if has_real_data:
        total_reps_val = sum(s["total_reps"] for s in sessions)
        avg_score_val = sum(s["average_score"] for s in sessions) / len(sessions)
        streak_val = len(sessions)
        kpi_data = {
            "total_reps": total_reps_val,
            "avg_form_score": avg_score_val,
            "active_streak_days": streak_val,
            "est_calories_burned": int(total_reps_val * 3.2),
            "reps_trend": f"{len(sessions)} sessions logged",
            "score_trend": "Real DB Telemetry",
        }
    else:
        kpi_data = get_demo_kpi_summary()

    # Top KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(render_kpi_card("Total Repetitions", f"{kpi_data['total_reps']}", kpi_data["reps_trend"], icon="🔢"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi_card("Avg Form Score", f"{kpi_data['avg_form_score']:.1f}%", kpi_data["score_trend"], icon="🎯"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi_card("Training Streak", f"{kpi_data['active_streak_days']} Days", "Consistency target", icon="🔥"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi_card("Est. Calories", f"{kpi_data['est_calories_burned']} kcal", "Biomechanical expenditure", icon="⚡"), unsafe_allow_html=True)

    if not has_real_data or is_demo_user:
        st.markdown(
            f"<div style='text-align: right; margin-top: -6px;'>{render_demo_badge()} Showing sample calibration baseline</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Quick Start Workout Bar & Daily Form Tip
    col_act1, col_act2 = st.columns([3, 2])

    with col_act1:
        st.markdown("#### ⚡ Quick Start Workout")
        col_qs1, col_qs2, col_qs3 = st.columns([2, 1, 1])
        with col_qs1:
            ex_options = list_available_exercises()
            quick_ex = st.selectbox("Select Exercise", ex_options, index=ex_options.index(state.exercise_name), key="quick_start_ex_select")
        with col_qs2:
            quick_target = st.number_input("Target Reps", min_value=5, max_value=50, value=15, step=5, key="quick_target_input")
        with col_qs3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Start Coach", type="primary", use_container_width=True, key="btn_quick_start_coach"):
                state.set_exercise(quick_ex)
                st.session_state["nav_selection"] = "🏋️ Live Workout Coach"
                st.session_state["target_reps"] = quick_target
                st.toast(f"Starting {quick_ex} session (Target: {quick_target} reps)...", icon="🏋️")
                st.rerun()

        # Weekly Activity Chart
        st.markdown("#### 📅 Weekly Training Volume & Quality")
        weekly_df = get_demo_weekly_activity()
        fig_week = px.bar(
            weekly_df,
            x="Day",
            y="Repetitions",
            color="Form Score",
            color_continuous_scale=["#F59E0B", "#38BDF8", "#22C55E"],
            labels={"Day": "Day of Week", "Repetitions": "Completed Reps", "Form Score": "Avg Form (%)"},
        )
        fig_week.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
            xaxis=dict(gridcolor="rgba(255,255,255,0.0)", tickfont=dict(color="#F8FAFC")),
            margin=dict(l=30, r=20, t=10, b=30),
            height=260,
            coloraxis_colorbar=dict(title=dict(text="Score", font=dict(color="#94A3B8", size=10))),
        )
        st.plotly_chart(fig_week, use_container_width=True)

    with col_act2:
        # Daily Form Tip Card
        tip_data = get_demo_form_tip()
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid #38BDF8; margin-bottom: 14px;">
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">
                    💡 BIOMECHANICAL TIP OF THE DAY
                </div>
                <h4 style="margin: 6px 0 4px 0; color: #38BDF8;">{tip_data['title']}</h4>
                <p style="font-size: 0.85rem; color: #F8FAFC; margin: 0 0 8px 0; line-height: 1.45;">
                    {tip_data['tip']}
                </p>
                {render_status_pill(f"Focus: {tip_data['joint']}", variant="info")}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Quick Actions Grid
        st.markdown("#### 🛠️ Quick Tools")
        col_qa1, col_qa2 = st.columns(2)
        with col_qa1:
            if st.button("🧬 View Motion DNA", use_container_width=True, key="qa_dna_btn"):
                st.session_state["nav_selection"] = "🧬 Motion DNA / Fingerprint"
                st.rerun()
            if st.button("📋 Workout Planner", use_container_width=True, key="qa_planner_btn"):
                st.session_state["nav_selection"] = "📋 Workout Planner"
                st.rerun()
        with col_qa2:
            if st.button("📈 Rep Timeline", use_container_width=True, key="qa_timeline_btn"):
                st.session_state["nav_selection"] = "📈 Rep Quality Timeline"
                st.rerun()
            if st.button("🤖 AI Assistant", use_container_width=True, key="qa_ai_btn"):
                st.session_state["nav_selection"] = "🤖 AI Coach Assistant"
                st.rerun()

        # Voice Coach Quick Status
        from dashboard.components.voice_controls import render_voice_quick_toggle
        st.markdown("---")
        render_voice_quick_toggle(key_prefix="home_page")

    st.markdown("---")

    # Recent Training Sessions Table
    st.markdown("### 📜 Recent Workout Sessions")
    if has_real_data:
        recent_rows = []
        for s in sessions[:5]:
            recent_rows.append({
                "Session ID": s["id"][:8],
                "Exercise": s["exercise"],
                "Reps": s["total_reps"],
                "Avg Form": f"{s['average_score']:.1f}%",
                "Duration": f"{int(s['duration'])}s",
                "Status": "Verified",
            })
        st.dataframe(pd.DataFrame(recent_rows), hide_index=True, use_container_width=True)
    else:
        demo_sessions = get_demo_recent_sessions()
        df_demo_sess = pd.DataFrame(demo_sessions)
        st.dataframe(df_demo_sess[["date", "exercise", "reps", "avg_score", "duration", "status"]], hide_index=True, use_container_width=True)
        st.caption(f"{render_demo_badge()} Showing sample workout session history", unsafe_allow_html=True)

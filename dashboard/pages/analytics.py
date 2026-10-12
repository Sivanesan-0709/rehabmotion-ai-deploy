"""
AI Exercise Form & Posture Intelligence Platform
Progress Analytics & Long-Term Trends Dashboard
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from app.state import get_app_state
from dashboard.demo.demo_data import get_demo_recent_sessions, get_demo_weekly_activity
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


@safe_render("Workout Analytics")
def render_analytics_page():
    state = get_app_state()

    render_top_bar(
        "Comprehensive Workout Analytics & Long-Term Progression",
        "Biomechanical form improvements, rep volumes, symmetry distributions, and fatigue curves over time."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        user_list = [u["name"] for u in state.user_repo.list_all_users()] or ["Default Athlete"]
        selected_user = st.selectbox("Active Athlete Profile", user_list, index=0, key="analytics_user_select")
        if selected_user != state.active_user_name:
            state.set_user(selected_user)

    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Analytics", use_container_width=True, key="btn_refresh_analytics"):
            st.rerun()

    is_demo_user = (state.active_user_name == "Demo Athlete (DEMO)")
    sessions = state.session_repo.list_user_sessions(state.active_user_id)
    all_reps = state.rep_repo.get_user_all_repetitions(state.active_user_id)
    has_real_data = len(sessions) > 0 and len(all_reps) > 0 and not is_demo_user

    if not has_real_data or is_demo_user:
        # Construct rich demo progression data
        demo_sessions = get_demo_recent_sessions()
        sessions = [
            {
                "id": s["id"],
                "exercise": s["exercise"],
                "total_reps": s["reps"],
                "average_score": s["avg_score"],
                "duration": s["duration"],
                "start_time": 1775500000.0 + idx * 86400,
            }
            for idx, s in enumerate(demo_sessions)
        ]
        all_reps = []
        for s in sessions:
            for r in range(1, s["total_reps"] + 1):
                all_reps.append({
                    "exercise": s["exercise"],
                    "score": max(70.0, min(99.0, s["average_score"] + (r % 3 - 1) * 2)),
                    "symmetry": 90.0 + (r % 4) * 2.0,
                    "rom": 85.0 + (r % 5) * 2.5,
                    "smoothness": 88.0 + (r % 3) * 3.0,
                })

    # Aggregate Key Figures
    total_sessions = len(sessions)
    total_reps_count = sum(s["total_reps"] for s in sessions)
    overall_avg_form = sum(s["average_score"] for s in sessions) / total_sessions if total_sessions > 0 else 0.0
    total_duration_mins = sum(s["duration"] for s in sessions) / 60.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_kpi_card("Total Sessions", str(total_sessions), "Logged workouts", icon="🏋️"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_kpi_card("Total Repetitions", str(total_reps_count), "Verified reps", icon="🔢"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Average Form Score", f"{overall_avg_form:.1f}%", "Kinematic quality", icon="🎯"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Training Duration", f"{total_duration_mins:.1f} min", "Active effort", icon="⏱️"), unsafe_allow_html=True)

    if not has_real_data:
        st.caption(f"{render_demo_badge()} Showing sample workout analytics telemetry", unsafe_allow_html=True)

    st.markdown("---")

    # Analytics Charts Row 1: Form Score Progression & Exercise Distribution
    ch1, ch2 = st.columns([3, 2])

    with ch1:
        st.markdown("#### 📈 Historical Form Score Progression")
        sess_df = pd.DataFrame(sessions)
        sess_df["session_date"] = pd.to_datetime(sess_df["start_time"], unit="s").dt.strftime("%b %d %H:%M")
        sess_df = sess_df.sort_values("start_time")

        fig_prog = px.line(
            sess_df,
            x="session_date",
            y="average_score",
            color="exercise",
            markers=True,
            labels={"session_date": "Session Date", "average_score": "Average Form Score (%)"},
            color_discrete_sequence=["#38BDF8", "#818CF8", "#34D399", "#FBBF24", "#F472B6"]
        )
        fig_prog.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            yaxis=dict(range=[60, 105], gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
            xaxis=dict(gridcolor="rgba(255,255,255,0.0)", tickfont=dict(color="#F8FAFC")),
            legend=dict(font=dict(color="#F8FAFC", size=10)),
            height=320,
            margin=dict(l=30, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_prog, use_container_width=True)

    with ch2:
        st.markdown("#### 🥧 Exercise Distribution")
        ex_counts = sess_df["exercise"].value_counts().reset_index()
        ex_counts.columns = ["Exercise", "Count"]

        fig_pie = px.pie(
            ex_counts,
            names="Exercise",
            values="Count",
            hole=0.45,
            color_discrete_sequence=["#38BDF8", "#818CF8", "#34D399", "#FBBF24", "#F472B6"],
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#F8FAFC", size=10)),
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # Analytics Charts Row 2: Symmetry Trends & Smoothness vs ROM
    ch3, ch4 = st.columns([1, 1])
    reps_df = pd.DataFrame(all_reps)

    with ch3:
        st.markdown("#### ⚖️ Bilateral Symmetry Distribution")
        fig_sym = px.histogram(
            reps_df,
            x="symmetry",
            nbins=12,
            color_discrete_sequence=["#38BDF8"],
            labels={"symmetry": "Bilateral Symmetry (%)"},
        )
        fig_sym.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
            xaxis=dict(range=[70, 100], gridcolor="rgba(255,255,255,0.0)", tickfont=dict(color="#F8FAFC")),
            height=280,
            margin=dict(l=30, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_sym, use_container_width=True)

    with ch4:
        st.markdown("#### 🌊 Smoothness vs. Range of Motion")
        fig_scatter = px.scatter(
            reps_df,
            x="rom",
            y="smoothness",
            color="exercise",
            size="score",
            labels={"rom": "Range of Motion (%)", "smoothness": "Smoothness (%)"},
            color_discrete_sequence=["#38BDF8", "#818CF8", "#34D399", "#FBBF24", "#F472B6"],
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#F8FAFC")),
            legend=dict(font=dict(color="#F8FAFC", size=10)),
            height=280,
            margin=dict(l=30, r=20, t=20, b=30),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

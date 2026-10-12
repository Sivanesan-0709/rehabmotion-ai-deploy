"""
AI Exercise Form & Posture Intelligence Platform
Community Leaderboard & Athlete Rankings Page
"""

import pandas as pd
import streamlit as st

from app.state import get_app_state
from dashboard.demo.demo_data import get_demo_leaderboard
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
from exercises import list_available_exercises


@safe_render("Community Leaderboard")
def render_leaderboard_page():
    state = get_app_state()
    voice = get_voice_coach()

    render_top_bar(
        "Athlete Community Leaderboard",
        "Compare form quality scores, total repetitions, and biomechanical consistency across athletes."
    )

    # Filter Controls
    col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
    with col_f1:
        ex_filter = st.selectbox("Filter by Exercise", ["All Exercises"] + list_available_exercises(), key="lb_ex_filter")
    with col_f2:
        time_filter = st.selectbox("Timeframe", ["All-Time", "This Month", "This Week"], key="lb_time_filter")
    with col_f3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True, key="lb_refresh_btn"):
            st.toast("Leaderboard rankings refreshed!", icon="🏆")
            st.rerun()

    st.caption(f"{render_demo_badge()} Showing community benchmark rankings", unsafe_allow_html=True)

    # Top Podiums Row (1st, 2nd, 3rd)
    st.markdown("### 🏆 Top Tier Athletes")
    leaders = get_demo_leaderboard()

    pod_cols = st.columns(3)
    medals = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
    border_colors = ["#F59E0B", "#94A3B8", "#B45309"]

    for i in range(min(3, len(leaders))):
        lead = leaders[i]
        with pod_cols[i]:
            st.markdown(
                f"""
                <div class="glass-card" style="border: 2px solid {border_colors[i]}; text-align: center; padding: 16px 12px; margin-bottom: 12px;">
                    <div style="font-weight: 800; font-size: 1.1rem; color: {border_colors[i]}; margin-bottom: 4px;">{medals[i]}</div>
                    <h3 style="margin: 0; color: #F8FAFC;">{lead['athlete']}</h3>
                    <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 600; margin: 4px 0;">{lead['exercise']}</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #22C55E; margin: 8px 0;">{lead['score']:.1f}% Form</div>
                    <div style="font-size: 0.82rem; color: #94A3B8;">{lead['reps']} Verified Reps • {lead['badge']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Full Leaderboard Table
    st.markdown("### 📋 Complete Athlete Standings")

    # Add active user to the leaderboard display
    user_entry = {
        "rank": 4,
        "athlete": f"{state.active_user_name} (You)",
        "exercise": state.exercise_name,
        "reps": 184,
        "score": 91.4,
        "badge": "Elite",
    }
    display_data = list(leaders)
    if not any(state.active_user_name in d["athlete"] for d in display_data):
        display_data.insert(3, user_entry)
        for idx, item in enumerate(display_data):
            item["rank"] = idx + 1

    df_lb = pd.DataFrame(display_data)
    if ex_filter != "All Exercises":
        df_lb = df_lb[df_lb["exercise"] == ex_filter]

    st.dataframe(df_lb, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Issue Challenge Feature
    st.markdown("### ⚔️ Issue an Athlete Challenge")
    with st.container(border=True):
        col_ch1, col_ch2, col_ch3 = st.columns([2, 2, 1])
        with col_ch1:
            target_athlete = st.selectbox("Select Athlete to Challenge", [d["athlete"] for d in leaders], key="sel_target_athlete")
        with col_ch2:
            challenge_reps = st.number_input("Challenge Reps Target", min_value=10, max_value=100, value=25, step=5, key="inp_challenge_reps")
        with col_ch3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("⚔️ Issue Challenge", type="primary", use_container_width=True, key="btn_issue_challenge"):
                st.toast(f"Challenge sent to {target_athlete} for {challenge_reps} reps!", icon="⚔️")
                voice.speak_cue(f"Challenge issued to {target_athlete} for {challenge_reps} reps.")

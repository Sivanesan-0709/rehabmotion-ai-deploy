"""
AI Exercise Form & Posture Intelligence Platform
Rep Quality Timeline & Movement Consistency Alert Page
"""

import pandas as pd
import streamlit as st

from app.state import get_app_state
from dashboard.components.cards import render_consistency_alert
from dashboard.components.timeline_chart import create_rep_quality_timeline_chart
from dashboard.demo.demo_data import get_demo_recent_sessions
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
from exercises import list_available_exercises
from intelligence.scoring import FormScoringEngine


@safe_render("Rep Quality Timeline")
def render_rep_quality_page():
    state = get_app_state()

    render_top_bar(
        "Repetition Quality Timeline & Fatigue Degradation",
        "Granular per-rep scoring, bilateral symmetry tracking, fatigue-induced form drop alerts, and telemetry logs."
    )

    # User & Exercise Filter
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        user_list = [u["name"] for u in state.user_repo.list_all_users()] or ["Default Athlete"]
        selected_user = st.selectbox("Select Athlete Profile", user_list, index=0, key="rq_user_select")
        if selected_user != state.active_user_name:
            state.set_user(selected_user)

    with col2:
        exercise_list = list_available_exercises()
        cur_idx = exercise_list.index(state.exercise_name) if state.exercise_name in exercise_list else 0
        selected_ex = st.selectbox("Select Exercise", exercise_list, index=cur_idx, key="rq_ex_select")

    with col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", use_container_width=True, key="rq_refresh_btn"):
            st.rerun()

    is_demo_user = (state.active_user_name == "Demo Athlete (DEMO)")
    user_reps = state.rep_repo.get_user_all_repetitions(state.active_user_id, selected_ex)
    has_real_reps = len(user_reps) > 0 and not is_demo_user

    if not has_real_reps:
        # Fallback to demo reps for vivid demonstration
        demo_reps_data = []
        for i in range(1, 13):
            sc = max(65.0, min(98.0, 95.0 - (i * 1.8) + float((i % 3) * 2.5)))
            demo_reps_data.append({
                "rep_number": i,
                "score": round(sc, 1),
                "rom": round(88.0 + (i % 4) * 2.0, 1),
                "symmetry": round(93.0 - (i % 3) * 2.0, 1),
                "smoothness": round(90.0 - (i % 4) * 1.5, 1),
                "duration": round(2.3 + (i % 3) * 0.15, 2),
                "status": "Correct" if sc >= 80 else "Form Flaw",
                "issues": [] if sc >= 80 else ["Slight fatigue knee cave during ascent"],
            })
        user_reps = demo_reps_data

    # Aggregate metrics
    scores = [r["score"] for r in user_reps]
    rep_nums = list(range(1, len(scores) + 1))
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    correct_count = sum(1 for r in user_reps if r["status"] == "Correct")

    # Metrics Summary Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_kpi_card("Total Repetitions", str(len(user_reps)), "Evaluated reps", icon="🔢"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_kpi_card("Average Form Score", f"{avg_score:.1f}%", "Kinematic quality", icon="🎯"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_kpi_card("Best Repetition", f"{best_score:.1f}%", "Peak efficiency", icon="⭐"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_kpi_card("Form Accuracy", f"{(correct_count/len(user_reps)*100):.0f}%", "Target compliant", icon="✅"), unsafe_allow_html=True)

    if not has_real_reps:
        st.caption(f"{render_demo_badge()} Showing sample repetition telemetry for {selected_ex}", unsafe_allow_html=True)

    st.markdown("---")

    # Movement Consistency Alert Check
    scoring_engine = FormScoringEngine()
    is_alert, alert_msg = scoring_engine.check_consistency_degradation(scores, window_size=4, drop_threshold=12.0)

    if is_alert and alert_msg:
        st.markdown(render_consistency_alert(alert_msg), unsafe_allow_html=True)

    # Interactive Timeline Chart
    st.markdown("### 📊 Repetition Quality Timeline & Trend")
    fig_tl = create_rep_quality_timeline_chart(rep_nums, scores)
    st.plotly_chart(fig_tl, use_container_width=True)

    st.markdown("---")

    # Interactive Rep Quality Table & CSV Download
    st.markdown("### 📋 Repetition Breakdown Matrix")
    table_rows = []
    for r in user_reps:
        issues_str = ", ".join(r["issues"]) if r["issues"] else "None (Clean Form)"
        table_rows.append({
            "Rep #": r["rep_number"],
            "Form Score (%)": f"{r['score']:.1f}",
            "ROM (%)": f"{r['rom']:.1f}%",
            "Symmetry (%)": f"{r['symmetry']:.1f}%",
            "Smoothness (%)": f"{r['smoothness']:.1f}%",
            "Duration (s)": f"{r['duration']:.2f}s",
            "Status": r["status"],
            "Biomechanical Findings": issues_str,
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, hide_index=True, use_container_width=True)

    # CSV Download Button
    csv_bytes = df_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Repetition Telemetry (.csv)",
        data=csv_bytes,
        file_name=f"rep_quality_telemetry_{selected_ex}.csv",
        mime="text/csv",
        key="btn_dl_rep_csv",
    )

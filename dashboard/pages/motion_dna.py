"""
AI Exercise Form & Posture Intelligence Platform
Motion DNA & Form Fingerprint Analytics Page
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

from app.state import get_app_state
from dashboard.components.radar_chart import create_motion_dna_radar_chart
from dashboard.demo.demo_data import get_demo_motion_dna_profile
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


@safe_render("Motion DNA")
def render_motion_dna_page():
    state = get_app_state()

    render_top_bar(
        "Personalized Motion DNA & Kinematic Fingerprint",
        "Statistical biomechanical baseline tracking individual movement consistency and joint trajectory deviations."
    )

    # Top Selection Filters
    col1, col2 = st.columns([1, 1])
    with col1:
        user_list = [u["name"] for u in state.user_repo.list_all_users()] or ["Default Athlete"]
        selected_user = st.selectbox("Select Athlete Profile", user_list, index=0, key="dna_user_select")
        if selected_user != state.active_user_name:
            state.set_user(selected_user)

    with col2:
        exercise_list = list_available_exercises()
        cur_idx = exercise_list.index(state.exercise_name) if state.exercise_name in exercise_list else 0
        selected_ex = st.selectbox("Select Exercise", exercise_list, index=cur_idx, key="dna_ex_select")
        if selected_ex != state.exercise_name:
            state.set_exercise(selected_ex)

    state.refresh_profile()
    profile = state.active_profile

    # Disclaimer Callout
    st.info(
        "💡 **Academic Note:** Motion DNA is an explainable biomechanical profiling engine calculating Mahalanobis & Z-score "
        "deviations across multi-joint kinematics. It provides personalized athletic consistency analysis."
    )

    if not profile:
        # Fallback to demo profile baseline for live preview / viva demonstration
        demo_prof = get_demo_motion_dna_profile(selected_ex)
        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom: 16px; border-left: 4px solid #38BDF8;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <h3 style="margin: 0; color: #38BDF8;">{selected_user}'s {selected_ex.upper()} CALIBRATION BASELINE</h3>
                        <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.88rem;">
                            Showing calibrated baseline ({demo_prof['baseline_reps']} verified repetitions)
                        </p>
                    </div>
                    <div>
                        {render_status_pill(f"Consistency: {demo_prof['consistency_index']:.1f}%", variant='success')}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_radar, col_metrics = st.columns([1, 1])
        with col_radar:
            st.markdown("#### 🎯 Biomechanical Fingerprint Profile")
            fig_radar = create_motion_dna_radar_chart(demo_prof["scores"], exercise_name=selected_ex)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_metrics:
            st.markdown("#### 📐 Baseline Joint & Kinematic Distributions")
            df_dist = pd.DataFrame(demo_prof["metrics_table"])
            st.dataframe(df_dist, hide_index=True, use_container_width=True)

        st.caption(f"{render_demo_badge()} Showing sample Motion DNA calibration baseline", unsafe_allow_html=True)

        st.markdown("---")
        # Calibration Action Buttons
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("⚡ Generate Live Baseline in Database", type="primary", use_container_width=True, key="btn_gen_db_baseline"):
                from exercises.base import RepetitionSummary
                from features.extractor import FeatureExtractor
                from tests.helpers import create_synthetic_pose

                extractor = FeatureExtractor()
                pose = create_synthetic_pose()
                ex_engine = state.exercise_engine

                sim_summaries = []
                for rep_i in range(8):
                    f = extractor.extract(pose)
                    f.angles["mean_knee"] = 86.0 + float(np.random.normal(0, 3.0))
                    f.angles["mean_elbow"] = 52.0 + float(np.random.normal(0, 2.0))
                    summary = ex_engine.evaluate_completed_rep(
                        rep_number=rep_i + 1,
                        duration=2.3 + float(np.random.normal(0, 0.15)),
                        features_history=[f] * 15
                    )
                    sim_summaries.append(summary)

                new_prof = state.motion_dna_engine.build_profile(
                    user_id=state.active_user_id,
                    exercise_name=selected_ex,
                    reps=sim_summaries,
                    timestamp=time.time()
                )
                if new_prof:
                    state.profile_repo.save_profile(new_prof)
                    state.refresh_profile()
                    st.toast("Personalized Motion DNA baseline saved to database!", icon="🧬")
                    st.rerun()

        with col_c2:
            if st.button("🏋️ Start Live Coach to Calibrate", use_container_width=True, key="btn_calib_live_coach"):
                st.session_state["nav_selection"] = "🏋️ Live Workout Coach"
                st.rerun()
        return

    # Render Active Real Motion DNA Profile
    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom: 16px; border-left: 4px solid #38BDF8;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h3 style="margin: 0; color: #38BDF8;">{selected_user}'s {profile.exercise_name.upper()} FINGERPRINT</h3>
                    <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.88rem;">
                        Calibrated on {profile.total_baseline_reps} verified repetitions from SQLite
                    </p>
                </div>
                <div>
                    {render_status_pill(f"Consistency: {profile.consistency_index:.1f}%", variant='success')}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Radar Chart & Summary Cards
    col_radar, col_metrics = st.columns([1, 1])

    with col_radar:
        st.markdown("#### 🎯 Biomechanical Fingerprint Profile")
        fig_radar = create_motion_dna_radar_chart(profile.fingerprint_scores, exercise_name=profile.exercise_name)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_metrics:
        st.markdown("#### 📐 Baseline Joint & Kinematic Distributions")
        rows = []
        for metric_name, dist in profile.metrics.items():
            label = metric_name.replace("_score", "").replace("_", " ").title()
            unit = "sec" if "duration" in metric_name else ("deg" if "angle" in metric_name else "%")
            rows.append({
                "Biomechanical Metric": label,
                "Mean ± Std": f"{dist.mean:.1f} ± {dist.std:.1f} {unit}",
                "Median": f"{dist.median:.1f} {unit}",
                "IQR (25%-75%)": f"{dist.p25:.1f} - {dist.p75:.1f}",
                "Observed Range": f"{dist.min_val:.1f} - {dist.max_val:.1f}",
            })
        df_dist = pd.DataFrame(rows)
        st.dataframe(df_dist, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Repetition Comparator Tool
    st.markdown("### 🔍 Compare a Repetition Against Your Baseline")
    user_reps = state.rep_repo.get_user_all_repetitions(state.active_user_id, selected_ex)

    if user_reps:
        rep_options = [f"Rep #{r['rep_number']} (Score: {r['score']}, Date: {r['id'][:8]})" for r in user_reps]
        chosen_rep_str = st.selectbox("Select Repetition to Inspect", rep_options, index=len(rep_options) - 1, key="sel_rep_inspect")
        chosen_idx = rep_options.index(chosen_rep_str)
        selected_rep_data = user_reps[chosen_idx]

        from exercises.base import RepetitionSummary
        rep_summary = RepetitionSummary(
            rep_id=selected_rep_data["id"],
            exercise_name=selected_ex,
            rep_number=selected_rep_data["rep_number"],
            duration_seconds=selected_rep_data["duration"],
            form_score=selected_rep_data["score"],
            score_breakdown=selected_rep_data["breakdown"],
            range_of_motion=selected_rep_data["rom"],
            symmetry_score=selected_rep_data["symmetry"],
            smoothness_score=selected_rep_data["smoothness"],
            torso_stability_score=selected_rep_data["torso_stability"],
            issues_detected=selected_rep_data["issues"],
            is_correct=(selected_rep_data["status"] == "Correct"),
            peak_angles={},
            timestamp_start=selected_rep_data["timestamp_start"],
            timestamp_end=selected_rep_data["timestamp_end"]
        )

        deviation = state.motion_dna_engine.calculate_deviation(rep_summary, profile)

        dev_col1, dev_col2 = st.columns([1, 2])
        with dev_col1:
            dev_val = deviation.overall_deviation_pct
            var = "success" if dev_val < 15.0 else ("warning" if dev_val < 30.0 else "danger")
            st.markdown(
                f"""
                <div class="glass-card" style="padding: 16px;">
                    <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">Composite Baseline Deviation</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; margin: 6px 0;">{dev_val:.1f}%</div>
                    {render_status_pill(deviation.deviation_message, variant=var)}
                </div>
                """,
                unsafe_allow_html=True
            )

        with dev_col2:
            st.markdown("##### Standardized Z-Score Deviation Matrix")
            z_rows = []
            for m_key, d_info in deviation.deviation_breakdown.items():
                m_label = m_key.replace("_score", "").replace("_", " ").title()
                z_rows.append({
                    "Feature": m_label,
                    "Current Rep": d_info["current"],
                    "Baseline (Mean ± Std)": f"{d_info['baseline_mean']} ± {d_info['baseline_std']}",
                    "Z-Score": d_info["z_score"],
                    "Deviation Status": "Normal" if abs(d_info["z_score"]) < 1.8 else "Significant Shift ⚠️"
                })
            st.dataframe(pd.DataFrame(z_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No recorded repetitions found for this exercise in SQLite. Perform a workout in Live Coach or click 'Generate Live Baseline' above.")

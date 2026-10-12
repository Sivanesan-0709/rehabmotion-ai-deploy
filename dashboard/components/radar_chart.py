"""
AI Exercise Form & Posture Intelligence Platform
Motion DNA & Form Fingerprint Radar Chart Visualizer
"""

from typing import Dict, List, Optional
import plotly.graph_objects as go


def create_motion_dna_radar_chart(
    fingerprint_scores: Dict[str, float],
    comparison_scores: Optional[Dict[str, float]] = None,
    exercise_name: str = "Squat"
) -> go.Figure:
    """
    Renders an interactive dark-themed Plotly Radar Chart comparing baseline Motion DNA
    against a target repetition or ideal standard.
    """
    categories = list(fingerprint_scores.keys())
    baseline_values = [fingerprint_scores[c] for c in categories]

    # Close the radar loop
    categories_closed = categories + [categories[0]]
    baseline_closed = baseline_values + [baseline_values[0]]

    fig = go.Figure()

    # User Baseline Trace
    fig.add_trace(go.Scatterpolar(
        r=baseline_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(56, 189, 248, 0.25)',
        line=dict(color='#38BDF8', width=2.5),
        name='Personal Motion DNA',
        hoverinfo='r+theta'
    ))

    # Optional Comparison Rep Trace
    if comparison_scores:
        comp_values = [comparison_scores.get(c, 0.0) for c in categories]
        comp_closed = comp_values + [comp_values[0]]
        fig.add_trace(go.Scatterpolar(
            r=comp_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor='rgba(129, 140, 248, 0.20)',
            line=dict(color='#818CF8', width=2, dash='dot'),
            name='Current Repetition',
            hoverinfo='r+theta'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color='#94A3B8'),
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(255, 255, 255, 0.15)',
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#F8FAFC', family='Inter'),
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(255, 255, 255, 0.15)',
            ),
            bgcolor='rgba(15, 23, 42, 0.6)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(
            font=dict(color='#F8FAFC', size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        showlegend=True,
        height=380,
    )

    return fig

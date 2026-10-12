"""
AI Exercise Form & Posture Intelligence Platform
Rep Quality Timeline & Fatigue Degradation Trend Chart
"""

from typing import List, Optional
import plotly.graph_objects as go


def create_rep_quality_timeline_chart(
    rep_numbers: List[int],
    scores: List[float],
    degradation_threshold: float = 70.0,
    height: int = 180,
) -> go.Figure:
    """
    Renders an interactive Plotly bar & trendline chart for Repetition Quality scores.
    Color codes individual reps by quality tier (Emerald >= 85, Teal >= 70, Amber >= 50, Rose < 50).
    """
    if not rep_numbers or not scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No repetitions recorded in this session yet.",
            showarrow=False,
            font=dict(color="#94A3B8", size=12)
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=height,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig

    # Compute bar colors based on score
    colors = []
    for s in scores:
        if s >= 85.0:
            colors.append("#10B981")  # Restrained Emerald
        elif s >= 70.0:
            colors.append("#14B8A6")  # Teal Accent
        elif s >= 50.0:
            colors.append("#F59E0B")  # Amber Warning
        else:
            colors.append("#EF4444")  # Accessible Red

    fig = go.Figure()

    # Bar chart for individual rep scores
    fig.add_trace(go.Bar(
        x=[f"Rep {r}" for r in rep_numbers],
        y=scores,
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.12)", width=1)),
        name="Rep Score",
        hovertemplate="<b>%{x}</b><br>Quality Score: %{y:.1f}%<extra></extra>"
    ))

    # Trend line (Rolling Average or Smoothed Fit)
    if len(scores) >= 3:
        import numpy as np
        rolling_avg = np.convolve(scores, np.ones(3)/3, mode='same')
        fig.add_trace(go.Scatter(
            x=[f"Rep {r}" for r in rep_numbers],
            y=rolling_avg,
            mode="lines+markers",
            line=dict(color="#F8FAFC", width=2, dash="dot"),
            name="Trend",
            hovertemplate="Trend: %{y:.1f}%<extra></extra>"
        ))

    # Target Quality Baseline line at threshold
    fig.add_hline(
        y=degradation_threshold,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"Target ({int(degradation_threshold)}%)",
        annotation_position="bottom right",
        annotation_font=dict(color="#F59E0B", size=9),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        yaxis=dict(
            range=[0, 105],
            title=dict(
                text="Form Score (0-100)",
                font=dict(color="#94A3B8", size=10),
            ),
            tickfont=dict(color="#94A3B8", size=10),
            gridcolor="rgba(255,255,255,0.06)",
        ),
        xaxis=dict(
            tickfont=dict(color="#F8FAFC", size=10),
            gridcolor="rgba(255,255,255,0.0)",
        ),
        margin=dict(l=35, r=15, t=15, b=25),
        height=height,
        showlegend=True,
        legend=dict(font=dict(color="#F8FAFC", size=9), orientation="h", y=1.18, x=0.0),
    )

    return fig

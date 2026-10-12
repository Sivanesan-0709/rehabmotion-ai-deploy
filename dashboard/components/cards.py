"""
AI Exercise Form & Posture Intelligence Platform
Dashboard UI Components: Metric Cards, Status Pills & Badges
"""

from typing import Optional


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    is_positive_delta: bool = True,
    icon: str = ""
) -> str:
    """Generates HTML for a dark glassmorphism metric card."""
    delta_html = ""
    if delta:
        delta_class = "metric-delta-positive" if is_positive_delta else "metric-delta-negative"
        arrow = "&uarr;" if is_positive_delta else "&darr;"
        delta_html = f'<div class="{delta_class}">{arrow} {delta}</div>'

    icon_html = f'<span style="margin-right: 6px;">{icon}</span>' if icon else ""

    return f"""
    <div class="metric-card">
        <div class="metric-label">{icon_html}{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def render_status_pill(status: str, variant: str = "success") -> str:
    """Variants: success, warning, danger, info"""
    icon_map = {
        "success": "&#10003;",
        "warning": "&#9888;",
        "danger": "&#10005;",
        "info": "&#9432;",
    }
    icon = icon_map.get(variant, "&#9679;")
    return f"""
    <span class="status-pill status-pill-{variant}">
        <span>{icon}</span> {status}
    </span>
    """


def render_consistency_alert(message: str) -> str:
    return f"""
    <div class="consistency-alert">
        <strong>&#9888; Movement Consistency Alert</strong><br/>
        <span style="font-size: 0.90rem; opacity: 0.95;">{message}</span>
    </div>
    """

"""
AI Exercise Form & Posture Intelligence Platform
Centralized Design System, Styling Tokens, Glassmorphism Theme & UI Helpers
"""

import functools
import traceback
from typing import Any, Callable, Dict, List, Optional
import streamlit as st


# ==============================================================================
# DESIGN TOKENS
# ==============================================================================
class ThemeTokens:
    # Primary Palette
    BG_DARK = "#0B0F19"
    BG_CARD = "rgba(15, 23, 42, 0.75)"
    BG_CARD_HOVER = "rgba(30, 41, 59, 0.85)"
    BORDER_GLASS = "rgba(255, 255, 255, 0.08)"
    BORDER_GLOW = "rgba(20, 184, 166, 0.35)"

    # Accents
    ACCENT_TEAL = "#14B8A6"       # Primary Teal
    ACCENT_CYAN = "#06B6D4"       # Cyan
    ACCENT_VIOLET = "#8B5CF6"     # Secondary Muted Violet
    ACCENT_INDIGO = "#818CF8"     # Secondary Indigo
    ACCENT_EMERALD = "#10B981"    # Success / Target (Restrained Green)
    ACCENT_AMBER = "#F59E0B"      # Warning
    ACCENT_ROSE = "#EF4444"       # Danger / Accessible Red

    # Typography & Text
    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#94A3B8"
    TEXT_MUTED = "#64748B"
    FONT_FAMILY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

    # Radii & Shadows
    RADIUS_SM = "6px"
    RADIUS_MD = "10px"
    RADIUS_LG = "16px"
    SHADOW_GLASS = "0 4px 20px 0 rgba(0, 0, 0, 0.30)"
    SHADOW_GLOW = "0 0 16px rgba(20, 184, 166, 0.18)"


# ==============================================================================
# GLOBAL CSS STYLES
# ==============================================================================
CUSTOM_CSS = f"""
<style>
/* Global Imports & Base Styling */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: {ThemeTokens.FONT_FAMILY};
    color: {ThemeTokens.TEXT_PRIMARY};
}}

/* Hide Default Streamlit Chrome */
#MainMenu, header, footer {{visibility: hidden !important;}}
.stDeployButton {{display: none !important;}}

/* Top App Container - Compact for Single-Screen Desktop Viewport */
.block-container {{
    padding-top: 0.75rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1650px !important;
}}

/* Sidebar Glassmorphism */
[data-testid="stSidebar"] {{
    background-color: #070A11 !important;
    border-right: 1px solid {ThemeTokens.BORDER_GLASS};
}}

[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
}}

/* Glass Cards */
.glass-card {{
    background: {ThemeTokens.BG_CARD};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid {ThemeTokens.BORDER_GLASS};
    border-radius: {ThemeTokens.RADIUS_MD};
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: {ThemeTokens.SHADOW_GLASS};
    transition: all 0.25s ease-in-out;
}}

.glass-card:hover {{
    border-color: {ThemeTokens.BORDER_GLOW};
    box-shadow: {ThemeTokens.SHADOW_GLOW};
}}

/* KPI Metric Cards */
.kpi-container {{
    display: flex;
    flex-direction: column;
    background: {ThemeTokens.BG_CARD};
    backdrop-filter: blur(12px);
    border: 1px solid {ThemeTokens.BORDER_GLASS};
    border-radius: {ThemeTokens.RADIUS_MD};
    padding: 16px 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}}

.kpi-container:hover {{
    transform: translateY(-2px);
    border-color: {ThemeTokens.BORDER_GLOW};
}}

.kpi-title {{
    font-size: 0.82rem;
    font-weight: 500;
    color: {ThemeTokens.TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 6px;
}}

.kpi-value {{
    font-size: 1.85rem;
    font-weight: 800;
    color: {ThemeTokens.TEXT_PRIMARY};
    margin: 4px 0;
}}

.kpi-subtitle {{
    font-size: 0.78rem;
    color: {ThemeTokens.TEXT_MUTED};
}}

/* Status Pills */
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    width: fit-content;
}}

.status-pill-success {{
    background: rgba(34, 197, 94, 0.15);
    color: #4ADE80;
    border: 1px solid rgba(34, 197, 94, 0.35);
}}

.status-pill-warning {{
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}}

.status-pill-danger {{
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.35);
}}

.status-pill-info {{
    background: rgba(6, 182, 212, 0.15);
    color: #22D3EE;
    border: 1px solid rgba(6, 182, 212, 0.35);
}}

.status-pill-teal {{
    background: rgba(20, 184, 166, 0.15);
    color: #2DD4BF;
    border: 1px solid rgba(20, 184, 166, 0.35);
}}

.status-pill-neutral {{
    background: rgba(148, 163, 184, 0.12);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.25);
}}

/* Demo Data Badge */
.demo-badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    background: rgba(129, 140, 248, 0.18);
    color: #A5B4FC;
    border: 1px dashed rgba(129, 140, 248, 0.45);
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}}

/* Info Banners */
.info-banner {{
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid {ThemeTokens.ACCENT_CYAN};
    padding: 12px 18px;
    border-radius: 4px 10px 10px 4px;
    font-size: 0.88rem;
    color: {ThemeTokens.TEXT_SECONDARY};
    margin: 10px 0 16px 0;
}}

/* Top Page Header */
.top-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 12px;
    margin-bottom: 18px;
    border-bottom: 1px solid {ThemeTokens.BORDER_GLASS};
}}

.top-header-title {{
    font-size: 1.65rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}

.top-header-sub {{
    font-size: 0.90rem;
    color: {ThemeTokens.TEXT_SECONDARY};
    margin: 4px 0 0 0;
}}

/* Streamlit Button Overrides */
div.stButton > button {{
    border-radius: {ThemeTokens.RADIUS_SM};
    font-weight: 600;
    transition: all 0.2s ease;
}}

div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
}}

div.stButton > button[kind="primary"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
}}
</style>
"""


def inject_custom_theme():
    """Injects the global CSS design system into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_custom_theme_css():
    """Alias for inject_custom_theme."""
    inject_custom_theme()


def render_section_header(title: str, subtitle: str = "") -> str:
    """Returns HTML for a styled section heading."""
    return f"""
    <div style="margin: 20px 0 12px 0;">
        <h3 style="color: #F8FAFC; margin: 0; font-size: 1.25rem;">{title}</h3>
        {f'<p style="color: #94A3B8; font-size: 0.85rem; margin: 2px 0 0 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """


# ==============================================================================
# REUSABLE UI COMPONENT HELPERS
# ==============================================================================
def render_compact_workspace_header(
    title: str = "RehabMotion AI",
    athlete_name: str = "Default Athlete",
    exercise_name: str = "Squat",
    is_active: bool = False,
    camera_info: str = "DirectShow 720p • MediaPipe Tasks (33 Keypoints)",
) -> str:
    """Renders a single-line compact workspace header for single-screen 1080p/768p layout."""
    status_badge = (
        '<span class="status-pill status-pill-success">● ACTIVE WORKOUT</span>'
        if is_active
        else '<span class="status-pill status-pill-neutral">○ READY / IDLE</span>'
    )
    return f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 14px; margin-bottom: 8px; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.25);">
        <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
            <span style="font-weight: 800; font-size: 1.05rem; color: #14B8A6; letter-spacing: 0.5px;">⚡ {title}</span>
            <span style="font-size: 0.75rem; color: #475569;">|</span>
            <span style="font-size: 0.82rem; color: #94A3B8;">Athlete: <strong style="color: #F8FAFC;">{athlete_name}</strong></span>
            <span style="font-size: 0.75rem; color: #475569;">|</span>
            <span style="font-size: 0.82rem; color: #94A3B8;">Exercise: <strong style="color: #2DD4BF;">{exercise_name}</strong></span>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 0.74rem; color: #64748B;">📷 {camera_info}</span>
            {status_badge}
        </div>
    </div>
    """


def render_top_bar(title: str, subtitle: str = ""):
    """Renders standardized top bar with notification drawer."""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(
            f"""
            <div style="padding-bottom: 6px;">
                <h2 class="top-header-title">{title}</h2>
                {f'<p class="top-header-sub">{subtitle}</p>' if subtitle else ''}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Initialize notifications in session state if missing
        if "notifications" not in st.session_state:
            from dashboard.demo.demo_data import get_demo_notifications
            st.session_state["notifications"] = get_demo_notifications()

        unread_count = sum(1 for n in st.session_state["notifications"] if not n["read"])
        badge_str = f"🔔 ({unread_count})" if unread_count > 0 else "🔔"

        with st.popover(badge_str, help="System Alerts & Notifications"):
            st.markdown("#### 📬 Notifications")
            if unread_count > 0:
                if st.button("Mark All as Read", key=f"mark_all_read_{title.lower().replace(' ', '_')}"):
                    for n in st.session_state["notifications"]:
                        n["read"] = True
                    st.rerun()

            for n in st.session_state["notifications"]:
                icon = "🟢" if n["type"] == "success" else ("⚠️" if n["type"] == "warning" else "ℹ️")
                weight = "font-weight: 700;" if not n["read"] else "opacity: 0.75;"
                st.markdown(
                    f"""
                    <div style="padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.06); {weight}">
                        {icon} <strong>{n['title']}</strong><br>
                        <span style="font-size: 0.8rem; color: #94A3B8;">{n['message']}</span><br>
                        <span style="font-size: 0.7rem; color: #64748B;">{n['time']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def render_kpi_card(title: str, value: str, subtitle: str = "", icon: str = "") -> str:
    """Returns HTML for an aesthetic glassmorphic KPI metric card."""
    return f"""
    <div class="kpi-container">
        <div class="kpi-title">{icon} {title}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ''}
    </div>
    """


def render_status_pill(text: str, variant: str = "info") -> str:
    """Returns HTML for a color-coded status badge pill."""
    var_class = f"status-pill-{variant}" if variant in ["success", "warning", "danger", "info"] else "status-pill-info"
    return f'<span class="status-pill {var_class}">{text}</span>'


def render_demo_badge() -> str:
    """Returns HTML for a standardized DEMO DATA badge."""
    return '<span class="demo-badge">📊 DEMO DATA</span>'


def render_info_banner(message: str) -> str:
    """Returns HTML for a styled information callout banner."""
    return f'<div class="info-banner">💡 {message}</div>'


def render_empty_state(
    title: str,
    description: str,
    icon: str = "📭",
    button_label: Optional[str] = None,
    button_key: Optional[str] = None,
    on_click_action: Optional[Callable] = None,
):
    """Renders a friendly empty state placeholder with working action button."""
    st.markdown(
        f"""
        <div class="glass-card" style="text-align: center; padding: 40px 20px; margin: 20px 0;">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">{icon}</div>
            <h3 style="color: {ThemeTokens.ACCENT_CYAN}; margin-bottom: 6px;">{title}</h3>
            <p style="color: {ThemeTokens.TEXT_SECONDARY}; max-width: 500px; margin: 0 auto 16px auto; font-size: 0.9rem;">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if button_label:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(button_label, key=button_key or f"empty_btn_{title}", type="primary", use_container_width=True):
                if on_click_action:
                    on_click_action()


# ==============================================================================
# SAFE RENDER WRAPPER (NO UNCAUGHT EXCEPTIONS)
# ==============================================================================
def safe_render(page_name: str):
    """
    Decorator that wraps any page render function to catch unexpected exceptions
    and render a graceful, user-friendly recovery card instead of a raw traceback.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_detail = traceback.format_exc()
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left: 4px solid {ThemeTokens.ACCENT_ROSE};">
                        <h3 style="color: {ThemeTokens.ACCENT_ROSE}; margin: 0 0 8px 0;">⚠️ Page Notice: {page_name}</h3>
                        <p style="color: {ThemeTokens.TEXT_SECONDARY}; font-size: 0.9rem;">
                            An unexpected issue occurred while rendering this page. You can refresh the view or reset your session state.
                        </p>
                        <div class="status-pill status-pill-danger" style="margin-top: 4px;">
                            Error: {str(e)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                with st.expander("🛠️ Technical Diagnostic Details"):
                    st.code(err_detail, language="python")

                col_rec1, col_rec2 = st.columns([1, 4])
                with col_rec1:
                    if st.button(f"🔄 Reload {page_name}", key=f"retry_{page_name.lower().replace(' ', '_')}"):
                        st.rerun()
        return wrapper
    return decorator

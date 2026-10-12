"""
AI Exercise Form & Posture Intelligence Platform
Voice Coach UI Components (HUD Quick Toggle, Settings Panel, and Real-Time Speaking Indicator)
"""

import streamlit as st
from dashboard.voice import get_voice_coach
from dashboard.theme import render_status_pill, ThemeTokens


def render_voice_quick_toggle(key_prefix: str = "sidebar"):
    """Renders a compact voice mute/unmute toggle switch."""
    voice = get_voice_coach()
    is_active = voice.enabled and voice.mode != "Silent"

    col1, col2 = st.columns([1, 4])
    with col1:
        icon = "🔊" if is_active else "🔇"
        if st.button(icon, key=f"{key_prefix}_voice_toggle_btn", help="Click to toggle Voice Coach"):
            new_state = not is_active
            voice.update_settings(enabled=new_state, mode="Coach" if new_state else "Silent")
            st.toast("Voice Coach Enabled" if new_state else "Voice Coach Muted", icon="🔊" if new_state else "🔇")
            st.rerun()

    with col2:
        st.markdown(
            f"""
            <div style="padding-top: 6px; font-size: 0.85rem;">
                <strong>Voice Coach:</strong> {'Active (Coach)' if is_active else 'Muted'}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_voice_status_chip():
    """Renders real-time HUD chip indicating what the voice coach just said."""
    voice = get_voice_coach()

    if not voice.enabled or voice.mode == "Silent":
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 10px 14px; margin-bottom: 10px;">
                <div style="font-size: 0.75rem; color: #64748B;">🎙️ VOICE COACH STATUS</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 2px;">Muted / Disabled</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    latest_text = voice.last_spoken_text or "Listening for reps..."
    is_speaking = voice.is_currently_speaking
    pill_variant = "success" if is_speaking else "info"
    status_label = "🔊 Speaking..." if is_speaking else "🎙️ Ready"

    st.markdown(
        f"""
        <div class="glass-card" style="padding: 12px 14px; margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">🎙️ AUDIO COACH HUD</span>
                {render_status_pill(status_label, variant=pill_variant)}
            </div>
            <div style="font-size: 0.92rem; font-weight: 700; color: #38BDF8; font-style: italic;">
                "{latest_text}"
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_voice_settings_panel():
    """Renders comprehensive Voice Coach configuration card with working interactive controls."""
    voice = get_voice_coach()
    available_voices = voice.get_available_voices()

    st.markdown("#### 🔊 Voice Coach Preferences")

    col1, col2 = st.columns(2)

    with col1:
        enabled = st.toggle("Enable Voice Coach", value=voice.enabled, key="voice_enabled_toggle")
        mode_options = ["Coach", "Count only", "Silent"]
        current_mode_idx = mode_options.index(voice.mode) if voice.mode in mode_options else 0
        selected_mode = st.selectbox("Speaking Mode", mode_options, index=current_mode_idx, key="voice_mode_select")

        freq_options = [1, 2, 5]
        freq_labels = ["Every Repetition (1, 2, 3...)", "Every 2nd Repetition", "Every 5th Repetition (Milestones only)"]
        curr_freq_idx = freq_options.index(voice.announce_interval) if voice.announce_interval in freq_options else 0
        selected_freq_label = st.radio("Announcement Frequency", freq_labels, index=curr_freq_idx, key="voice_freq_radio")
        selected_freq = freq_options[freq_labels.index(selected_freq_label)]

    with col2:
        speed = st.slider("Speech Rate (WPM)", 120, 260, int(voice.rate), 10, key="voice_speed_slider")
        volume = st.slider("Audio Volume", 0.1, 1.0, float(voice.volume), 0.05, key="voice_volume_slider")
        voice_idx = st.selectbox(
            "Synthesizer Voice",
            range(len(available_voices)),
            format_func=lambda i: available_voices[i],
            index=min(voice.voice_index, len(available_voices) - 1),
            key="voice_picker_select"
        )

    col_btn1, col_btn2 = st.columns([2, 3])
    with col_btn1:
        if st.button("🔊 Test Voice Output", type="primary", use_container_width=True, key="btn_test_voice_panel"):
            voice.update_settings(
                enabled=enabled,
                mode=selected_mode,
                rate=speed,
                volume=volume,
                voice_index=voice_idx,
                announce_interval=selected_freq,
            )
            voice.test_voice()
            st.toast("Voice test triggered: 'Voice Coach active. One, two, three.'", icon="🔊")

    with col_btn2:
        if st.button("💾 Save Settings", use_container_width=True, key="btn_save_voice_panel"):
            voice.update_settings(
                enabled=enabled,
                mode=selected_mode,
                rate=speed,
                volume=volume,
                voice_index=voice_idx,
                announce_interval=selected_freq,
            )
            st.success("Voice Coach preferences saved!")

    # Recent Spoken Logs Expander
    if voice.recent_log:
        with st.expander("📜 Recent Audio Logs"):
            for log in reversed(voice.recent_log):
                st.markdown(f"• `[{log['time']}]` (Priority {log['priority']}): **{log['text']}**")

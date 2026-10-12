"""
AI Exercise Form & Posture Intelligence Platform
AI Coach Assistant & Biomechanical Q&A Page
"""

import streamlit as st

from app.state import get_app_state
from dashboard.theme import (
    ThemeTokens,
    render_kpi_card,
    render_section_header,
    render_status_pill,
    render_top_bar,
    safe_render,
)
from dashboard.voice import get_voice_coach

# Domain Knowledge Base for Common Form Queries
COACH_KNOWLEDGE_BASE = {
    "knee cave": (
        "**Biomechanical Analysis: Knee Valgus (Knee Cave)**\n\n"
        "• **Root Cause:** Often driven by weak gluteus medius/maximus muscles or limited ankle dorsiflexion.\n"
        "• **Immediate Fix:** 'Screw your feet into the floor' to activate external hip rotators. Push your knees out over your 2nd and 3rd toes.\n"
        "• **Corrective Drill:** Banded goblet squats and clamshells before heavy sets.\n"
        "• **Target Metric:** Maintain a Knee-Hip-Ankle collinearity angle > 165° during both eccentric and concentric phases."
    ),
    "lower back": (
        "**Biomechanical Analysis: Lumbar Flexion (Rounding Lower Back)**\n\n"
        "• **Root Cause:** Inadequate hip hinge mechanics, tight hamstrings, or weak abdominal bracing (intra-abdominal pressure).\n"
        "• **Immediate Fix:** Engage the lats before initiating pull ('protect your armpits'). Maintain a neutral cervical and lumbar spine.\n"
        "• **Corrective Drill:** Romanian Deadlifts (RDLs) with a PVC pipe along the spine (head, upper back, sacrum touching).\n"
        "• **Target Metric:** Spine deviation angle should remain < 12° from neutral axis throughout the rep."
    ),
    "pushup depth": (
        "**Biomechanical Analysis: Pushup Range of Motion & Elbow Flare**\n\n"
        "• **Root Cause:** Excessive elbow flare (> 70° relative to torso) places extreme torque on the anterior shoulder capsule.\n"
        "• **Immediate Fix:** Tuck elbows to approximately 45°, forming an arrow shape with your torso, not a 'T'.\n"
        "• **Target Metric:** Elbow inflection angle should reach ≤ 90° at the bottom of the movement for full chest engagement."
    ),
    "symmetry": (
        "**Biomechanical Analysis: Bilateral Symmetry Index (BSI)**\n\n"
        "• **Definition:** Compares left vs. right joint kinematics during simultaneous bilateral movements.\n"
        "• **Benchmark:** A BSI > 90% is considered elite symmetry. Under 80% indicates unilateral compensation.\n"
        "• **Fix:** Integrate unilateral exercises (Bulgarian split squats, single-arm presses) to address imbalances."
    ),
    "default": (
        "**AI Biomechanical Coach Response:**\n\n"
        "Focus on controlled eccentric tempo (2-3 seconds down), explosive concentric drive, and full joint lockout at peak extension. "
        "Keep your core actively braced, maintain a neutral spine, and ensure your camera is positioned 2.5–3 meters away perpendicular to your movement plane for optimal 3D joint landmark tracking."
    ),
}


def _generate_coach_response(prompt: str) -> str:
    p_lower = prompt.lower()
    if "knee" in p_lower or "cave" in p_lower or "valgus" in p_lower:
        return COACH_KNOWLEDGE_BASE["knee cave"]
    elif "back" in p_lower or "lumbar" in p_lower or "spine" in p_lower or "deadlift" in p_lower:
        return COACH_KNOWLEDGE_BASE["lower back"]
    elif "pushup" in p_lower or "depth" in p_lower or "chest" in p_lower or "elbow" in p_lower:
        return COACH_KNOWLEDGE_BASE["pushup depth"]
    elif "symmetry" in p_lower or "bsi" in p_lower or "imbalance" in p_lower:
        return COACH_KNOWLEDGE_BASE["symmetry"]
    else:
        return COACH_KNOWLEDGE_BASE["default"]


@safe_render("AI Coach Assistant")
def render_ai_coach_page():
    state = get_app_state()
    voice = get_voice_coach()

    render_top_bar(
        "AI Biomechanical Coach & Form Assistant",
        "Ask questions regarding exercise kinematics, form correction drills, injury prevention, and scoring metrics."
    )

    # Initialize chat history in session_state
    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = [
            {
                "role": "assistant",
                "content": f"Hello {state.active_user_name}! I am your AI Biomechanical Coach. I analyze your 3D joint kinematics, posture deviations, and repetition quality. How can I help optimize your training today?"
            }
        ]

    # Quick Prompt Chips
    st.markdown("##### 💡 Suggested Questions")
    chips = [
        "How do I fix knee cave in squats?",
        "What causes lower back rounding in deadlifts?",
        "How can I improve my pushup depth and elbow angle?",
        "Explain bilateral symmetry score (BSI)",
    ]

    col_chips = st.columns(len(chips))
    for i, chip_text in enumerate(chips):
        with col_chips[i]:
            if st.button(chip_text, key=f"chip_btn_{i}", use_container_width=True):
                st.session_state["ai_chat_history"].append({"role": "user", "content": chip_text})
                response = _generate_coach_response(chip_text)
                st.session_state["ai_chat_history"].append({"role": "assistant", "content": response})
                st.rerun()

    st.markdown("---")

    # Render Chat Message Stream
    for idx, msg in enumerate(st.session_state["ai_chat_history"]):
        role = msg["role"]
        with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
            st.markdown(msg["content"])
            if role == "assistant" and idx > 0:
                col_btn, _ = st.columns([1, 4])
                with col_btn:
                    if st.button("🔊 Read Aloud", key=f"btn_speak_msg_{idx}"):
                        # Extract first 2 sentences for TTS speech
                        summary = msg["content"].split("\n")[0].replace("*", "")
                        voice.speak_cue(summary)
                        st.toast("Voice coach reading advice...", icon="🔊")

    # Chat Input Box
    user_input = st.chat_input("Ask a question about your exercise form, kinematics, or training drills...")
    if user_input:
        st.session_state["ai_chat_history"].append({"role": "user", "content": user_input})
        response = _generate_coach_response(user_input)
        st.session_state["ai_chat_history"].append({"role": "assistant", "content": response})
        st.rerun()

    # Clear Chat History Button
    if len(st.session_state["ai_chat_history"]) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Conversation", key="btn_clear_ai_chat"):
            st.session_state["ai_chat_history"] = [
                {
                    "role": "assistant",
                    "content": f"Conversation cleared. How else can I assist your workout, {state.active_user_name}?"
                }
            ]
            st.rerun()

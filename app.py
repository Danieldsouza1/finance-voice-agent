import streamlit as st
import os
import tempfile
import time  # <-- Added for unique audio filenames
from dotenv import load_dotenv

from agent.finance_agent import build_agent, run_agent
from agent.memory import ConversationMemory
from stt import transcribe_audio
from tts import text_to_speech
import subprocess

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance AI Assistant",
    page_icon="📈",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0a0a; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); }

    .header-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .header-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #0066ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: #888;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border: 1px solid #1e4d7b;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e0f0ff;
        font-size: 0.95rem;
    }
    .chat-bubble-agent {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #c8d8e8;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .label-user {
        font-size: 0.75rem;
        color: #4a9eff;
        font-weight: 600;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .label-agent {
        font-size: 0.75rem;
        color: #888;
        font-weight: 600;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }

    .status-box {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 10px 16px;
        color: #6b7280;
        font-size: 0.85rem;
        text-align: center;
        margin: 8px 0;
    }
    .status-box.active {
        border-color: #00d4ff;
        color: #00d4ff;
    }

    .disclaimer {
        font-size: 0.75rem;
        color: #4b5563;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        border-top: 1px solid #1f2937;
        margin-top: 1rem;
    }

    div[data-testid="stAudioInput"] {
        background: #111827;
        border: 2px dashed #1e4d7b;
        border-radius: 16px;
        padding: 1rem;
    }
    .stTextInput input {
        background: #111827 !important;
        color: #e0f0ff !important;
        border: 1px solid #1e4d7b !important;
        border-radius: 10px !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #0066ff, #00d4ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("Loading AI agent..."):
        st.session_state.agent = build_agent()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if "status" not in st.session_state:
    st.session_state.status = "Ready"

if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="header-title">📈 Finance AI</div>
    <div class="header-subtitle">VOICE-POWERED INVESTMENT ASSISTANT</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ── Helper: process a query ───────────────────────────────────────────────────
def process_query(user_text: str):
    if not user_text.strip():
        return

    st.session_state.chat_log.append({"role": "user", "text": user_text})
    st.session_state.status = "Thinking..."

    with st.spinner("Analyzing market data..."):
        response = run_agent(
            st.session_state.agent,
            user_text,
            st.session_state.memory.get_history()
        )

    st.session_state.memory.add_turn(user_text, response)

    st.session_state.status = "Generating voice response..."
    
    # <-- FIX 2: Unique audio filename so history doesn't get overwritten
    unique_filename = f"response_{int(time.time())}.mp3"
    audio_path = os.path.join(tempfile.gettempdir(), unique_filename)
    
    tts_result = text_to_speech(response, audio_path)

    st.session_state.chat_log.append({
        "role": "agent",
        "text": response,
        "audio": tts_result if tts_result else None
    })

    st.session_state.status = "Ready"


# ── Input section ─────────────────────────────────────────────────────────────
st.markdown("### 🎙️ Ask a question")

tab1, tab2 = st.tabs(["Voice Input", "Text Input"])

with tab1:
    st.markdown("Record your question below:")
    audio_input = st.audio_input("Record", key="audio_recorder", label_visibility="collapsed")

    if audio_input is not None:
        raw_bytes = audio_input.getvalue()
        audio_hash = hash(raw_bytes)
        
        # <-- FIX 1: Wrap in an if-statement instead of using st.stop()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            st.caption(f"Audio captured: {len(raw_bytes):,} bytes")

            # Write raw bytes and let Whisper load directly via ffmpeg
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(raw_bytes)
                raw_path = tmp.name

            with st.spinner("Transcribing your voice..."):
                transcribed = transcribe_audio(raw_path)

            try:
                os.unlink(raw_path)
            except:
                pass

            if transcribed:
                st.success(f'Heard: *"{transcribed}"*')
                with st.spinner("Getting response..."):
                    process_query(transcribed)
                st.rerun()
            else:
                st.error("Could not transcribe audio. Please try again.")


with tab2:
    col1, col2 = st.columns([4, 1])

    with col1:
        text_input = st.text_input(
            "",
            placeholder="e.g. Which IT stocks should I consider?",
            key="text_query",
            label_visibility="collapsed"
        )

    with col2:
        ask_btn = st.button("Ask", use_container_width=True)

    if ask_btn and text_input:
        process_query(text_input)
        st.rerun()


# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.chat_log:
    st.markdown("---")
    st.markdown("### 💬 Conversation")

    for entry in reversed(st.session_state.chat_log):
        if entry["role"] == "user":
            st.markdown('<div class="label-user">YOU</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-user">{entry["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="label-agent">FINANCE AI</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-agent">{entry["text"]}</div>', unsafe_allow_html=True)

            if entry.get("audio") and os.path.exists(entry["audio"]):
                st.audio(entry["audio"], format="audio/mp3", autoplay=False)

    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_log = []
        st.session_state.memory.clear()
        st.rerun()



# ── Status bar ────────────────────────────────────────────────────────────────
status_class = "active" if st.session_state.status != "Ready" else ""
st.markdown(
    f'<div class="status-box {status_class}">⬤ {st.session_state.status}</div>',
    unsafe_allow_html=True
)


# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    This tool is for informational and educational purposes only.<br>
    It does not constitute financial advice. Always consult a certified financial advisor.
</div>
""", unsafe_allow_html=True)
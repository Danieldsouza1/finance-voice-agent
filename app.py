import streamlit as st
import os
import tempfile
import time
import json
import re
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

    .header-container { text-align: center; padding: 2rem 0 1rem 0; }
    .header-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #0066ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .header-subtitle { color: #888; font-size: 0.95rem; letter-spacing: 0.05em; }

    .chat-bubble-user {
        background: linear-gradient(135deg, #1e3a5f, #0d2137); border: 1px solid #1e4d7b;
        border-radius: 16px 16px 4px 16px; padding: 12px 16px; margin: 8px 0; color: #e0f0ff;
    }
    .chat-bubble-agent {
        background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #2a2a4a;
        border-radius: 16px 16px 16px 4px; padding: 12px 16px; margin: 8px 0; color: #c8d8e8;
    }
    .label-user, .label-agent { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; margin-bottom: 4px; }
    .label-user { color: #4a9eff; }
    .label-agent { color: #888; }

    .status-box {
        background: #111827; border: 1px solid #1f2937; border-radius: 10px;
        padding: 10px 16px; color: #6b7280; font-size: 0.85rem; text-align: center; margin: 8px 0;
    }
    .status-box.active { border-color: #00d4ff; color: #00d4ff; }

    div[data-testid="stAudioInput"] { background: #111827; border: 2px dashed #1e4d7b; border-radius: 16px; padding: 1rem; }
    .stTextInput input { background: #111827 !important; color: #e0f0ff !important; border: 1px solid #1e4d7b !important; border-radius: 10px !important; }
    .stButton button { background: linear-gradient(90deg, #0066ff, #00d4ff) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "agent" not in st.session_state:
    with st.spinner("Loading AI agent..."):
        st.session_state.agent = build_agent()
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "status" not in st.session_state:
    st.session_state.status = "Ready"
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "memory_data" not in st.session_state:
    st.session_state.memory_data = []
if "pending_transcription" not in st.session_state:
    st.session_state.pending_transcription = None
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

# Rebuild memory object from serialized data each run
memory = ConversationMemory.from_dict(st.session_state.memory_data)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="header-container"><div class="header-title">📈 Finance AI</div><div class="header-subtitle">VOICE-POWERED INVESTMENT ASSISTANT</div></div>', unsafe_allow_html=True)
st.markdown("---")

# ── Helper: process a query ───────────────────────────────────────────────────
def process_query(user_text: str):
    if not user_text.strip():
        return

    st.session_state.chat_log.append({"role": "user", "ui_text": user_text, "spoken_text": ""})
    st.session_state.status = "Thinking..."

    with st.spinner("Analyzing market data..."):
        try:
            raw_response = run_agent(
                st.session_state.agent,
                user_text,
                memory.get_history()  # CORRECT — uses local rebuilt object
            )
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RateLimitError" in error_msg or "quota" in error_msg.lower():
                raw_response = '<ui_text>⚠️ **Rate Limit Reached.** Please wait 30 seconds and try again.</ui_text><spoken_text>I am receiving too many requests. Please wait 30 seconds and try again.</spoken_text>'
            elif "tool_use_failed" in error_msg or "Failed to call a function" in error_msg:
                raw_response = '<ui_text>⚠️ **Temporary Error.** The AI had trouble calling the data tool. Please try asking your question again.</ui_text><spoken_text>I had a temporary issue fetching the data. Please ask your question again.</spoken_text>'
            else:
                raw_response = f'<ui_text>⚠️ **System Error:** {error_msg}</ui_text><spoken_text>I encountered an error. Please try again.</spoken_text>'

    # ── Bulletproof XML Parsing Logic ──
    # ── Bulletproof XML Parsing Logic ──
    try:
        # Look for the UI text and Spoken text between the tags
        ui_match = re.search(r'<ui_text>(.*?)</ui_text>', raw_response, re.DOTALL | re.IGNORECASE)
        spoken_match = re.search(r'<spoken_text>(.*?)</spoken_text>', raw_response, re.DOTALL | re.IGNORECASE)
        
        if ui_match and spoken_match:
            ui_display = ui_match.group(1).strip().replace('\\n', '\n')
            voice_script = spoken_match.group(1).strip()
        else:
            # If tags are missing, force the fallback
            raise ValueError("XML tags missing")

    except Exception:
        # Fallback: If the AI just writes normal text, show it all on screen
        ui_display = raw_response
        
        # For the voice, grab just the very last paragraph and clean out markdown symbols
        paragraphs = [p for p in raw_response.split("\n") if p.strip()]
        if paragraphs:
            voice_script = paragraphs[-1].replace("*", "").replace("#", "").replace("_", "")
        else:
            voice_script = "I have displayed the information on your screen."

    # --- SAFETY CHECK: If Voice Script is too short (e.g., just the disclaimer) ---
    if len(voice_script.strip()) < 20:
        clean = re.sub(r'[#*|`\[\]()]', '', ui_display)
        clean = re.sub(r'\s+', ' ', clean).strip()
        voice_script = clean[:400] 

    # --- MEMORY FIX: Save raw_response so the AI remembers to use XML tags ---
    memory.add_turn(user_text, raw_response)
    st.session_state.memory_data = memory.to_dict()

    st.session_state.status = "Generating voice response..."
    unique_filename = f"jio_response_{int(time.time())}.mp3"
    audio_path = os.path.join(tempfile.gettempdir(), unique_filename)
    
    # Pass ONLY the clean voice script to TTS
    tts_result = text_to_speech(voice_script, audio_path)

    # Add `played: False` for the autoplay functionality
    st.session_state.chat_log.append({
        "role": "agent",
        "ui_text": ui_display,
        "audio": tts_result if tts_result else None,
        "played": False
    })

    st.session_state.status = "Ready"

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown("### 🎙️ Ask a question")

tab1, tab2 = st.tabs(["Voice Input", "Text Input"])

with tab1:
    st.markdown("Record your question below:")
    audio_input = st.audio_input("Record your voice", key=f"audio_recorder_{st.session_state.audio_key}", label_visibility="collapsed")

    if audio_input is not None:
        raw_bytes = audio_input.getvalue()
        audio_hash = hash(raw_bytes)
        
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            st.caption(f"Audio captured: {len(raw_bytes):,} bytes")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(raw_bytes)
                raw_path = tmp.name

            with st.spinner("Transcribing your voice..."):
                transcribed = transcribe_audio(raw_path)

            try: os.unlink(raw_path)
            except: pass

            if transcribed:
                # Store transcription in session state, don't process yet
                st.session_state.pending_transcription = transcribed
                st.session_state.audio_key += 1
                st.rerun()
            else:
                st.error("Could not transcribe audio. Please try again.")

    # Show confirmation UI if there's a pending transcription
    if st.session_state.get("pending_transcription"):
        transcribed = st.session_state.pending_transcription
        st.info(f'🎤 Heard: **"{transcribed}"**')
        
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("✅ Yes, that's correct", use_container_width=True):
                st.session_state.pending_transcription = None
                with st.spinner("Getting response..."):
                    process_query(transcribed)
                st.rerun()
        with col_b:
            if st.button("❌ No, let me retype", use_container_width=True):
                st.session_state.pending_transcription = None
                st.rerun()

with tab2:
    col1, col2 = st.columns([4, 1])
    with col1:
        text_input = st.text_input(
            "Type your question", 
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
            st.markdown(f'<div class="chat-bubble-user">{entry["ui_text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="label-agent">FINANCE AI</div>', unsafe_allow_html=True)
            # Use st.markdown directly so tables and formatting render properly
            with st.container():
                st.markdown(entry["ui_text"])

            if entry.get("audio") and os.path.exists(entry["audio"]):
                if not entry.get("played", True):
                    st.audio(entry["audio"], format="audio/mp3", autoplay=True)
                    entry["played"] = True
                else:
                    st.audio(entry["audio"], format="audio/mp3", autoplay=False)

    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_log = []
        st.session_state.memory_data = []  # CORRECT — clears the serialized memory
        st.rerun()

# ── Status bar ────────────────────────────────────────────────────────────────
status_class = "active" if st.session_state.status != "Ready" else ""
st.markdown(f'<div class="status-box {status_class}">⬤ {st.session_state.status}</div>', unsafe_allow_html=True)
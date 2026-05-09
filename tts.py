import os
import re
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        from elevenlabs.client import ElevenLabs
        _client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    return _client


def clean_text_for_tts(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s?', '', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'-{2,}', '', text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # Fix: add space before dollar amounts instead of just removing $
    text = re.sub(r'\$(\d)', r' \1 dollars ', text)
    text = text.replace('₹', ' rupees ')
    text = re.sub(r'(\d+\.?\d*)\s*%', r'\1 percent', text)
    # Fix collapsed words — add space before capital letters after lowercase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_speech_elevenlabs(text: str, output_path: str = "response.mp3") -> str:
    """Tries ElevenLabs TTS. Returns path on success, empty string on failure."""
    try:
        from elevenlabs import VoiceSettings
        client = get_client()
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

        if not voice_id:
            return ""

        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.3,
                use_speaker_boost=True,
            ),
            output_format="mp3_44100_128",
        )

        with open(output_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

        return output_path

    except Exception as e:
        print(f"ElevenLabs error: {e}")
        return ""


def text_to_speech_gtts(text: str, output_path: str = "response.mp3") -> str:
    """Fallback TTS using gTTS. Always free."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"gTTS error: {e}")
        return ""


def text_to_speech(text: str, output_path: str = "response.mp3") -> str:
    """
    Main TTS function. Tries ElevenLabs first, falls back to gTTS automatically.
    """
    cleaned = clean_text_for_tts(text)

    # Try ElevenLabs first
    result = text_to_speech_elevenlabs(cleaned, output_path)
    if result:
        print("Using ElevenLabs voice.")
        return result

    # Fallback to gTTS
    print("Falling back to gTTS.")
    return text_to_speech_gtts(cleaned, output_path)
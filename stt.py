import whisper
import os
import warnings
import numpy as np
import subprocess
warnings.filterwarnings("ignore")

_model = None

def load_whisper_model(size: str = "base"):
    global _model
    if _model is None:
        _model = whisper.load_model(size)
    return _model

def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 1000:
        return ""
    try:
        model = load_whisper_model("base")
        wav_path = audio_path + "_converted.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True
        )
        audio = whisper.load_audio(wav_path if os.path.exists(wav_path) else audio_path)
        if np.abs(audio).max() < 0.001:
            return ""
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(language="en", fp16=False, without_timestamps=True)
        result = whisper.decode(model, mel, options)
        try:
            os.unlink(wav_path)
        except:
            pass
        return result.text.strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
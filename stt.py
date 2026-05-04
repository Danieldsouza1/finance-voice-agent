import whisper
import os
import warnings
import numpy as np
import subprocess
import tempfile
warnings.filterwarnings("ignore")

_model = None

def load_whisper_model(size: str = "base"):
    global _model
    if _model is None:
        print(f"Loading Whisper {size} model...")
        _model = whisper.load_model(size)
        print("Whisper model loaded.")
    return _model


def transcribe_audio(audio_path: str) -> str:
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        return ""

    print(f"Transcribing: {audio_path} ({os.path.getsize(audio_path):,} bytes)")

    try:
        model = load_whisper_model("base")

        # Convert to PCM wav using ffmpeg with explicit codec
        wav_path = audio_path + "_converted.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-vn",                    # no video
            "-acodec", "pcm_s16le",   # force PCM 16-bit signed little-endian
            "-ar", "16000",           # 16kHz sample rate
            "-ac", "1",               # mono
            wav_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"ffmpeg return code: {result.returncode}")
        if result.returncode != 0:
            print(f"ffmpeg stderr: {result.stderr[-500:]}")

        # Load and inspect
        if os.path.exists(wav_path):
            audio = whisper.load_audio(wav_path)
            print(f"Audio: {len(audio)} samples, {len(audio)/16000:.1f}s, range: {audio.min():.4f} to {audio.max():.4f}")

            if np.abs(audio).max() < 0.001:
                print("Still silent after conversion — trying direct load")
                # Try loading the original webm directly
                audio = whisper.load_audio(audio_path)
                print(f"Direct load: {len(audio)} samples, range: {audio.min():.4f} to {audio.max():.4f}")

            try:
                os.unlink(wav_path)
            except:
                pass
        else:
            print("WAV not created, loading original directly")
            audio = whisper.load_audio(audio_path)
            print(f"Direct load: {len(audio)} samples, range: {audio.min():.4f} to {audio.max():.4f}")

        if np.abs(audio).max() < 0.001:
            print("Audio is silent after all attempts.")
            return ""

        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected = max(probs, key=probs.get)
        print(f"Detected language: {detected}")

        options = whisper.DecodingOptions(language="en", fp16=False, without_timestamps=True)
        result = whisper.decode(model, mel, options)
        text = result.text.strip()
        print(f"Result: '{text}'")
        return text

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return ""
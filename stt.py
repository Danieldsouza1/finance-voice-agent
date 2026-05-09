import whisper
import os
import warnings
import numpy as np
import subprocess
import difflib
import re

warnings.filterwarnings("ignore")

_model = None

# --- CHANGE 1: The Whisper Prompt Bias ---
FINANCE_VOCABULARY_PROMPT = (
    "Tata Motors, Tata Power, Tata Steel, Tata Consultancy, TCS, "
    "Zomato, Infosys, Reliance, HDFC Bank, ICICI Bank, Axis Bank, "
    "SBI, State Bank of India, Wipro, HCL Tech, Bajaj Finance, "
    "Adani, Maruti Suzuki, Asian Paints, Sun Pharma, ONGC, "
    "Kotak Mahindra, MRF, ITC, Bharti Airtel, Nifty, Sensex, NSE, BSE, "
    "stock, fundamentals, PE ratio, EPS, market cap, dividend yield."
)

# --- CHANGE 2: The Merged & Updated Dictionary ---
FINANCIAL_DICTIONARY = {
    "dcs": "TCS",
    "tcs": "TCS",
    "enforces": "Infosys",
    "infosys": "Infosys",
    "in faucets": "Infosys",
    "in forces": "Infosys",
    "reliance industry": "Reliance Industries",
    "reliance industries": "Reliance Industries",
    "hdfc bank": "HDFC Bank",
    "icici bank": "ICICI Bank",
    "eye see eye see eye": "ICICI",
    "wipro": "Wipro",
    "bajaj finance": "Bajaj Finance",
    "state bank of india": "State Bank of India",
    "sbi": "SBI",
    "tata motors": "Tata Motors",
    "tata consultancy services": "TCS",
    "tata consultancy": "TCS",
    "hcl tech": "HCL Tech",
    "asian paints": "Asian Paints",
    "ultratech cement": "UltraTech Cement",
    "kotak mahindra": "Kotak Mahindra",
    "sun pharma": "Sun Pharma",
    "mrph": "MRF",
    "mrf": "MRF",
    "nestle": "Nestle",
    "maruti suzuki": "Maruti Suzuki",
    "adani enterprises": "Adani Enterprises",
    "data power": "Tata Power",
    "data motors": "Tata Motors",
    "data steel": "Tata Steel",
    "data consultancy": "Tata Consultancy",
    "tata pavore": "Tata Power",
    "tata pawar": "Tata Power",
    "tata pawa": "Tata Power",
    "access bank": "Axis Bank",
    "access": "Axis Bank",
    "tomato": "Zomato",
    "geo financial": "Jio Financial",
    "geo": "Jio",
}

def load_whisper_model(size: str = "base"):
    global _model
    if _model is None:
        _model = whisper.load_model(size)
    return _model

def apply_financial_correction(text: str) -> str:
    """
    Light-touch correction for the most common and consistent STT errors only.
    Does NOT attempt aggressive fuzzy matching which causes wrong corrections.
    """
    if not text:
        return text

    corrected = text
    # We now loop over the comprehensive global dictionary
    for mishear, correct in FINANCIAL_DICTIONARY.items():
        # Using \b (word boundaries) ensures we only replace whole words.
        # This prevents "access" from accidentally modifying "accessibility".
        pattern = r'\b' + re.escape(mishear) + r'\b'
        corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)

    return corrected

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
        
        # --- Whisper Prompt successfully injected here ---
        options = whisper.DecodingOptions(
            language="en", 
            fp16=False, 
            without_timestamps=True,
            prompt=FINANCE_VOCABULARY_PROMPT
        )
        
        result = whisper.decode(model, mel, options)
        try:
            os.unlink(wav_path)
        except:
            pass
            
        # Apply the post-processing
        text = result.text.strip()
        text = apply_financial_correction(text)
        return text
        
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
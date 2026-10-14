"""
voice_recognizer.py — Robust Voice Recognition for AgroRunner6
Direct Hindi, Hinglish & English Speech Recognition using Google Speech API.
Converts any incoming voice audio payload into recognized text.
"""
import speech_recognition as sr

def transcribe_audio_payload(audio_payload, language: str = 'hi-IN') -> tuple[bool, str]:
    """
    Transcribes audio dictionary from mic_recorder:
    audio_payload: {
        'bytes': bytes,
        'sample_rate': int,
        'sample_width': int,
        'format': str
    }
    """
    if not audio_payload:
        return False, "No audio payload received."

    recognizer = sr.Recognizer()
    
    if isinstance(audio_payload, dict):
        audio_bytes = audio_payload.get('bytes')
        sample_rate = audio_payload.get('sample_rate', 16000)
        sample_width = audio_payload.get('sample_width', 2)
    else:
        audio_bytes = audio_payload
        sample_rate = 16000
        sample_width = 2

    if not audio_bytes:
        return False, "Audio bytes empty."

    # 1. Primary Speech Recognition pass
    try:
        audio_data = sr.AudioData(audio_bytes, sample_rate, sample_width)
        text = recognizer.recognize_google(audio_data, language=language)
        if text and text.strip():
            return True, text.strip()
    except sr.UnknownValueError:
        pass
    except Exception as e:
        # Retry with secondary config if initial pass failed
        pass

    # 2. Secondary fallback pass
    try:
        fallback_lang = 'en-IN' if language.startswith('hi') else 'hi-IN'
        audio_data = sr.AudioData(audio_bytes, sample_rate, sample_width)
        text = recognizer.recognize_google(audio_data, language=fallback_lang)
        if text and text.strip():
            return True, text.strip()
    except Exception:
        pass

    return False, "Could not detect clear speech. Please speak closely into the microphone and try again."

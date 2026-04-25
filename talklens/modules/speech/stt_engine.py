"""
Speech Module – Speech-to-Text (STT)
──────────────────────────────────────
Uses OpenAI Whisper for accurate transcription.
Falls back to Google Speech Recognition for zero-install scenarios.
"""
from __future__ import annotations

import io
import threading
import time
import tempfile
from pathlib import Path
from typing import Optional, Callable
from loguru import logger
from config.settings import config


class STTEngine:
    """Whisper-based speech-to-text with microphone capture."""

    def __init__(self) -> None:
        self._cfg = config.speech
        self._whisper_model = None
        self._recognizer = None
        self._listening = False
        self._callback: Optional[Callable[[str], None]] = None
        self._load_engines()

    def _load_engines(self) -> None:
        # Whisper (primary)
        try:
            import whisper
            self._whisper_model = whisper.load_model(self._cfg.whisper_model)
            logger.info(f"Whisper STT loaded → model={self._cfg.whisper_model}")
        except Exception as e:
            logger.warning(f"Whisper unavailable: {e}; falling back to SpeechRecognition")

        # SpeechRecognition (fallback + microphone capture)
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._microphone = sr.Microphone(sample_rate=self._cfg.sample_rate)
        except Exception as e:
            logger.error(f"SpeechRecognition init failed: {e}")

    # ── Transcription ──────────────────────────────────────────────────────────

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe an audio file using Whisper with Google fallback."""
        if self._whisper_model:
            try:
                result = self._whisper_model.transcribe(audio_path, fp16=False)
                return result["text"].strip()
            except Exception as e:
                logger.warning(f"Whisper transcription failed (likely missing ffmpeg): {e}. Falling back to Google.")
        
        return self._sr_file(audio_path)

    def _sr_file(self, audio_path: str) -> str:
        """Fallback: transcribe using Google SR."""
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as src:
            audio = r.record(src)
        try:
            return r.recognize_google(audio)
        except Exception:
            return ""

    # ── Real-time Microphone Listening ─────────────────────────────────────────

    def start_listening(self, callback: Callable[[str], None]) -> None:
        """Start background microphone listening. callback(text) on transcript."""
        if self._recognizer is None:
            logger.error("No STT engine available.")
            return
        self._callback = callback
        self._listening = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Microphone listening started.")

    def stop_listening(self) -> None:
        self._listening = False

    def _listen_loop(self) -> None:
        import speech_recognition as sr
        logger.info("STT: Adjusting for ambient noise...")
        try:
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            logger.info(f"STT: Energy threshold set to {self._recognizer.energy_threshold}")
        except Exception as e:
            logger.warning(f"STT: Noise adjustment failed: {e}")

        while self._listening:
            try:
                logger.debug("STT: Listening for speech...")
                with self._microphone as source:
                    audio = self._recognizer.listen(source, timeout=2, phrase_time_limit=10)
                
                logger.info("STT: Audio captured, transcribing...")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio.get_wav_data())
                    tmp_path = tmp.name
                
                text = self.transcribe_file(tmp_path)
                Path(tmp_path).unlink(missing_ok=True)
                
                if text and len(text.strip()) > 1:
                    logger.success(f"STT: Transcript detected -> '{text}'")
                    if self._callback:
                        self._callback(text)
                else:
                    logger.debug("STT: No clear text detected.")
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                logger.warning(f"STT listen error: {e}")
                time.sleep(1) # prevent tight loop on error

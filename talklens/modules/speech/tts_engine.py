"""
Speech Module – Text-to-Speech (TTS)
──────────────────────────────────────
Uses pyttsx3 as primary engine (offline, zero-latency).
Designed to be non-blocking via threading.
"""
from __future__ import annotations

import threading
import queue
from loguru import logger
from config.settings import config


class TTSEngine:
    """Thread-safe, non-blocking text-to-speech engine."""

    def __init__(self) -> None:
        self._cfg = config.speech
        self._queue: queue.Queue[str] = queue.Queue()
        self._engine = None
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self) -> None:
        import platform
        import subprocess
        is_mac = platform.system() == "Darwin"

        if not is_mac:
            # Traditional pyttsx3 for non-Mac
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self._cfg.tts_rate)
                self._engine.setProperty("volume", self._cfg.tts_volume)
                logger.info("pyttsx3 TTS engine initialised for non-macOS.")
            except Exception as e:
                logger.error(f"TTS init failed: {e}")
                self._engine = None

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if is_mac:
                    # Use native macOS 'say' command - extremely reliable
                    logger.info(f"Executing macOS 'say': {text[:40]}...")
                    subprocess.run(["say", "-r", str(self._cfg.tts_rate), text], check=False)
                elif self._engine:
                    self._engine.say(text)
                    self._engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"TTS error: {e}")

    def speak(self, text: str, priority: bool = False) -> None:
        """Queue a text string for speech. priority=True clears the queue first."""
        if not text or not text.strip():
            return
        if priority:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        self._queue.put(text)
        logger.debug(f"TTS queued: '{text[:60]}…'" if len(text) > 60 else f"TTS queued: '{text}'")

    def stop(self) -> None:
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)

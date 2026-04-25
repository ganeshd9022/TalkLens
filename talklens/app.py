"""
TalkLens – Streamlit Application Entry Point
─────────────────────────────────────────────
Run: streamlit run app.py
Ready for Deployment (WebRTC & Production Optimizations)
"""
from __future__ import annotations

import sys
import os
import time
import threading
from pathlib import Path
from typing import List, Optional, Any, Dict
import queue
from loguru import logger
from dotenv import load_dotenv

# Load hardcoded API keys from .env
load_dotenv()

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
try:
    from streamlit_webrtc import AudioProcessorBase
except ImportError:
    # Older streamlit-webrtc versions — define a compatible base class
    class AudioProcessorBase:  # type: ignore
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            return frame

# ── Page Config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title     = "TalkLens – Inclusive AI",
    page_icon      = "📡",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS Injection ──────────────────────────────────────────────────────────────
css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
else:
    # Minimal inline styles if asset is missing
    st.markdown("""
    <style>
        .tl-hero { padding: 2rem; background: linear-gradient(90deg, #0d1117 0%, #161b22 100%); border-bottom: 1px solid #30363d; margin-bottom: 2rem; border-radius: 12px; }
        .tl-hero-title { font-size: 2.5rem; font-weight: 800; color: #e6edf3; }
        .tl-hero-sub { color: #8b949e; margin-top: 0.5rem; font-size: 1.1rem; }
        .mode-badge { padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
        .mode-vision { background: rgba(88, 166, 255, 0.15); color: #58a6ff; border: 1px solid rgba(88, 166, 255, 0.3); }
        .mode-sign { background: rgba(63, 185, 80, 0.15); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.3); }
        .mode-convo { background: rgba(163, 113, 247, 0.15); color: #a371f7; border: 1px solid rgba(163, 113, 247, 0.3); }
        .tl-card { background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
        .tl-card-title { font-size: 0.9rem; font-weight: 600; color: #8b949e; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05rem; }
        .sign-letter { font-size: 4rem; font-weight: 900; color: #3fb950; }
        .fps-bar { font-family: monospace; font-size: 0.8rem; color: #8b949e; }
    </style>
    """, unsafe_allow_html=True)

# ── Module Imports ────────────────────────────────────────────────────────────
from config.settings import config
from utils.helpers   import setup_logger, frame_to_rgb, resize_frame

setup_logger(config.log_level)

# ── WebRTC Configuration ──────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ══════════════════════════════════════════════════════════════════════════════
# Session State Bootstrap
# ══════════════════════════════════════════════════════════════════════════════

# ── Global State for Background Threads ──────────────────────────────────────
_active_mode = "Vision Mode"

def _init_state():
    defaults = {
        "mode":              "Vision Mode",
        "tts_engine":        None,
        "stt_engine":        None,
        "vision_detector":   None,
        "sign_recognizer":   None,
        "orchestrator":      None,
        "detections":        [],
        "scene_description": "",
        "scene_spatial":     "",
        "sign_letter":       "",
        "sign_confidence":   0.0,
        "sign_word":         "",
        "conversation":      [],
        "last_spoken_at":    0.0,
        "speech_interval":   4.0,
        "ai_responses":      [],    # recent AI answers for the vision UI
        "last_voice_q":      "",    # dedup voice questions
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# Establish global scope for background threads immediately
_active_mode = st.session_state["mode"]


# ══════════════════════════════════════════════════════════════════════════════
# Lazy Module Loaders (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Vision Models…")
def _load_vision():
    from modules.vision.detector import VisionDetector
    return VisionDetector()

@st.cache_resource(show_spinner="Loading Sign Recognition…")
def _load_sign():
    from modules.sign_language.recognizer import SignLanguageRecognizer
    return SignLanguageRecognizer()

@st.cache_resource(show_spinner="Booting Speech Engine…")
def _load_tts():
    from modules.speech.tts_engine import TTSEngine
    # Wrap in Try-Except for Headless Server Stability
    try:
        return TTSEngine()
    except Exception as e:
        st.warning(f"Audio Output (TTS) disabled on this server: {e}")
        return None

@st.cache_resource(show_spinner="Preparing STT…")
def _load_stt():
    from modules.speech.stt_engine import STTEngine
    return STTEngine()

@st.cache_resource
def _load_orchestrator():
    from modules.integration.orchestrator import TalkLensOrchestrator
    return TalkLensOrchestrator()


# ══════════════════════════════════════════════════════════════════════════════
# Video Processors (WebRTC Backends)
# ══════════════════════════════════════════════════════════════════════════════

# Global shared scene state — audio processor reads this to answer questions
_shared_scene = {"description": "", "spatial": ""}


class VisionProcessor(VideoProcessorBase):
    """Processes video frames: detects objects, describes scene, announces via TTS."""

    def __init__(self, detector, orchestrator, tts, speech_interval):
        self.detector        = detector
        self.orchestrator    = orchestrator
        self.tts             = tts
        self.speech_interval = speech_interval
        self.last_spoken_at  = 0
        self.result_queue    = queue.Queue(maxsize=5)
        self.initial_desc_triggered = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img         = frame.to_ndarray(format="bgr24")
            process_img = resize_frame(img, width=640)
            detections  = self.detector.detect(process_img)

            disp_width  = 1024
            img         = resize_frame(img, width=disp_width)
            scale       = disp_width / 640.0
            for det in detections:
                det["bbox"] = [c * scale for c in det["bbox"]]

            annotated   = self.detector.annotate(img, detections)
            description = self.detector.describe(detections)
            spatial     = self.detector.spatial_context(detections, img.shape[1])

            # Keep global scene state fresh for the audio processor
            _shared_scene["description"] = description
            _shared_scene["spatial"]     = spatial
            # Keep orchestrator context fresh
            self.orchestrator._last_scene_description = description

            now = time.time()

            # Announce environment once on startup
            if not self.initial_desc_triggered and description and description != "No objects detected.":
                self.initial_desc_triggered = True
                self.last_spoken_at = now
                self.last_announced_desc = description
                threading.Thread(
                    target=lambda: self.tts and self.tts.speak(
                        f"Environment scan complete. {description} {spatial}", priority=True),
                    daemon=True
                ).start()

            # Periodic scene update
            is_new = description != getattr(self, "last_announced_desc", "")
            if self.tts and detections and is_new and (now - self.last_spoken_at) > self.speech_interval:
                self.last_announced_desc = description
                self.last_spoken_at = now
                threading.Thread(
                    target=lambda d=description: self.tts.speak(d),
                    daemon=True
                ).start()

            # Push to UI result queue (non-blocking)
            try:
                self.result_queue.put_nowait({"detections": detections, "description": description, "spatial": spatial})
            except queue.Full:
                try: self.result_queue.get_nowait()
                except: pass
                self.result_queue.put_nowait({"detections": detections, "description": description, "spatial": spatial})

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as e:
            logger.error(f"VisionProcessor error: {e}")
            return frame


class VisionAudioProcessor(AudioProcessorBase):
    """
    Hands-free voice Q&A for blind users.
    Listens continuously via WebRTC audio — no button press needed.
    When speech is detected and silence follows, transcribes and speaks answer.
    """
    ENERGY_THRESHOLD = 300    # RMS energy to detect speech (int16 scale)
    SILENCE_LIMIT    = 30     # audio chunks of silence to mark end of speech
    MIN_SPEECH_CHUNKS= 5      # minimum chunks to be considered speech

    def __init__(self, orchestrator, tts):
        self.orchestrator   = orchestrator
        self.tts            = tts
        self.ai_queue       = queue.Queue()  # for UI log updates
        self._buf           = []
        self._silence       = 0
        self._speaking      = False
        self._processing    = False
        self._last_q        = ""
        self._sample_rate   = 48000

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            # Convert audio frame to int16 numpy
            samples = frame.to_ndarray()  # shape: (channels, samples) for planar
            self._sample_rate = frame.sample_rate

            # Flatten to mono int16
            if samples.dtype.kind == 'f':          # float (fltp)
                mono = (samples[0] * 32768).astype(np.int16)
            else:
                mono = samples.flatten().astype(np.int16)

            energy = int(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))

            if energy > self.ENERGY_THRESHOLD:
                self._speaking = True
                self._silence  = 0
                self._buf.append(mono)
            elif self._speaking:
                self._buf.append(mono)
                self._silence += 1
                if self._silence >= self.SILENCE_LIMIT:
                    if len(self._buf) >= self.MIN_SPEECH_CHUNKS and not self._processing:
                        captured = list(self._buf)
                        threading.Thread(
                            target=self._transcribe_and_answer,
                            args=(captured,),
                            daemon=True
                        ).start()
                    self._buf     = []
                    self._silence = 0
                    self._speaking = False
        except Exception as e:
            logger.warning(f"VisionAudioProcessor.recv error: {e}")
        return frame

    def _transcribe_and_answer(self, audio_chunks):
        """Transcribe captured speech and answer via TTS. Runs in background thread."""
        self._processing = True
        try:
            import wave, tempfile, speech_recognition as sr

            combined = np.concatenate(audio_chunks)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self._sample_rate)
                    wf.writeframes(combined.tobytes())
                tmp_path = tmp.name

            recognizer = sr.Recognizer()
            with sr.AudioFile(tmp_path) as src:
                audio = recognizer.record(src)
            Path(tmp_path).unlink(missing_ok=True)

            try:
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                logger.debug("STT: Speech unclear, skipping.")
                return
            except Exception as e:
                logger.warning(f"STT: Transcription error: {e}")
                return

            if not text or len(text.strip()) < 3:
                return
            if text.strip() == self._last_q:
                return  # avoid duplicate

            self._last_q = text.strip()
            logger.success(f"Voice question: '{text}'")

            # Use scene context from shared state
            scene = _shared_scene.get("description", "the environment")
            spatial = _shared_scene.get("spatial", "")
            self.orchestrator._last_scene_description = f"{scene} {spatial}"

            ans = self.orchestrator.handle_user_question(text)
            if not ans:
                ans = f"Based on what I can see: {scene}"

            logger.success(f"AI answer: '{ans[:80]}'")

            # Speak answer immediately — this is what the blind user hears
            if self.tts:
                self.tts.speak(ans, priority=True)

            # Put in queue so UI log updates on next rerun
            self.ai_queue.put({"q": text, "a": ans})

        except Exception as e:
            logger.error(f"Transcribe/Answer error: {e}")
        finally:
            self._processing = False


class SignProcessor(VideoProcessorBase):
    def __init__(self, recognizer, tts):
        self.recognizer = recognizer
        self.tts = tts
        self.result_queue = queue.Queue()
        self.hand_gone_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            img = resize_frame(img, width=480)
            letter, conf, annotated = self.recognizer.process_frame(img)
            
            # Auto-Speak Logic
            if not letter and self.recognizer.get_word():
                self.hand_gone_count += 1
                # If hand is gone for 30 frames (approx 1 second), speak and clear
                if self.hand_gone_count >= 30:
                    word = self.recognizer.get_word()
                    if self.tts:
                        self.tts.speak(word, priority=True)
                    self.recognizer.clear_word()
                    self.hand_gone_count = 0
            elif letter:
                self.hand_gone_count = 0

            self.result_queue.put({
                "letter": letter, 
                "confidence": conf,
                "word": self.recognizer.get_word()
            })
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as e:
            logger.error(f"SignProcessor error: {e}")
            return frame


class ConversationProcessor(VideoProcessorBase):
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer
        self.result_queue = queue.Queue()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            img = resize_frame(img, width=480)
            
            # Combined analysis
            letter, conf, annotated = self.recognizer.process_frame(img)
            dets = self.detector.detect(annotated)
            annotated = self.detector.annotate(annotated, dets)
            
            self.result_queue.put({
                "letter": letter,
                "det_count": len(dets)
            })

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as e:
            logger.error(f"ConversationProcessor error: {e}")
            return frame


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
<div style="text-align:center;margin-bottom:1.5rem">
  <div style="font-size:1.8rem;font-weight:800;color:#f59e0b;letter-spacing:0.05rem">TALKLENS</div>
  <div style="font-size:0.75rem;color:#94a3b8;font-weight:500;text-transform:uppercase;letter-spacing:0.1rem">Inclusive Assistant AI</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🎯 Active Mode**")
    # Ensure session state doesn't hold removed mode
    if st.session_state["mode"] == "Conversation Mode":
        st.session_state["mode"] = "Vision Mode"
        
    mode = st.radio(
        "Select Mode",
        ["Vision Mode", "Sign Language Mode"],
        index=["Vision Mode", "Sign Language Mode"].index(st.session_state["mode"]),
        label_visibility="collapsed",
    )
    if mode != st.session_state["mode"]:
        st.session_state["mode"] = mode
        # Update global tracker — background threads will pause immediately
        _active_mode = mode
        st.session_state["last_voice_q"] = ""
        st.session_state["ai_responses"] = []

        # Stop TTS and clear AI context so Vision conversation doesn't bleed into Sign mode
        tts = _load_tts()
        if tts:
            try: tts._queue.queue.clear()
            except: pass
        
        import os, platform
        if platform.system() == "Darwin":
            os.system("killall say 2>/dev/null")

        orch = _load_orchestrator()
        if orch:
            orch.clear_context()

        # Clear shared scene so old detections aren't used in new mode
        _shared_scene.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**⚙️ Settings**")

    with st.expander("Vision Settings"):
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9,
                                config.vision.confidence_threshold, 0.05)
        config.vision.confidence_threshold = conf_thresh
        speech_interval = st.slider("TTS Interval (sec)", 1, 15,
                                    int(st.session_state["speech_interval"]))
        st.session_state["speech_interval"] = float(speech_interval)

    with st.expander("Sign Language Settings"):
        sign_conf = st.slider("Sign Confidence", 0.5, 0.99,
                              config.sign.confidence_threshold, 0.05)
        config.sign.confidence_threshold = sign_conf

    with st.expander("LLM Settings"):
        enable_llm = st.toggle("Enable AI Responses (LLM)", value=config.integration.enable_llm)
        config.integration.enable_llm = enable_llm
        if enable_llm:
            provider = st.selectbox("AI Provider", ["groq", "openai", "gemini"], 
                                   index=["groq", "openai", "gemini"].index(config.integration.llm_provider))
            config.integration.llm_provider = provider
            
            if provider == "gemini":
                # Default Gemini model
                config.integration.llm_model = "gemini-1.5-flash"
                env_key = os.getenv("GOOGLE_API_KEY", "")
                label = "Google API Key (Loaded from .env)" if env_key else "Google API Key"
                g_key = st.text_input(label, type="password", value=env_key)
                if g_key: os.environ["GOOGLE_API_KEY"] = g_key
            else:
                env_var = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
                env_key = os.getenv(env_var, "")
                label = f"{provider.capitalize()} API Key (Loaded from .env)" if env_key else f"{provider.capitalize()} API Key"
                api_key = st.text_input(label, type="password", value=env_key)
                if api_key: os.environ[env_var] = api_key

    st.markdown("---")
    st.markdown("**🛠️ Diagnostics**")
    if st.button("🔊 Test Speakers", use_container_width=True):
        tts = _load_tts()
        if tts:
            tts.speak("Speaker test successful. TalkLens is ready.", priority=True)
            st.success("Test signal sent to speakers.")
        else:
            st.error("TTS Engine not available.")
    
    st.markdown("---")



# ══════════════════════════════════════════════════════════════════════════════
# Hero Header
# ══════════════════════════════════════════════════════════════════════════════

mode_badge_map = {
    "Vision Mode":       ('<span class="mode-badge mode-vision">👁 Vision</span>', "vision"),
    "Sign Language Mode":('<span class="mode-badge mode-sign">🤟 Sign Language</span>', "sign"),
    "Conversation Mode": ('<span class="mode-badge mode-convo">💬 Conversation</span>', "convo"),
}
badge_html, _ = mode_badge_map[mode]

st.markdown(f"""
<div class="tl-hero">
  <div class="tl-hero-title">👁️ TalkLens</div>
  <div class="tl-hero-sub">
    Integrated Vision &amp; Sign Language Recognition Systems
    &nbsp;&nbsp;{badge_html}
  </div>
</div>
""", unsafe_allow_html=True)



class _VoiceAssistant:
    """
    Always-on background voice listener.
    Captures speech via system mic → answers via TTS → stores in log queue.
    Completely independent of Streamlit reruns.
    """
    def __init__(self, orchestrator, tts):
        self.orchestrator = orchestrator
        self.tts          = tts
        self.log_queue    = queue.Queue()   # UI polls this for conversation log
        self._active      = True
        self._last_q      = ""
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        logger.info("VoiceAssistant: background thread started")

    def _run(self):
        try:
            import speech_recognition as sr
        except ImportError:
            logger.error("speech_recognition not installed — voice disabled")
            return

        r   = sr.Recognizer()
        r.dynamic_energy_threshold = True
        r.energy_threshold         = 400
        mic = sr.Microphone()

        # Calibrate once
        try:
            with mic as src:
                r.adjust_for_ambient_noise(src, duration=1)
            logger.info(f"VoiceAssistant: mic ready, energy={r.energy_threshold:.0f}")
        except Exception as e:
            logger.warning(f"VoiceAssistant: mic init failed — {e}")
            return

        while self._active:
            # ONLY listen if we are in Vision Mode
            if _active_mode != "Vision Mode":
                time.sleep(1)
                continue

            try:
                with mic as src:
                    audio = r.listen(src, timeout=3, phrase_time_limit=12)

                logger.debug("VoiceAssistant: audio captured, transcribing…")
                try:
                    text = r.recognize_google(audio)
                except sr.UnknownValueError:
                    continue      # silence / unclear — keep listening
                except Exception as e:
                    logger.warning(f"VoiceAssistant STT: {e}")
                    continue

                text = text.strip()
                if len(text) < 3 or text == self._last_q:
                    continue
                self._last_q = text
                logger.success(f"VoiceAssistant: heard → '{text}'")

                # Pull scene context from global shared dict
                scene   = _shared_scene.get("description", "")
                spatial = _shared_scene.get("spatial", "")
                self.orchestrator._last_scene_description = f"{scene} {spatial}".strip()

                # Get AI answer
                ans = self.orchestrator.handle_user_question(text)
                if not ans:
                    ans = f"I can see: {scene}" if scene else "I cannot see anything yet."

                logger.success(f"VoiceAssistant: answer → '{ans[:80]}'")

                # ── Speak answer immediately via TTS (works in background on macOS) ──
                if self.tts:
                    self.tts.speak(ans, priority=True)

                # Store for UI conversation log
                self.log_queue.put({"q": text, "a": ans})

            except sr.WaitTimeoutError:
                continue   # no speech in timeout window — keep looping
            except Exception as e:
                logger.warning(f"VoiceAssistant loop error: {e}")
                time.sleep(1)

    def stop(self):
        self._active = False


@st.cache_resource
def _start_voice_assistant(_orchestrator, _tts):
    """Cached — starts ONCE, persists for entire app session."""
    return _VoiceAssistant(_orchestrator, _tts)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 – VISION ASSISTANCE  (Hands-Free for Blind Users)
# ══════════════════════════════════════════════════════════════════════════════

def render_vision_mode():
    detector     = _load_vision()
    tts          = _load_tts()
    orchestrator = _load_orchestrator()

    # ── Start persistent background voice assistant ──
    va = _start_voice_assistant(orchestrator, tts)

    # Pull any new voice answers into session_state
    _refreshed = False
    while True:
        try:
            item = va.log_queue.get_nowait()
            st.session_state["ai_responses"].insert(0, {"q": item["q"], "a": item["a"]})
            if len(st.session_state["ai_responses"]) > 10:
                st.session_state["ai_responses"].pop()
            _refreshed = True
        except queue.Empty:
            break

    col_feed, col_info = st.columns([3, 2], gap="large")

    # ── RIGHT PANEL ──────────────────────────────────────────────────────────
    with col_info:
        # Status banner
        # Status banner removed per user request

        # Live scene panels
        st.markdown('<div class="tl-card-title">🔍 Live Scene</div>', unsafe_allow_html=True)
        det_placeholder     = st.empty()
        desc_placeholder    = st.empty()
        spatial_placeholder = st.empty()

        st.markdown("---")

        # Text fallback (for sighted assistants / testing)
        st.markdown('<div class="tl-card-title">⌨️ Type a Question</div>', unsafe_allow_html=True)
        with st.form("vision_q_form", clear_on_submit=True):
            q_col1, q_col2 = st.columns([5, 1])
            with q_col1:
                q_input = st.text_input("Q:", placeholder="e.g. What is in front of me?", label_visibility="collapsed")
            with q_col2:
                submitted = st.form_submit_button("➤", use_container_width=True)
                
            if submitted:
                q = q_input.strip()
                if q:
                    with st.spinner("Thinking..."):
                        ans = orchestrator.handle_user_question(q)
                    if not ans:
                        ans = f"I can see: {st.session_state.get('scene_description', 'nothing yet.')}"
                    if tts: tts.speak(ans, priority=True)
                    st.session_state["ai_responses"].insert(0, {"q": q, "a": ans})
                    st.rerun()

        # Conversation log
        if st.session_state["ai_responses"]:
            st.markdown('<div class="tl-card-title" style="margin-top:1rem">💬 Conversation</div>',
                        unsafe_allow_html=True)
            for r in st.session_state["ai_responses"]:
                st.markdown(f"""
                <div style="border-left:3px solid #3fb950;padding:0.6rem 0.8rem;
                     margin-bottom:0.4rem;background:rgba(63,185,80,0.05);border-radius:0 8px 8px 0;">
                  <div style="font-size:0.75rem;color:#8b949e;">🗣️ <b>You:</b> {r['q']}</div>
                  <div style="font-size:0.9rem;color:#e6edf3;margin-top:0.2rem;">
                    🤖 <b>TalkLens:</b> {r['a']}
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Waiting for your first question…")

        if st.button("🔊 Repeat Scene", use_container_width=True):
            desc = st.session_state.get("scene_description", "")
            if tts and desc:
                tts.speak(desc, priority=True)

    # ── LEFT PANEL – Camera Feed ──────────────────────────────────────────────
    with col_feed:
        speech_interval = st.session_state["speech_interval"]
        ctx = webrtc_streamer(
            key="vision_webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=lambda: VisionProcessor(
                detector, orchestrator, tts, speech_interval
            ),
            async_processing=True,
            media_stream_constraints={
                "video": {"width": 1280, "height": 720, "frameRate": 24},
                "audio": False,
            },
        )

        if ctx.video_processor:
            res = None
            while True:
                try:
                    res = ctx.video_processor.result_queue.get_nowait()
                except queue.Empty:
                    break

            if res:
                st.session_state["scene_description"] = res["description"]
                st.session_state["scene_spatial"]     = res.get("spatial", "")

                if res["detections"]:
                    tags = "".join(
                        f'<span style="background:#3fb95022;border:1px solid #3fb95055;'
                        f'padding:0.15rem 0.5rem;border-radius:20px;margin:2px;'
                        f'font-size:0.8rem;color:#3fb950;display:inline-block;">{d["label"]}</span>'
                        for d in res["detections"]
                    )
                    det_placeholder.markdown(f'<div class="tl-card">{tags}</div>',
                                             unsafe_allow_html=True)
                else:
                    det_placeholder.markdown(
                        '<div class="tl-card" style="color:#8b949e;">📷 Scanning…</div>',
                        unsafe_allow_html=True)

                desc_placeholder.markdown(
                    f'<div class="tl-card" style="font-size:1rem;line-height:1.6;">'
                    f'{res["description"]}</div>', unsafe_allow_html=True)
                spatial_placeholder.markdown(
                    f'<div class="tl-card" style="color:#94a3b8;font-size:0.85rem;">'
                    f'{res.get("spatial") or "No spatial info yet."}</div>',
                    unsafe_allow_html=True)

        # Auto-refresh UI to pull new voice questions/answers into the conversation log
        if ctx and ctx.state.playing:
            time.sleep(1.5)
            st.rerun()




# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 – SIGN LANGUAGE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════

def render_sign_mode():
    recognizer = _load_sign()
    tts = _load_tts()

    col_feed, col_info = st.columns([3, 2], gap="large")

    with col_info:
        st.markdown('<div class="tl-card-title">🤟 Predicted Sign</div>', unsafe_allow_html=True)
        letter_placeholder = st.empty()
        st.markdown('<div class="tl-card-title" style="margin-top:1rem">📝 Word Construction</div>', unsafe_allow_html=True)
        word_placeholder = st.empty()

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("➕ Commit", use_container_width=True):
                recognizer.commit_letter()
        with col_b2:
            if st.button("🔊 Speak", use_container_width=True):
                word = recognizer.get_word()
                if word and tts: tts.speak(word, priority=True)
            if st.button("🗑 Clear", use_container_width=True):
                recognizer.clear_word()
        
        st.markdown('<div class="tl-card-title" style="margin-top:1.5rem">🎨 Text to Sign (Synthesis)</div>', unsafe_allow_html=True)
        synth_text = st.text_input("Enter text to convert:", placeholder="Type a word...", key="synth_input")
        if synth_text:
            st.markdown(f'<div class="tl-card" style="text-align:center; padding:1rem;">', unsafe_allow_html=True)
            chars = [c for c in synth_text if c.isalpha()]
            if chars:
                # Primary: Typography based (works offline if font loaded)
                st.markdown(f'<div class="asl-font">{"".join(chars).upper()}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="color:#8b949e; letter-spacing:0.5em; font-weight:600; margin-bottom:1rem;">{" ".join(chars).upper()}</div>', unsafe_allow_html=True)
                
                # Secondary: Visual Reference Charts
                st.info("Visual Reference Charts (A-M and N-Z) for cross-check:")
                ref_col1, ref_col2 = st.columns(2)
                with ref_col1:
                    st.image("talklens/assets/asl/a_m.png", caption="ASL Reference A-M")
                with ref_col2:
                    st.image("talklens/assets/asl/n_z.png", caption="ASL Reference N-Z")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_feed:
        ctx = webrtc_streamer(
            key="sign_webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=lambda: SignProcessor(recognizer, tts),
            async_processing=True,
            media_stream_constraints={
                "video": {"width": 1280, "height": 720, "frameRate": 24},
                "audio": False
            },
        )

        if ctx.video_processor:
            res = None
            while True:
                try:
                    res = ctx.video_processor.result_queue.get_nowait()
                except queue.Empty:
                    break
            
            if res:
                letter_placeholder.markdown(f'<div class="tl-card" style="text-align:center"><div class="sign-letter">{res["letter"] or "—"}</div><div style="color:#8b949e;font-size:0.8rem">{res["confidence"]:.0%} confidence</div></div>', unsafe_allow_html=True)
                word_placeholder.markdown(f'<div class="tl-card"><span style="font-size:1.5rem; color:#3fb950; font-weight:700;">{res["word"] or "..."}</span></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 – CONVERSATION MODE
# ══════════════════════════════════════════════════════════════════════════════

def render_conversation_mode():
    detector = _load_vision()
    recognizer = _load_sign()
    tts = _load_tts()
    orchestrator = _load_orchestrator()

    col_cam, col_chat = st.columns([2, 3], gap="large")

    with col_cam:
        ctx = webrtc_streamer(
            key="convo_webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=lambda: ConversationProcessor(detector, recognizer),
            async_processing=True,
            media_stream_constraints={
                "video": {"width": 1280, "height": 720, "frameRate": 24},
                "audio": False
            },
        )
        
        status_placeholder = st.empty()
        if ctx.video_processor:
            res = None
            while True:
                try:
                    res = ctx.video_processor.result_queue.get_nowait()
                except queue.Empty:
                    break
            if res:
                status_placeholder.markdown(f'<div class="tl-card" style="padding:0.5rem; font-size:0.8rem; color:#8b949e">Sign Channel: <b style="color:#3fb950">{res["letter"] or "None"}</b> | Detected Objects: {res["det_count"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="tl-card-title" style="margin-top:1rem">⌨️ Interface</div>', unsafe_allow_html=True)
        deaf_input = st.text_input("Type Sign Message:", placeholder="Sign or type…")
        if st.button("📤 Send", use_container_width=True):
            if deaf_input:
                resp = orchestrator.handle_sign_text(deaf_input)
                st.session_state["conversation"].append({"role": "deaf", "text": deaf_input})
                if tts: tts.speak(resp, priority=True)
        
        if st.button("🧹 Reset Chat", use_container_width=True):
            st.session_state["conversation"].clear()

    with col_chat:
        st.markdown('<div class="tl-card-title">💬 Interactive Log</div>', unsafe_allow_html=True)
        msgs = st.session_state["conversation"]
        if not msgs:
            st.info("Start a conversation to see history here.")
        else:
            for m in msgs[-10:]:
                role_label = "🤟 DEAF" if m["role"] == "deaf" else "🤖 AI"
                st.markdown(f"**{role_label}**: {m['text']}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Dispatch
# ══════════════════════════════════════════════════════════════════════════════

RENDER_MAP = {
    "Vision Mode":       render_vision_mode,
    "Sign Language Mode": render_sign_mode,
}

# ── Main Dispatch ──────────────────────────────────────────────────────────────

# ── Main Dispatch ──────────────────────────────────────────────────────────────
RENDER_MAP[st.session_state["mode"]]()

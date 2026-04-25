"""
TalkLens – Centralized configuration using Pydantic BaseSettings.
All environment overrides are supported via .env file.
"""
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT_DIR / "models"
DATA_DIR   = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


class VisionConfig(BaseModel):
    yolo_model: str = "yolov8n.pt"           # nano for speed; swap to yolov8m for accuracy
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_detections: int = 20
    frame_skip: int = 1                       # process every Nth frame
    device: str = "cpu"                       # "cuda" if GPU available


class SignLanguageConfig(BaseModel):
    model_config = {"protected_namespaces": ()}   # suppress Pydantic 'model_' namespace warning
    model_path: str = str(MODEL_DIR / "sign_classifier.pth")
    labels_path: str = str(DATA_DIR / "processed" / "labels.json")
    sequence_length: int = 30                 # frames per gesture
    num_landmarks: int = 63                   # 21 hand points × 3 (x,y,z)
    num_classes: int = 26                     # A–Z ASL alphabet
    confidence_threshold: float = 0.5
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5


class SpeechConfig(BaseModel):
    whisper_model: str = "base"              # tiny/base/small/medium/large
    tts_rate: int = 150                       # words per minute
    tts_volume: float = 0.9
    sample_rate: int = 16000
    tts_engine: str = "pyttsx3"              # "pyttsx3" or "coqui"


class IntegrationConfig(BaseModel):
    llm_provider: str = "groq"               # "groq" | "openai" | "none"
    llm_model: str = "llama-3.1-8b-instant"
    context_window: int = 10                  # conversation turns to retain
    enable_llm: bool = True                  # enabled by default for AI interaction


class AppConfig(BaseModel):
    vision:      VisionConfig      = Field(default_factory=VisionConfig)
    sign:        SignLanguageConfig = Field(default_factory=SignLanguageConfig)
    speech:      SpeechConfig      = Field(default_factory=SpeechConfig)
    integration: IntegrationConfig  = Field(default_factory=IntegrationConfig)
    debug:       bool               = False
    log_level:   str               = "INFO"


# Singleton config
config = AppConfig()

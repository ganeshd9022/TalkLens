"""
TalkLens – Smoke Test Suite
────────────────────────────
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest


def test_landmark_lstm_forward():
    from modules.sign_language.model import LandmarkLSTM
    model = LandmarkLSTM(input_size=63, num_classes=26)
    x     = torch.randn(4, 30, 63)
    out   = model(x)
    assert out.shape == (4, 26), f"Expected (4,26), got {out.shape}"


def test_landmark_lstm_predict_proba():
    from modules.sign_language.model import LandmarkLSTM
    model = LandmarkLSTM()
    x     = torch.randn(1, 30, 63)
    probs = model.predict_proba(x)
    assert probs.shape == (1, 26)
    assert abs(probs.sum().item() - 1.0) < 1e-4


def test_vision_describe_empty():
    from modules.vision.detector import VisionDetector
    result = VisionDetector.describe([])
    assert result == "No objects detected."


def test_vision_describe_single():
    from modules.vision.detector import VisionDetector
    dets = [{"label": "cat", "confidence": 0.91, "bbox": [10, 10, 100, 100]}]
    result = VisionDetector.describe(dets)
    assert "cat" in result and "91%" in result


def test_fps_counter():
    from utils.helpers import FPSCounter
    import time
    fps = FPSCounter(smoothing=5)
    for _ in range(6):
        fps.tick()
        time.sleep(0.02)
    assert fps.tick() > 0


def test_conversation_memory():
    from modules.integration.orchestrator import ConversationMemory
    mem = ConversationMemory(max_turns=3)
    for i in range(10):
        mem.add("user", f"msg {i}")
    history = mem.get()
    assert len(history) <= 6   # max_turns * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

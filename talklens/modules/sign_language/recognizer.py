"""
Sign Language Recognition – MediaPipe Landmark Extractor + Real-Time Classifier
────────────────────────────────────────────────────────────────────────────────
Extracts 21-point hand landmarks from BGR frames using MediaPipe.
Feeds 30-frame sequences into the trained LSTM for letter/gesture classification.
"""
from __future__ import annotations

import json
import collections
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
from loguru import logger

from config.settings import config
from modules.sign_language.model import LandmarkLSTM


class SignLanguageRecognizer:
    """End-to-end sign language recognition for a single webcam stream."""

    # ── Minimal built-in label map (A-Z ASL) ──────────────────────────────────
    DEFAULT_LABELS = {i: chr(65 + i) for i in range(26)}   # 0→A … 25→Z

    def __init__(self) -> None:
        self._cfg  = config.sign
        self._mp   = None
        self._hands = None
        self._model: Optional[LandmarkLSTM] = None
        self._labels: Dict[int, str] = self.DEFAULT_LABELS
        self._seq: collections.deque = collections.deque(maxlen=self._cfg.sequence_length)
        self._last_prediction = ""
        self._last_confidence = 0.0
        self._hand_loss_count = 0
        self._stable_count = 0
        self._last_committed_letter = ""
        self._word_buffer: List[str] = []
        self._init_mediapipe()
        self._load_model()
        self._load_labels()

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp
            self._mp = mp
            self._mp_hands = mp.solutions.hands
            self._mp_draw  = mp.solutions.drawing_utils
            self._hands    = self._mp_hands.Hands(
                static_image_mode       = False,
                max_num_hands           = 1,
                model_complexity        = 0,
                min_detection_confidence = self._cfg.min_detection_confidence,
                min_tracking_confidence  = self._cfg.min_tracking_confidence,
            )
            logger.info("MediaPipe Hands initialised.")
        except Exception as e:
            logger.error(f"MediaPipe init failed: {e}")

    def _load_model(self) -> None:
        model_path = Path(self._cfg.model_path)
        self._model = LandmarkLSTM(
            input_size  = self._cfg.num_landmarks,
            num_classes = self._cfg.num_classes,
        )
        if model_path.exists():
            state = torch.load(str(model_path), map_location="cpu")
            self._model.load_state_dict(state)
            self._model.eval()
            logger.info(f"Sign classifier loaded from {model_path}")
        else:
            logger.warning(
                f"No trained model at {model_path}. Run the training pipeline first. "
                "Recognition will produce random outputs until then."
            )

    def _load_labels(self) -> None:
        labels_path = Path(self._cfg.labels_path)
        if labels_path.exists():
            with open(labels_path) as f:
                raw = json.load(f)
            self._labels = {int(k): v for k, v in raw.items()}
            logger.info(f"Labels loaded: {self._labels}")

    # ── Word Building ──────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Main pipeline:
          1. Extract landmarks and annotate hands in a single pass.
          2. Append to rolling sequence buffer.
          3. Classify when buffer is full.
          4. Return (letter, confidence, annotated_frame)
        """
        if self._hands is None:
            return self._last_prediction, self._last_confidence, frame

        annotated = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        landmarks = None
        if results.multi_hand_landmarks:
            self._hand_loss_count = 0
            # Annotate
            for hand_lm in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    annotated, hand_lm, self._mp_hands.HAND_CONNECTIONS
                )
            
            # Extract landmarks for classification (take first hand)
            lm = results.multi_hand_landmarks[0]
            landmarks = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            self._seq.append(landmarks)
        else:
            self._hand_loss_count += 1
            # Grace period of 10 frames before clearing
            if self._hand_loss_count > 10 and len(self._seq) > 0:
                self._seq.clear()

        prediction, confidence = "", 0.0
        if self._model and len(self._seq) == self._cfg.sequence_length:
            seq_tensor = torch.tensor(
                np.array(self._seq), dtype=torch.float32
            ).unsqueeze(0)                          # (1, 30, 63)
            
            probs = self._model.predict_proba(seq_tensor)[0]
            idx   = probs.argmax().item()
            conf  = probs[idx].item()

            if conf >= self._cfg.confidence_threshold:
                prediction  = self._labels.get(idx, f"Class_{idx}")
                confidence  = conf
                
                # Auto-Commit Logic
                if prediction == self._last_prediction:
                    self._stable_count += 1
                else:
                    self._stable_count = 1
                
                # If stable for 10 frames, commit automatically
                if self._stable_count == 10 and prediction != self._last_committed_letter:
                    self.commit_letter(prediction)
                    self._last_committed_letter = prediction
                
                self._last_prediction = prediction
                self._last_confidence = confidence
            else:
                self._stable_count = 0
        else:
            self._stable_count = 0

        # Draw overlay
        if self._last_prediction:
            self._draw_sign_overlay(annotated, self._last_prediction, self._last_confidence)

        return prediction, confidence, annotated

    @staticmethod
    def _draw_sign_overlay(frame: np.ndarray, sign: str, conf: float) -> None:
        h, w = frame.shape[:2]
        label = f"Sign: {sign} ({conf:.0%})"
        cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), -1)
        cv2.putText(
            frame, label, (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 240, 150), 2, cv2.LINE_AA,
        )

    # ── Word Building ──────────────────────────────────────────────────────────

    def commit_letter(self, letter: Optional[str] = None) -> None:
        """Append current or specific prediction to the word buffer."""
        target = letter or self._last_prediction
        if target:
            self._word_buffer.append(target)
            logger.info(f"Committed letter: {target} | Word: {self.get_word()}")

    def get_word(self) -> str:
        return "".join(self._word_buffer)

    def clear_word(self) -> None:
        self._word_buffer.clear()

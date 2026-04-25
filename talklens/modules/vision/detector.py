"""
Vision Assistance Module
────────────────────────
Runs YOLOv8 on each webcam frame and returns:
  • structured detection dicts
  • human-readable scene description for TTS
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
from loguru import logger

from config.settings import config, MODEL_DIR
from utils.helpers import draw_detections, resize_frame


class VisionDetector:
    """YOLOv8-backed real-time object detector with natural-language output."""

    def __init__(self) -> None:
        self._cfg = config.vision
        self._model = None
        self._frame_count = 0
        self._last_detections: List[Dict] = []
        self._load_model()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            model_path = MODEL_DIR / self._cfg.yolo_model
            self._model = YOLO(str(model_path))
            self._model.to(self._cfg.device)
            logger.info(f"YOLOv8 loaded → {self._cfg.yolo_model} on {self._cfg.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8: {e}")
            raise

    # ── Core Inference ─────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a BGR frame.
        Returns list of dicts: {label, confidence, bbox, class_id}
        """
        self._frame_count += 1
        if self._frame_count % self._cfg.frame_skip != 0:
            return self._last_detections       # return cached result on skipped frames

        try:
            results = self._model.track(
                source=frame,
                conf=self._cfg.confidence_threshold,
                iou=self._cfg.iou_threshold,
                max_det=self._cfg.max_detections,
                persist=True,
                verbose=False,
            )
            detections = self._parse_results(results)
            self._last_detections = detections
            return detections
        except Exception as e:
            logger.warning(f"Detection error: {e}")
            return self._last_detections

    def _parse_results(self, results) -> List[Dict]:
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                detections.append({
                    "label":      result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox":       box.xyxy[0].tolist(),
                    "class_id":   int(box.cls[0]),
                })
        return detections

    # ── Annotated Frame ────────────────────────────────────────────────────────

    def annotate(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        return draw_detections(frame, detections)

    # ── Natural Language Description ───────────────────────────────────────────

    @staticmethod
    def describe(detections: List[Dict]) -> str:
        """
        Convert detection list → concise English sentence for TTS.
        Example: "I see: a person (97%), 2 chairs, a cup (81%)"
        """
        if not detections:
            return "No objects detected."

        counts: Dict[str, int] = {}
        best_conf: Dict[str, float] = {}
        for d in detections:
            lbl = d["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
            best_conf[lbl] = max(best_conf.get(lbl, 0.0), d["confidence"])

        parts = []
        for lbl, cnt in counts.items():
            if cnt == 1:
                parts.append(f"a {lbl}")
            else:
                parts.append(f"{cnt} {lbl}s")

        return "I see: " + ", ".join(parts) + "."

    # ── Spatial Context ────────────────────────────────────────────────────────

    @staticmethod
    def spatial_context(detections: List[Dict], frame_width: int) -> str:
        """
        Add directional cues: left, center, right.
        Useful for navigation guidance for blind users.
        """
        if not detections:
            return ""

        zone = frame_width / 3
        parts = []
        for d in detections:
            cx = (d["bbox"][0] + d["bbox"][2]) / 2
            if cx < zone:
                direction = "to your left"
            elif cx < zone * 2:
                direction = "ahead"
            else:
                direction = "to your right"
            parts.append(f"{d['label']} {direction}")
        return "; ".join(parts) + "."

"""Shared utility helpers used across all TalkLens modules."""
import cv2
import numpy as np
from loguru import logger
from typing import List, Tuple
import sys
import time


def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    logger.add(
        "talklens.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        backtrace=True,
    )


def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    ratio = width / w
    return cv2.resize(frame, (width, int(h * ratio)))


def draw_detections(
    frame: np.ndarray,
    detections: List[dict],
    color: Tuple[int, int, int] = (0, 200, 100),
) -> np.ndarray:
    """Draw bounding boxes + labels on a BGR frame."""
    overlay = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['label']} {det['confidence']:.0%}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(
            overlay, (x1, y1 - text_size[1] - 8), (x1 + text_size[0] + 4, y1), color, -1
        )
        cv2.putText(
            overlay, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
        )
    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class FPSCounter:
    def __init__(self, smoothing: int = 30):
        self._times: List[float] = []
        self._smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._smoothing:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

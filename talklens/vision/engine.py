import cv2
import time

from talklens.config import VisionConfig
from talklens.vision.camera import Camera
from talklens.vision.detector import ObjectDetector


class VisionEngine:
    """
    Core vision processing loop.
    """

    def __init__(self):
        self.config = VisionConfig()
        self.camera = Camera(self.config.camera_index)
        self.detector = ObjectDetector(
            self.config.model_path,
            self.config.confidence_threshold,
        )

    def run(self):
        prev_time = 0

        try:
            while True:
                frame = self.camera.read()
                detections = self.detector.detect(frame)

                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    label = det["label"]
                    conf = det["confidence"]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # FPS Calculation
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if prev_time else 0
                prev_time = current_time

                cv2.putText(
                    frame,
                    f"FPS: {int(fps)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

                cv2.imshow("TalkLens Vision", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.camera.release()
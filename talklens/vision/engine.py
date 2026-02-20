import cv2

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
        try:
            while True:
                frame = self.camera.read()
                detections = self.detector.detect(frame)

                print("Detected:", detections)

                cv2.imshow("TalkLens Vision", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.camera.release()
from ultralytics import YOLO


class ObjectDetector:
    """
    YOLOv8 wrapper for object detection.
    """

    def __init__(self, model_path: str, confidence_threshold: float):
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        detections = []

        for box in results[0].boxes:
            confidence = float(box.conf)
            if confidence < self.conf_threshold:
                continue

            cls_id = int(box.cls)
            label = self.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2)
            })

        return detections
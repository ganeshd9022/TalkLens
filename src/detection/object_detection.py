from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, conf=0.5, verbose=False)

        detections = []

        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls)]
                x_center = float(box.xywh[0][0])  # x center of box
                detections.append({
                    "label": label,
                    "x_center": x_center
                })

        return detections

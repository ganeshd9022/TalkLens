from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        # load a small YOLO model (fast, good for beginners)
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        # run object detection on one frame
        results = self.model(frame, conf=0.5, verbose=False)

        detected_objects = []

        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls)]
                detected_objects.append(label)

        return list(set(detected_objects))

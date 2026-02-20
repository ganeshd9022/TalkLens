from dataclasses import dataclass


@dataclass
class VisionConfig:
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.4
    camera_index: int = 0
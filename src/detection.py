from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


DEFAULT_CLASSES = ["dent", "scratch", "crack/shatter", "bumper_damage", "broken_lamp"]


@dataclass
class DetectionResult:
    damage_type: str
    bbox: list[int]
    confidence: float


class DamageDetector:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.model = None
        if model_path and YOLO and Path(model_path).exists():
            self.model = YOLO(model_path)

    def predict(self, image: Image.Image) -> tuple[list[DetectionResult], str]:
        if self.model is None:
            return self._mock_predict(image), "mock"
        return self._model_predict(image), "model"

    def _mock_predict(self, image: Image.Image) -> list[DetectionResult]:
        width, height = image.size
        array = np.asarray(image.convert("L"))
        center = array[height // 4 : (3 * height) // 4, width // 4 : (3 * width) // 4]
        variance = float(center.var()) if center.size else 0.0

        if variance < 50:
            return []

        bbox = [
            max(0, width // 5),
            max(0, height // 3),
            min(width - 1, (4 * width) // 5),
            min(height - 1, (2 * height) // 3),
        ]
        damage_type = "scratch" if variance < 500 else "dent"
        confidence = 0.61 if damage_type == "scratch" else 0.73
        return [DetectionResult(damage_type=damage_type, bbox=bbox, confidence=confidence)]

    def _model_predict(self, image: Image.Image) -> list[DetectionResult]:
        output = self.model.predict(image, verbose=False)
        detections: list[DetectionResult] = []
        if not output:
            return detections

        result = output[0]
        names: dict[int, Any] = getattr(result, "names", {})
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_idx = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            detections.append(
                DetectionResult(
                    damage_type=str(names.get(cls_idx, DEFAULT_CLASSES[cls_idx % len(DEFAULT_CLASSES)])),
                    bbox=[int(value) for value in xyxy],
                    confidence=confidence,
                )
            )
        return detections

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    from torchvision import models, transforms
except Exception:
    torch = None
    models = None
    transforms = None


SEVERITY_LABELS = ["Minor", "Moderate", "Severe"]


@dataclass
class SeverityResult:
    label: str
    confidence: float


class SeverityClassifier:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.model = None
        self.transform = None
        if model_path and torch and models and transforms and Path(model_path).exists():
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(SEVERITY_LABELS))
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

    def predict(self, crop: Image.Image) -> tuple[SeverityResult, str]:
        if self.model is None or self.transform is None:
            return self._mock_predict(crop), "mock"
        return self._model_predict(crop), "model"

    def _mock_predict(self, crop: Image.Image) -> SeverityResult:
        array = np.asarray(crop.convert("L"))
        mean = float(array.mean()) if array.size else 0.0
        std = float(array.std()) if array.size else 0.0

        if std > 70:
            return SeverityResult(label="Severe", confidence=0.67)
        if mean < 110 or std > 40:
            return SeverityResult(label="Moderate", confidence=0.64)
        return SeverityResult(label="Minor", confidence=0.71)

    def _model_predict(self, crop: Image.Image) -> SeverityResult:
        tensor = self.transform(crop).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        index = int(torch.argmax(probs).item())
        return SeverityResult(label=SEVERITY_LABELS[index], confidence=float(probs[index].item()))

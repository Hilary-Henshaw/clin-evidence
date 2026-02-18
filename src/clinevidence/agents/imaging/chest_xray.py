"""Chest X-ray COVID-19 / normal classification."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

_CLASSES = ["covid19", "normal"]
_INPUT_SIZE = 150
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)


class ImagingResult(TypedDict):
    """Result of a medical imaging analysis."""

    diagnosis: str
    confidence: float
    explanation: str
    model_name: str


class _ChestXrayModel(nn.Module):
    """DenseNet-121 with custom chest X-ray classifier head."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        backbone = models.densenet121(weights=None)
        in_features: int = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.features = backbone
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)  # type: ignore[no-any-return]


class ChestXrayAnalyzer:
    """Classifies chest X-rays as covid19 or normal."""

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: _ChestXrayModel | None = None

    def _load_model(self) -> _ChestXrayModel:
        """Load model weights from disk."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Chest X-ray model weights not found at: "
                f"{self._model_path}. Download the model and "
                f"place it at the configured path "
                f"(chest_xray_model_path in settings)."
            )
        model = _ChestXrayModel(num_classes=len(_CLASSES))
        state = torch.load(
            str(self._model_path),
            map_location=self._device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        logger.info(
            "Chest X-ray model loaded",
            extra={
                "path": str(self._model_path),
                "device": str(self._device),
            },
        )
        return model

    def _get_model(self) -> _ChestXrayModel:
        """Lazy-load the model on first call."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def analyse(self, image_path: Path) -> ImagingResult:
        """Classify a chest X-ray image.

        Args:
            image_path: Path to the X-ray image file.

        Returns:
            ImagingResult with diagnosis, confidence, explanation.

        Raises:
            FileNotFoundError: If model weights are missing.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        model = self._get_model()

        image = Image.open(image_path).convert("RGB")
        tensor: torch.Tensor = _PREPROCESS(image).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        diagnosis = _CLASSES[pred_idx]

        explanation = self._build_explanation(diagnosis, confidence)

        logger.info(
            "Chest X-ray analysed",
            extra={
                "diagnosis": diagnosis,
                "confidence": round(confidence, 3),
            },
        )
        return ImagingResult(
            diagnosis=diagnosis,
            confidence=confidence,
            explanation=explanation,
            model_name="DenseNet-121 Chest X-ray",
        )

    def _build_explanation(self, diagnosis: str, confidence: float) -> str:
        """Generate a clinical explanation string."""
        conf_label = (
            "high"
            if confidence >= 0.8
            else "moderate"
            if confidence >= 0.6
            else "low"
        )
        if diagnosis == "covid19":
            return (
                f"The model detected findings consistent with "
                f"COVID-19 pneumonia with {conf_label} confidence "
                f"({confidence:.1%}). Typical findings may include "
                f"bilateral ground-glass opacities, consolidation, "
                f"and peripheral distribution. Clinical correlation "
                f"and RT-PCR confirmation is strongly recommended."
            )
        return (
            f"The model found no significant findings "
            f"consistent with COVID-19 ({conf_label} confidence: "
            f"{confidence:.1%}). The chest X-ray appearance is "
            f"within normal limits per this model. Clinical "
            f"correlation with patient history is advised."
        )

"""Brain MRI tumor classification using DenseNet-121."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from clinevidence.agents.imaging.chest_xray import ImagingResult

logger = logging.getLogger(__name__)

_CLASSES = [
    "glioma",
    "meningioma",
    "pituitary_tumor",
    "no_tumor",
]
_INPUT_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)

_EXPLANATIONS: dict[str, str] = {
    "glioma": (
        "The model detected features consistent with a glioma. "
        "Gliomas arise from glial cells and range from low-grade "
        "(WHO I-II) to high-grade (WHO III-IV, glioblastoma). "
        "Histopathological confirmation via biopsy is essential."
    ),
    "meningioma": (
        "The model detected features consistent with a meningioma. "
        "Most meningiomas are benign (WHO grade I) and arise from "
        "the meninges. Neurosurgical evaluation is recommended."
    ),
    "pituitary_tumor": (
        "The model detected features consistent with a pituitary "
        "tumor. Pituitary adenomas are typically benign; functional "
        "tumors may cause hormonal abnormalities. Endocrinology and "
        "neurosurgery referral is advised."
    ),
    "no_tumor": (
        "The model found no features consistent with a brain tumor. "
        "No significant mass lesion was detected. Clinical "
        "correlation with patient symptoms and history is advised."
    ),
}


class _BrainMRIModel(nn.Module):
    """DenseNet-121 with a custom brain tumor classifier head."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        backbone = models.densenet121(weights=None)
        in_features: int = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.features = backbone
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)  # type: ignore[no-any-return]


class BrainMRIAnalyzer:
    """Classifies brain MRI scans for tumor type."""

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: _BrainMRIModel | None = None

    def _load_model(self) -> _BrainMRIModel:
        """Load model weights from disk."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Brain MRI model weights not found at: "
                f"{self._model_path}. Download the model and "
                f"place it at the configured path "
                f"(brain_mri_model_path in settings)."
            )
        model = _BrainMRIModel(num_classes=len(_CLASSES))
        state = torch.load(
            str(self._model_path),
            map_location=self._device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        logger.info(
            "Brain MRI model loaded",
            extra={
                "path": str(self._model_path),
                "device": str(self._device),
            },
        )
        return model

    def _get_model(self) -> _BrainMRIModel:
        """Lazy-load the model on first call."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def analyse(self, image_path: Path) -> ImagingResult:
        """Classify a brain MRI image.

        Args:
            image_path: Path to the MRI image file.

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

        conf_label = (
            "high"
            if confidence >= 0.8
            else "moderate"
            if confidence >= 0.6
            else "low"
        )
        explanation = (
            f"{_EXPLANATIONS[diagnosis]} "
            f"(Model confidence: {conf_label} — "
            f"{confidence:.1%})"
        )

        logger.info(
            "Brain MRI analysed",
            extra={
                "diagnosis": diagnosis,
                "confidence": round(confidence, 3),
            },
        )
        return ImagingResult(
            diagnosis=diagnosis,
            confidence=confidence,
            explanation=explanation,
            model_name="DenseNet-121 Brain MRI",
        )

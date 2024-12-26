"""Skin lesion classification using a U-Net based model."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from clinevidence.agents.imaging.chest_xray import ImagingResult

logger = logging.getLogger(__name__)

_CLASSES = [
    "melanoma",
    "nevus",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
]
_INPUT_SIZE = 256
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)


class _ConvBlock(nn.Module):
    """Two-layer conv block with batch norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)  # type: ignore[no-any-return]


class _SkinLesionNet(nn.Module):
    """U-Net style classifier for skin lesion classification."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        # Encoder
        self.enc1 = _ConvBlock(3, 64)
        self.enc2 = _ConvBlock(64, 128)
        self.enc3 = _ConvBlock(128, 256)
        self.enc4 = _ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = _ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = _ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _ConvBlock(128, 64)

        # Output segmentation head (not used for classification)
        self.out_conv = nn.Conv2d(64, 1, 1)

        # Classification head
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        bn = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Classification via GAP on decoder features
        pooled = self.gap(d1).flatten(1)
        return self.classifier(pooled)  # type: ignore[no-any-return]


class SkinLesionAnalyzer:
    """Classifies skin lesions into seven dermatological categories."""

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: _SkinLesionNet | None = None

    def _load_model(self) -> _SkinLesionNet:
        """Load model weights from a checkpoint file."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Skin lesion model weights not found at: "
                f"{self._model_path}. Download the model and "
                f"place it at the configured path "
                f"(skin_lesion_model_path in settings)."
            )
        model = _SkinLesionNet(num_classes=len(_CLASSES))
        checkpoint = torch.load(
            str(self._model_path),
            map_location=self._device,
            weights_only=True,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        logger.info(
            "Skin lesion model loaded",
            extra={
                "path": str(self._model_path),
                "device": str(self._device),
            },
        )
        return model

    def _get_model(self) -> _SkinLesionNet:
        """Lazy-load the model on first call."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def analyse(self, image_path: Path) -> ImagingResult:
        """Classify a skin lesion image.

        Args:
            image_path: Path to the dermoscopy image.

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
            "Skin lesion analysed",
            extra={
                "diagnosis": diagnosis,
                "confidence": round(confidence, 3),
            },
        )
        return ImagingResult(
            diagnosis=diagnosis,
            confidence=confidence,
            explanation=explanation,
            model_name="UNet Skin Lesion Classifier",
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
        clinical_notes = {
            "melanoma": (
                "Melanoma is a potentially life-threatening skin "
                "cancer. Urgent dermatological review and "
                "dermoscopic evaluation are recommended. "
                "Excisional biopsy may be indicated."
            ),
            "nevus": (
                "Features are consistent with a benign melanocytic "
                "nevus. Routine follow-up and monitoring for "
                "changes (ABCDE criteria) are recommended."
            ),
            "basal_cell_carcinoma": (
                "Basal cell carcinoma is the most common skin "
                "cancer. While rarely metastatic, treatment "
                "is recommended. Dermatology referral advised."
            ),
            "actinic_keratosis": (
                "Actinic keratosis is a pre-malignant lesion "
                "associated with UV exposure. Treatment to prevent "
                "progression to squamous cell carcinoma is advised."
            ),
            "benign_keratosis": (
                "Features consistent with a benign keratosis "
                "(seborrhoeic keratosis or similar). "
                "No treatment required unless symptomatic."
            ),
            "dermatofibroma": (
                "Dermatofibromas are benign fibrous nodules. "
                "No treatment is required unless causing symptoms "
                "or cosmetic concern."
            ),
            "vascular_lesion": (
                "Vascular lesion detected. This may include "
                "haemangiomas, pyogenic granulomas, or other "
                "vascular anomalies. Dermatology review advised."
            ),
        }
        note = clinical_notes.get(
            diagnosis, "Dermatological review is recommended."
        )
        return f"{note} (Model confidence: {conf_label} — {confidence:.1%})"

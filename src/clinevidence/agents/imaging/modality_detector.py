"""Medical image modality detection using GPT-4o vision."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from typing_extensions import TypedDict

from clinevidence.settings import Settings

logger = logging.getLogger(__name__)

_VALID_TYPES = frozenset(
    {
        "BRAIN_MRI",
        "CHEST_XRAY",
        "SKIN_LESION",
        "OTHER",
        "NON_MEDICAL",
    }
)

_DETECTION_PROMPT = """\
You are a medical imaging expert. Classify the provided image
into exactly one of the following categories:

- BRAIN_MRI: MRI scans of the brain or head
- CHEST_XRAY: X-ray images of the chest or thorax
- SKIN_LESION: Dermatological images of skin conditions
- OTHER: Other medical images (CT, ultrasound, ECG, etc.)
- NON_MEDICAL: Non-medical images, photographs, documents

Respond with a JSON object only, no markdown, no explanation:
{
  "image_type": "<TYPE>",
  "reasoning": "<brief clinical reasoning>",
  "confidence": <float between 0.0 and 1.0>
}
"""


class ModalityDetectionResult(TypedDict):
    """Result of medical image modality detection."""

    image_type: str
    reasoning: str
    confidence: float


class ModalityDetector:
    """Classifies medical images by modality using GPT-4o."""

    def __init__(self, settings: Settings) -> None:
        self._llm = AzureChatOpenAI(
            azure_deployment=settings.llm_deployment,
            azure_endpoint=settings.azure_endpoint,
            api_key=settings.openai_api_key,
            api_version=settings.openai_api_version,
            temperature=0.0,
        )

    def detect(self, image_path: Path) -> ModalityDetectionResult:
        """Detect the medical imaging modality.

        Args:
            image_path: Path to the image file.

        Returns:
            ModalityDetectionResult with image_type, reasoning,
            and confidence.

        Raises:
            ValueError: If the image path does not exist.
        """
        if not image_path.exists():
            raise ValueError(f"Image path does not exist: {image_path}")

        b64_image, mime_type = self._encode_image(image_path)

        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": _DETECTION_PROMPT,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (f"data:{mime_type};base64,{b64_image}"),
                        "detail": "low",
                    },
                },
            ]
        )

        try:
            response = self._llm.invoke([msg])
            result = self._parse_response(str(response.content))
        except Exception:
            logger.warning(
                "Modality detection LLM call failed, defaulting to OTHER",
                exc_info=True,
            )
            result = ModalityDetectionResult(
                image_type="OTHER",
                reasoning="Detection failed",
                confidence=0.0,
            )

        logger.info(
            "Modality detected",
            extra={
                "image": str(image_path),
                "type": result["image_type"],
                "confidence": result["confidence"],
            },
        )
        return result

    def _encode_image(self, image_path: Path) -> tuple[str, str]:
        """Base64-encode the image and determine MIME type."""
        with image_path.open("rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        suffix = image_path.suffix.lstrip(".").lower()
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
        return b64, mime

    def _parse_response(self, raw: str) -> ModalityDetectionResult:
        """Parse LLM JSON response into a result dict."""
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            data: dict[str, object] = json.loads(clean)
            image_type = str(data.get("image_type", "OTHER")).upper()
            if image_type not in _VALID_TYPES:
                image_type = "OTHER"
            return ModalityDetectionResult(
                image_type=image_type,
                reasoning=str(data.get("reasoning", "")),
                confidence=float(str(data.get("confidence", 0.5))),
            )
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Could not parse modality detection JSON",
                extra={"raw": raw[:200]},
            )
            return ModalityDetectionResult(
                image_type="OTHER",
                reasoning="JSON parse failed",
                confidence=0.0,
            )

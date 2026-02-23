"""Imaging router directing images to the correct analyser."""

from __future__ import annotations

import logging
from pathlib import Path

from clinevidence.agents.imaging.brain_mri import (
    BrainMRIAnalyzer,
)
from clinevidence.agents.imaging.chest_xray import (
    ChestXrayAnalyzer,
    ImagingResult,
)
from clinevidence.agents.imaging.skin_lesion import (
    SkinLesionAnalyzer,
)
from clinevidence.settings import Settings

logger = logging.getLogger(__name__)

_VALID_MODALITIES = frozenset({"BRAIN_MRI", "CHEST_XRAY", "SKIN_LESION"})


class ImagingRouter:
    """Routes images to the appropriate specialist analyser."""

    def __init__(self, settings: Settings) -> None:
        self._brain_mri = BrainMRIAnalyzer(settings.brain_mri_model_path)
        self._chest_xray = ChestXrayAnalyzer(settings.chest_xray_model_path)
        self._skin_lesion = SkinLesionAnalyzer(settings.skin_lesion_model_path)

    def route_and_analyse(
        self,
        image_path: Path,
        image_type: str,
    ) -> ImagingResult:
        """Route an image to the correct analyser and return result.

        Args:
            image_path: Path to the medical image.
            image_type: Detected modality string.

        Returns:
            ImagingResult from the appropriate analyser.

        Raises:
            ValueError: If image_type is not a supported modality.
        """
        modality = image_type.upper()
        if modality not in _VALID_MODALITIES:
            raise ValueError(
                f"Unsupported imaging modality: '{image_type}'. "
                f"Supported: {sorted(_VALID_MODALITIES)}"
            )

        logger.info(
            "Routing image for analysis",
            extra={
                "modality": modality,
                "image": str(image_path),
            },
        )

        if modality == "BRAIN_MRI":
            return self._brain_mri.analyse(image_path)
        if modality == "CHEST_XRAY":
            return self._chest_xray.analyse(image_path)
        return self._skin_lesion.analyse(image_path)

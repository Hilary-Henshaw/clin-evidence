"""Structured content extraction from clinical documents."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {".pdf", ".docx", ".pptx", ".html"}


class DocumentExtractor:
    """Extracts structured content from clinical documents."""

    def __init__(self, image_scale: float = 2.0) -> None:
        self._image_scale = image_scale
        self._converter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        """Build a Docling converter with full pipeline options."""
        pipeline_opts = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            do_formula_enrichment=True,
            images_scale=self._image_scale,
            generate_page_images=True,
            generate_picture_images=True,
        )
        pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE  # type: ignore[attr-defined]
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_opts
                )
            }
        )

    def extract(self, source: Path) -> tuple[str, list[Path]]:
        """Extract markdown and images from a document.

        Args:
            source: Path to the document file.

        Returns:
            Tuple of (markdown_text, list_of_image_paths).

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the file format is unsupported.
        """
        if not source.exists():
            raise FileNotFoundError(f"Document not found: {source}")
        if source.suffix.lower() not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{source.suffix}'. "
                f"Supported: {_SUPPORTED_FORMATS}"
            )

        logger.info(
            "Extracting document",
            extra={
                "path": str(source),
                "size_kb": source.stat().st_size // 1024,
            },
        )
        t0 = time.perf_counter()

        result = self._converter.convert(str(source))
        markdown = result.document.export_to_markdown()

        images: list[Path] = []
        for element, _ in result.document.iterate_items():
            if hasattr(element, "image") and element.image:
                img_path = Path(str(element.image.uri))
                if img_path.exists():
                    images.append(img_path)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Document extracted",
            extra={
                "path": str(source),
                "images_found": len(images),
                "elapsed_s": round(elapsed, 2),
            },
        )
        return markdown, images

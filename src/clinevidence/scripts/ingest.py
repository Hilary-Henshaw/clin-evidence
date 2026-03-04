"""CLI script for ingesting clinical documents into the KB."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _collect_pdfs(source: Path) -> list[Path]:
    """Return all PDF files at the given path."""
    if source.is_file():
        if source.suffix.lower() == ".pdf":
            return [source]
        return []
    if source.is_dir():
        return sorted(source.glob("**/*.pdf"))
    return []


def ingest_files(
    files: list[Path],
    collection_override: str | None = None,
) -> int:
    """Ingest a list of PDF files into the knowledge base.

    Args:
        files: List of PDF file paths to ingest.
        collection_override: Optional collection name override.

    Returns:
        Exit code: 0 on success, 1 on any failure.
    """
    from clinevidence.agents.orchestrator import (
        WorkflowOrchestrator,
    )
    from clinevidence.settings import get_settings

    settings = get_settings()
    if collection_override:
        settings.kb_collection_name = collection_override

    logger.info(
        "Initialising orchestrator for ingestion",
        extra={"file_count": len(files)},
    )
    orchestrator = WorkflowOrchestrator(settings)
    kb = orchestrator.knowledge_base

    total_chunks = 0
    total_images = 0
    failed: list[Path] = []

    for i, file_path in enumerate(files, start=1):
        logger.info(f"Processing file {i}/{len(files)}: {file_path}")
        try:
            result = kb.ingest(file_path)
            total_chunks += result["chunks_indexed"]
            total_images += result["images_processed"]
            print(
                f"  [{i}/{len(files)}] {file_path.name}: "
                f"{result['chunks_indexed']} chunks, "
                f"{result['images_processed']} images "
                f"({result['elapsed_s']:.1f}s)"
            )
        except FileNotFoundError as exc:
            logger.error(f"File not found: {file_path} — {exc}")
            failed.append(file_path)
        except ValueError as exc:
            logger.error(f"Unsupported format: {file_path} — {exc}")
            failed.append(file_path)
        except Exception:
            logger.error(
                f"Ingestion failed for {file_path}",
                exc_info=True,
            )
            failed.append(file_path)

    print("\n" + "=" * 50)
    print("Ingestion complete:")
    print(f"  Files processed : {len(files) - len(failed)}")
    print(f"  Files failed    : {len(failed)}")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Total images    : {total_images}")

    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")
        return 1

    return 0


def main() -> None:
    """CLI entry point for document ingestion."""
    _configure_logging()
    parser = argparse.ArgumentParser(
        prog="clinevidence-ingest",
        description=(
            "Ingest clinical documents (PDF) into the "
            "ClinEvidence knowledge base."
        ),
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help=("Path to a PDF file or directory of PDF files to ingest."),
    )
    parser.add_argument(
        "--collection",
        default=None,
        type=str,
        help="Override the Qdrant collection name.",
    )
    args = parser.parse_args()

    source: Path = args.path
    if not source.exists():
        logger.error(f"Path does not exist: {source}")
        sys.exit(1)

    files = _collect_pdfs(source)
    if not files:
        logger.error(f"No PDF files found at: {source}")
        sys.exit(1)

    logger.info(f"Found {len(files)} PDF file(s) to ingest")
    exit_code = ingest_files(files, args.collection)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

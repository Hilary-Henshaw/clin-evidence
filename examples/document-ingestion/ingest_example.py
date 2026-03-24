"""Document ingestion example for ClinEvidence."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger(__name__)


def verify_environment() -> bool:
    """Check that required environment variables are set."""
    import os

    required = [
        "deployment_name",
        "azure_endpoint",
        "openai_api_key",
        "embedding_deployment_name",
        "embedding_azure_endpoint",
        "embedding_openai_api_key",
        "TAVILY_API_KEY",
        "ELEVEN_LABS_API_KEY",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(
            f"Missing environment variables: "
            f"{', '.join(missing)}"
        )
        print("Copy .env.example to .env and fill in values.")
        return False
    return True


def run_ingestion(path: Path) -> int:
    """Run document ingestion and return exit code."""
    from clinevidence.scripts.ingest import (
        _collect_pdfs,
        ingest_files,
    )

    files = _collect_pdfs(path)
    if not files:
        print(f"No PDF files found at: {path}")
        return 1

    print(f"Found {len(files)} PDF file(s) to ingest\n")
    return ingest_files(files)


def verify_ingestion() -> None:
    """Check the knowledge base is ready after ingestion."""
    try:
        from clinevidence.agents.orchestrator import (
            WorkflowOrchestrator,
        )
        from clinevidence.settings import get_settings

        settings = get_settings()
        orchestrator = WorkflowOrchestrator(settings)
        is_ready = orchestrator.knowledge_base.is_ready

        print(f"\nKnowledge base ready: {is_ready}")
        if is_ready:
            print(
                "Documents are indexed and ready for queries."
            )
        else:
            print(
                "Knowledge base not ready. "
                "Check ingestion logs."
            )
    except Exception as exc:
        logger.error(
            f"Could not verify knowledge base: {exc}"
        )


def main() -> None:
    """Entry point for the ingestion example."""
    parser = argparse.ArgumentParser(
        description="ClinEvidence document ingestion example"
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to a PDF file or directory of PDFs",
    )
    args = parser.parse_args()

    if not verify_environment():
        sys.exit(1)

    source: Path = args.path
    if not source.exists():
        print(f"Path does not exist: {source}")
        sys.exit(1)

    print("ClinEvidence Document Ingestion Example")
    print("=" * 50)

    exit_code = run_ingestion(source)

    if exit_code == 0:
        verify_ingestion()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

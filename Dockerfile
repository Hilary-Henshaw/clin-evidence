FROM python:3.11-slim

LABEL maintainer="ClinEvidence Team" \
      description="Clinical Evidence Retrieval & Diagnostic Support for ICU Teams" \
      version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/

RUN mkdir -p \
    uploads/sessions \
    uploads/images \
    uploads/audio \
    data/qdrant_db \
    data/docs_db \
    data/parsed_docs \
    data/raw \
    models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "clinevidence.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]

# Changelog

All notable changes to ClinEvidence are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2025-01-01

### Added

#### Core Architecture
- LangGraph-based `WorkflowOrchestrator` with stateful
  multi-agent routing and human-in-the-loop interrupts
- `WorkflowState` TypedDict shared across all agent nodes
- MemorySaver checkpointer for per-session state persistence

#### Safety System
- `SafetyFilter` with dual-layer input and output checking
- Fail-open input safety (allows on LLM error)
- Fail-closed output safety (blocks on LLM error)
- Blocks: harmful content, prompt injection, PII, self-harm,
  non-medical requests, dangerous procedures

#### RAG Pipeline
- `KnowledgeBase` orchestrating the full 5-step pipeline
- `DocumentExtractor` using Docling with OCR, table structure
  recognition, and formula enrichment
- `DocumentFormatter` with GPT-4o vision image summarisation
  and LLM-guided semantic chunking
- `KnowledgeStore` with hybrid BM25 + dense vector search
  via Qdrant (local file or remote cloud)
- `QueryEnricher` for medical synonym and ICD code expansion
- `ResultRanker` using `cross-encoder/ms-marco-TinyBERT-L-6`
- `AnswerSynthesizer` with numbered citations and confidence
  scoring based on query-term coverage

#### Web Evidence Search
- `TavilySearchClient` for real-time clinical web search
- `PubMedSearchClient` using NCBI E-utilities REST API
- `EvidenceSearcher` combining both with deduplication
- `EvidenceSearchProcessor` with LLM synthesis and context

#### Medical Imaging
- `ModalityDetector` using GPT-4o vision for image
  classification: BRAIN_MRI, CHEST_XRAY, SKIN_LESION, OTHER
- `BrainMRIAnalyzer`: DenseNet-121, 4-class tumour classifier
- `ChestXrayAnalyzer`: DenseNet-121, COVID-19/normal
- `SkinLesionAnalyzer`: U-Net, 7-class dermoscopy classifier
- `ImagingRouter` dispatching to the correct analyser
- All imaging analyses require human validation

#### Voice Interface
- `POST /v1/transcribe`: Eleven Labs speech-to-text
- `POST /v1/synthesize`: Eleven Labs text-to-speech with
  configurable voice ID

#### API Layer
- FastAPI application with OpenAPI documentation
- `POST /v1/chat`: Clinical query endpoint
- `POST /v1/upload`: Medical image upload endpoint
- `POST /v1/validate`: Human validation decision endpoint
- `GET /health`: Service health and readiness check
- Signed session cookies via itsdangerous
- Rate limiting with slowapi (30 req/min per IP)
- Prometheus metrics via prometheus-fastapi-instrumentator
- Request tracing middleware (X-Request-ID, X-Processing-Time)
- CORS middleware

#### Infrastructure
- Docker and Docker Compose configuration
- Qdrant container in Docker Compose with health checks
- Dockerfile with ffmpeg, OpenCV dependencies
- GitHub Actions CI: lint, type check, test, Docker build
- GitHub Actions release: GHCR push on version tags
- Dependabot configuration for pip and GitHub Actions

#### Documentation
- Architecture guide with ASCII workflow diagram
- Configuration reference with all environment variables
- API reference with curl examples
- Deployment guide (Docker Compose, plain Docker, manual)
- Troubleshooting guide for common issues
- Developer guide with contribution patterns

#### CLI
- `clinevidence-ingest` CLI for single file or directory
  batch ingestion with progress output

[1.0.0]: https://github.com/clinevidence/clinevidence/releases/tag/v1.0.0

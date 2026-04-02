<p align="center"><img src="assets/banner.png" alt="ClinEvidence" width="800" /></p>

<h1 align="center">ClinEvidence</h1>
<p align="center">Clinical evidence retrieval and diagnostic support for ICU teams</p>

<p align="center">
  <a href="https://github.com/Hilary-Henshaw/clin-evidence/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Hilary-Henshaw/clin-evidence/ci.yml?branch=main&label=CI&logo=github" alt="CI Status" /></a>
  <a href="https://github.com/Hilary-Henshaw/clin-evidence/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/coverage-48%25%2B-yellow?logo=pytest" alt="Test Coverage" /></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/version-1.0.0-green" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="MIT License">
</p>

<p align="center">
  <a href="#what-it-does">What it does</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#api-reference">API reference</a> •
  <a href="#cli-reference">CLI reference</a> •
  <a href="#design-decisions">Design decisions</a> •
  <a href="#security">Security</a> •
  <a href="#known-limitations">Known limitations</a> •
  <a href="#local-development">Local development</a> •
  <a href="#license">License</a>
</p>

---

## What it does

ClinEvidence answers clinical questions in real time by combining three evidence sources:

1. **Local knowledge base** - your institution's uploaded clinical guidelines, protocols, and literature, searched with hybrid BM25 + dense vector retrieval and reranked by a cross-encoder
2. **Live evidence search** - Tavily web search and PubMed/MEDLINE queries triggered when local KB confidence falls below threshold
3. **Medical image analysis** - DenseNet and U-Net models for brain MRI (glioma/meningioma/pituitary/normal), chest X-ray (COVID-19/normal), and skin lesion classification, all gated behind a human validation step before results are returned

All queries pass through a dual-layer safety filter (input blocked before processing; output blocked before returning) using GPT-4o semantic classification. Voice I/O is available via Eleven Labs speech-to-text and text-to-speech.

The system is designed for ICU and critical care teams that need rapid, cited, evidence-based answers at the point of care. It is not a diagnostic device and all responses carry a mandatory medical disclaimer.

---

## Architecture

<p align="center">
  <img src="assets/architecture.png" alt="ClinEvidence architecture" width="760" />
</p>

The workflow is orchestrated by a LangGraph state machine with ten nodes and conditional routing:

```
HTTP client
    │
    ▼
FastAPI app  ──────────────────────────────────────────────────────
    │                                                              │
    ▼                                                              │
WorkflowOrchestrator (LangGraph)                           Prometheus
    │                                                           metrics
    ├─ assess_input ──► SafetyFilter (input)
    │
    ├─ select_agent ──► LLM router
    │                      │
    │         ┌────────────┼────────────┬────────────┐
    │         ▼            ▼            ▼            ▼
    │   ConversationAgent  RAG      EvidenceSearch  ImagingRouter
    │         │            │            │                │
    │         │      KnowledgeStore  Tavily + PubMed    ├─ BrainMRIAnalyzer
    │         │         (Qdrant)                        ├─ ChestXrayAnalyzer
    │         │                                         └─ SkinLesionAnalyzer
    │         │                                               │
    │         └────────────────┬────────────────────────────-┘
    │                          │
    ├─ check_validation ◄──────┘
    │       │
    │  (imaging only)
    │       ▼
    │  await_validation  ◄── POST /v1/validate (human approval)
    │       │
    ▼       ▼
    apply_output_safety ──► SafetyFilter (output)
    │
    ▼
HTTP response
```

**External services:** Azure OpenAI (GPT-4o for all LLM calls + embeddings), Qdrant (vector store), Eleven Labs (speech), Tavily (web search), NCBI E-utilities (PubMed).

---

## Quick start

### Docker Compose (recommended)

```bash
git clone https://github.com/clinevidence/clinevidence.git
cd clinevidence

cp .env.example .env
# Edit .env with your API keys -- see Configuration below

docker compose up -d
```

The API is available at `http://localhost:8000`. Qdrant runs on `http://localhost:6333`.

### Manual install

Requires Python 3.11+ and [ffmpeg](https://ffmpeg.org/download.html).

```bash
pip install -e ".[dev]"
cp .env.example .env
# Edit .env

clinevidence          # starts uvicorn on :8000
```

### Ingest your first document

```bash
clinevidence-ingest --path ./data/raw/guidelines.pdf
```

The ingestion pipeline extracts text and images, summarizes images with GPT-4o, chunks the content semantically, embeds it, and stores it in Qdrant. Large PDFs may take several minutes.

### Send a query

```bash
curl -s -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the recommended tidal volume for ARDS patients?", "session_id": "demo-session"}' | jq
```

---

## Configuration

All settings are read from environment variables or a `.env` file. Copy `.env.example` to get started.

### Required

| Variable | Description |
|---|---|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key (used for all LLM and embedding calls) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployment name for your GPT-4o model |
| `AZURE_EMBEDDING_DEPLOYMENT_NAME` | Deployment name for your embedding model |
| `AZURE_EMBEDDING_ENDPOINT` | Azure OpenAI endpoint for embeddings (may differ) |
| `AZURE_EMBEDDING_API_KEY` | API key for embedding endpoint |
| `ELEVEN_LABS_API_KEY` | Eleven Labs key for speech I/O |
| `TAVILY_API_KEY` | Tavily key for web evidence search |
| `SESSION_SECRET_KEY` | Secret for signing session cookies |

Generate a session secret:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### Optional

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | *(local file)* | Remote Qdrant URL; omit to use local `./data/qdrant_db` |
| `QDRANT_API_KEY` | *(none)* | API key for remote Qdrant Cloud |
| `HUGGINGFACE_TOKEN` | *(none)* | HuggingFace token for gated models |
| `HOST` | `0.0.0.0` | Uvicorn bind address |
| `PORT` | `8000` | Uvicorn port |
| `RATE_LIMIT_PER_MINUTE` | `30` | Requests per IP per minute |
| `MAX_UPLOAD_BYTES` | `5242880` | Max image upload size (5 MB) |

### Knowledge base tuning

| Variable | Default | Description |
|---|---|---|
| `KB_CHUNK_SIZE` | `512` | Characters per document chunk |
| `KB_RETRIEVAL_TOP_K` | `5` | Candidates fetched from Qdrant before reranking |
| `KB_RERANKER_TOP_K` | `3` | Final results after cross-encoder reranking |
| `KB_MIN_CONFIDENCE` | `0.40` | Confidence threshold; below this triggers web evidence search |
| `KB_MAX_CONTEXT_TOKENS` | `8192` | Max tokens passed to answer synthesizer |

### Medical imaging model paths

Imaging model weights are not bundled. Obtain them separately and place them at:

| Variable | Default path | Notes |
|---|---|---|
| `BRAIN_MRI_MODEL_PATH` | `./models/brain_mri.pth` | DenseNet-121, 4-class |
| `CHEST_XRAY_MODEL_PATH` | `./models/chest_xray.pth` | DenseNet-121, 2-class |
| `SKIN_LESION_MODEL_PATH` | `./models/skin_lesion.pth.tar` | U-Net, 7-class |

See [docs/deployment.md](docs/deployment.md) for model acquisition instructions. If a model file is missing, the corresponding imaging endpoint returns an error; other endpoints are unaffected.

---

## API reference

Full documentation with request/response schemas is available at `http://localhost:8000/docs` (Swagger) or `http://localhost:8000/redoc`.

### POST /v1/chat

Submit a clinical text query.

**Request**
```json
{
  "message": "What are the diagnostic criteria for septic shock?",
  "session_id": "your-session-uuid"
}
```

**Response**
```json
{
  "message": "Septic shock is defined as sepsis with...",
  "agent_used": "knowledge_base",
  "sources": [
    {"title": "Surviving Sepsis Campaign 2021", "url": "...", "score": 0.92}
  ],
  "confidence": 0.87,
  "processing_time_ms": 1240
}
```

The session cookie `clinevidence_session` is set on the first request and must be present on subsequent requests in the same conversation.

### POST /v1/upload

Upload a medical image for analysis.

**Request:** `multipart/form-data` with fields `file` (image) and `session_id`.

Accepted formats: `png`, `jpg`, `jpeg`. Max size: 5 MB.

**Response**
```json
{
  "image_type": "BRAIN_MRI",
  "analysis": "DenseNet-121 classification: glioma (confidence: 0.84)...",
  "confidence": 0.84,
  "requires_validation": true
}
```

When `requires_validation` is `true`, the workflow is paused. Submit a validation decision before the response is returned to the session.

### POST /v1/validate

Resume a paused imaging workflow after human review.

**Request**
```json
{
  "session_id": "your-session-uuid",
  "approved": true,
  "feedback": "Confirmed, consistent with glioma presentation"
}
```

### POST /v1/transcribe

Convert audio to text.

**Request:** `multipart/form-data` with `file` field. Accepted: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`.

**Response**
```json
{"text": "What is the SOFA score threshold for organ dysfunction?"}
```

### POST /v1/synthesize

Convert text to speech.

**Request**
```json
{"text": "The SOFA score threshold is 2 or more points above baseline."}
```

**Response**
```json
{"audio_url": "/uploads/audio/response_abc123.mp3"}
```

### GET /health

Returns `200` when the app is up and the knowledge base collection is reachable.

```json
{"status": "healthy", "knowledge_base_ready": true}
```

### GET /metrics

Prometheus metrics endpoint.

---

## CLI reference

### `clinevidence`

Start the API server.

```bash
clinevidence                         # binds to HOST:PORT from env (default 0.0.0.0:8000)
```

### `clinevidence-ingest`

Ingest documents into the knowledge base.

```bash
clinevidence-ingest --path ./data/raw/guidelines.pdf
clinevidence-ingest --path ./data/raw/                # batch ingest entire directory
clinevidence-ingest --path ./file.pdf --collection custom_collection
```

| Flag | Description |
|---|---|
| `--path` | Path to a PDF file or directory of PDFs (required) |
| `--collection` | Qdrant collection name (default: `clinical_knowledge`) |

Ingestion is synchronous and blocks until complete. For large directories, run it as a background process or scheduled job.

---

## Design decisions

**LangGraph for workflow orchestration.** The human-in-the-loop requirement for imaging results needed a durable interrupt mechanism. LangGraph's `interrupt()` pauses the state machine mid-graph and `MemorySaver` checkpoints the state so the workflow can resume when `/v1/validate` is called. The trade-off is that MemorySaver is in-process only, which enforces single-worker deployment.

**Dual-layer safety with asymmetric failure modes.** Input filtering fails open (an LLM error allows the request through) because blocking all requests during a transient outage is worse for patient care. Output filtering fails closed (an LLM error blocks the response) because returning unvalidated output to a clinician is the higher-stakes failure. Both use GPT-4o at `temperature=0.0` for deterministic classification.

**Hybrid Qdrant search.** Dense vector retrieval alone misses exact clinical term matches (drug names, ICD codes, lab values). BM25 sparse retrieval alone misses semantic similarity. The hybrid mode combines both signals before the cross-encoder reranks the candidates.

**Cross-encoder reranking after retrieval.** Bi-encoder similarity (used during retrieval) is fast but less accurate for relevance. The cross-encoder (`ms-marco-TinyBERT-L-6`) jointly encodes the query and each candidate, producing better relevance scores at the cost of running N forward passes. Retrieval fetches `top_k=5`; reranking selects `top_3`.

**LLM-guided semantic chunking.** Fixed-size chunking splits clinical text at arbitrary boundaries, separating context like "administer X" from "contraindicated in Y" on the same guideline line. The chunker asks GPT-4o to identify logical content boundaries before splitting.

**Temperature by agent role.** The routing decision uses `temperature=0.1` for near-deterministic agent selection. The conversation agent uses `0.7` for natural variation. The answer synthesizer uses `0.3` to balance accuracy with readability. The safety filter and chunker use `0.0` for reproducibility.

**Lazy model loading for imaging.** Imaging models are several hundred MB each. Loading all three at startup would significantly delay service readiness and waste memory if only text queries are used. Each analyzer loads its weights on the first inference call.

---

## Security

**Session management.** Sessions are identified by a signed cookie (`clinevidence_session`) using itsdangerous with `SESSION_SECRET_KEY`. The cookie is HttpOnly and SameSite=Lax. Every API endpoint validates that the `session_id` in the request body matches the value in the signed cookie; a mismatch returns HTTP 403.

**Production cookie flag.** The session cookie is currently set with `secure=False` for local development. Before deploying behind HTTPS, set `secure=True` in `src/clinevidence/dependencies.py:96` or add a `SECURE_COOKIES=true` env setting.

**Rate limiting.** slowapi enforces 30 requests/minute per source IP. This limit is shared across all users behind the same IP (e.g., corporate NAT). Adjust `RATE_LIMIT_PER_MINUTE` as needed.

**No authorization layer.** There is no role-based access control or API key authentication. ClinEvidence is designed to run on an internal network or behind a reverse proxy that handles authentication. Do not expose it to the public internet without an auth layer in front.

**Input sanitization.** All text inputs pass through the safety filter before reaching agents. File uploads are validated by extension whitelist and size limit. Image filenames are not sanitized before being passed to the orchestrator (low-risk, but worth noting).

---

## Known limitations

- **Single worker only.** MemorySaver is in-process. Running multiple uvicorn workers causes sessions from different requests to be invisible to each other. To scale horizontally, replace MemorySaver with a Redis- or PostgreSQL-backed LangGraph checkpointer.

- **Model weights not included.** The three imaging models (brain MRI, chest X-ray, skin lesion) must be obtained and placed in `./models/` separately. See [docs/deployment.md](docs/deployment.md).

- **Azure OpenAI only.** All LLM and embedding calls are hardcoded to Azure OpenAI client configuration. Switching to the public OpenAI API or another provider requires changes to `settings.py` and the LangChain client instantiation in each agent.

- **Session expiry.** There is no TTL on sessions. MemorySaver retains all conversation state indefinitely, which becomes a memory leak in long-running services with many unique sessions. Restart the service to clear accumulated state.

- **Imaging validation flow is not restart-safe.** A paused workflow (awaiting `/v1/validate`) is held only in memory. If the service restarts between image upload and validation, the workflow is lost and the session must start over.

- **Confidence score is a proxy metric.** The KB confidence value is computed as the fraction of query terms found in the retrieved context, not as a calibrated probability of answer correctness. High confidence does not guarantee accuracy.

- **Tavily result count triggers PubMed fallback.** PubMed is queried only when Tavily returns fewer than 2 results, regardless of result quality. Two low-quality Tavily results suppress the PubMed search.

---

## Local development

```bash
make install-dev   # installs with dev extras (ruff, mypy, pytest, etc.)
make run           # uvicorn with --reload on :8000
make test          # pytest with coverage (48% minimum threshold)
make lint          # ruff check + mypy
```

**GPU inference.** Imaging models run on GPU automatically if `torch.cuda.is_available()` returns `True`. There is no warning if CUDA is expected but unavailable; inference silently falls back to CPU, which is significantly slower for the U-Net skin lesion model.

**Qdrant local vs. remote.** Omitting `QDRANT_URL` uses a local file-based Qdrant instance at `./data/qdrant_db`. This is convenient for development but not suitable for production data persistence.

**Docker volume permissions.** On some Linux hosts, the bind-mounted `./data` and `./models` directories may be owned by root inside the container. If `clinevidence-ingest` fails silently, check write permissions on the host directories.

**Empty knowledge base.** The `/health` endpoint returns `knowledge_base_ready: false` when no documents have been ingested. Queries still execute but return "no evidence found" messages rather than errors. Ingest at least one document before testing the RAG pipeline.

**Large PDF ingestion.** Docling extracts all images from PDFs and summarizes each with a GPT-4o vision call. A 200-page PDF with many figures will make hundreds of API calls and may take 10-20 minutes. For testing, use a short document.

---

## License

[MIT](LICENSE) - Copyright (c) 2025 ClinEvidence Contributors

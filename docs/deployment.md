# ClinEvidence Deployment Guide

---

## Recommended: Docker Compose

Docker Compose starts both Qdrant and the ClinEvidence app.

### Prerequisites
- Docker 24+ and Docker Compose v2
- An `.env` file (see `.env.example`)
- Model weight files in `./models/`

### Steps

```bash
# 1. Copy environment template
cp .env.example .env
# Edit .env with your API keys

# 2. Create required directories
mkdir -p data/raw models uploads

# 3. Place model weights
# See "Model Setup" section below

# 4. Start all services
docker compose up -d

# 5. Check health
curl http://localhost:8000/health

# 6. View logs
docker compose logs -f app
```

### Stop Services
```bash
docker compose down
```

### Rebuild After Code Changes
```bash
docker compose build app
docker compose up -d app
```

---

## Plain Docker (Single Container)

For environments where you run your own Qdrant.

```bash
docker build -t clinevidence:latest .

docker run -d \
  --name clinevidence \
  -p 8000:8000 \
  --env-file .env \
  -e QDRANT_URL=http://your-qdrant-host:6333 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/uploads:/app/uploads \
  clinevidence:latest
```

---

## Manual Installation

For development or environments without Docker.

### Requirements
- Python 3.11+
- ffmpeg (for audio processing)
- libgomp1 (for PyTorch)

```bash
# 1. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install ClinEvidence
pip install -e .

# 3. Copy and configure environment
cp .env.example .env
# Edit .env with your values

# 4. Create directories
mkdir -p data/qdrant_db data/docs_db data/parsed_docs \
         data/raw models uploads/sessions \
         uploads/images uploads/audio

# 5. Run the server
uvicorn clinevidence.main:app \
  --host 0.0.0.0 --port 8000 --reload
```

---

## Model Setup

The imaging agents require pre-trained model weight files.
Place them at the paths configured in `.env`:

### Brain MRI Model
- Default path: `./models/brain_mri.pth`
- Architecture: DenseNet-121 with 4-class classifier
- Classes: glioma, meningioma, pituitary_tumor, no_tumor
- Format: PyTorch `state_dict`

### Chest X-ray Model
- Default path: `./models/chest_xray.pth`
- Architecture: DenseNet-121 with 2-class classifier
- Classes: covid19, normal
- Format: PyTorch `state_dict`

### Skin Lesion Model
- Default path: `./models/skin_lesion.pth.tar`
- Architecture: U-Net with 7-class classifier
- Classes: melanoma, nevus, basal_cell_carcinoma,
  actinic_keratosis, benign_keratosis, dermatofibroma,
  vascular_lesion
- Format: checkpoint dict with `state_dict` key

If model files are missing, the imaging endpoints will return
a descriptive error message. Text-based endpoints remain
fully functional.

---

## Environment Variable Setup

Required variables that must be set before starting:

```bash
# Azure OpenAI
deployment_name=your-gpt4o-deployment
azure_endpoint=https://your-resource.openai.azure.com/
openai_api_key=your-key

# Embeddings
embedding_deployment_name=your-embedding-deployment
embedding_azure_endpoint=https://your-resource.openai.azure.com/
embedding_openai_api_key=your-key

# External services
ELEVEN_LABS_API_KEY=your-key
TAVILY_API_KEY=your-key

# Security
SESSION_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

---

## Document Ingestion

Ingest clinical guidelines, research papers, and protocols:

```bash
# Ingest a single PDF
clinevidence-ingest --path ./data/raw/sepsis_guidelines.pdf

# Ingest all PDFs in a directory
clinevidence-ingest --path ./data/raw/

# With a custom collection name
clinevidence-ingest --path ./data/raw/ \
  --collection my_collection
```

Or using Make:
```bash
make ingest PATH_ARG=./data/raw/sepsis_guidelines.pdf
```

---

## Health Monitoring

The `/health` endpoint returns:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "knowledge_base_ready": true
}
```

`knowledge_base_ready: false` indicates no documents have
been ingested yet.

### Prometheus Metrics

Metrics are exposed at `/metrics` via
`prometheus-fastapi-instrumentator`. Key metrics:
- `http_requests_total`
- `http_request_duration_seconds`
- `http_request_size_bytes`

---

## Scaling Considerations

- **Single worker**: The default configuration runs one uvicorn
  worker. This is intentional as LangGraph's MemorySaver
  checkpointer is in-process and not shared across workers.

- **Scaling across instances**: Replace `MemorySaver` in
  `orchestrator.py` with a Redis-backed or PostgreSQL-backed
  checkpointer from LangGraph for multi-instance deployments.

- **Qdrant**: For high-query volumes, use a dedicated Qdrant
  cloud instance with appropriate collection sharding.

- **Rate limiting**: Default is 30 requests/minute per IP.
  Adjust `rate_limit_per_minute` in settings.

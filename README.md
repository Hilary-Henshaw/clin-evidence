# ClinEvidence

[![CI](https://github.com/clinevidence/clinevidence/actions/workflows/ci.yml/badge.svg)](https://github.com/clinevidence/clinevidence/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/clinevidence/clinevidence/branch/main/graph/badge.svg)](https://codecov.io/gh/clinevidence/clinevidence)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](Dockerfile)

**Clinical Evidence Retrieval & Diagnostic Support for ICU Teams**

---

## Overview

ICU clinicians face one of the most demanding information environments
in medicine. Within minutes, a bedside decision may require synthesising
knowledge from clinical guidelines, recent trial data, imaging findings,
and patient-specific context — often simultaneously and under significant
time pressure. Existing tools require clinicians to leave their workflow,
open multiple databases, and manually evaluate contradictory evidence.

ClinEvidence addresses this gap with a multi-agent AI system that
integrates directly into the clinical workflow. Clinicians can ask
natural language questions and receive evidence-based answers sourced
from ingested clinical guidelines, real-time PubMed literature, and
web evidence — all within seconds. They can also upload medical images
for AI-assisted interpretation of chest X-rays, brain MRIs, and skin
lesions, with mandatory human validation before results are acted upon.

Unlike simple chatbots, ClinEvidence uses a LangGraph workflow with
specialised agents, hybrid vector retrieval, cross-encoder reranking,
and structured safety filtering on every input and output. The system
is designed to augment clinician expertise, not replace it: every
high-stakes analysis requires explicit human approval, every response
carries a medical disclaimer, and all outputs are filtered through
safety guardrails.

---

## Key Features

1. **Multi-Agent Routing** — LangGraph automatically routes each query
   to the most appropriate agent: knowledge base RAG, web/PubMed
   evidence search, or direct clinical conversation.

2. **Hybrid RAG Pipeline** — Documents are indexed using BM25 sparse
   retrieval combined with dense vector embeddings in Qdrant, then
   reranked with a cross-encoder model for maximum precision.

3. **Clinical Document Ingestion** — Docling extracts structured
   content from PDFs including OCR text, tables, formulas, and images.
   GPT-4o summarises extracted figures in clinical language.

4. **Medical Image Analysis** — Three specialised PyTorch models
   analyse chest X-rays (COVID-19/normal), brain MRIs (4-class tumour
   classification), and skin lesions (7-class dermoscopy).

5. **Human-in-the-Loop Validation** — Imaging analyses and
   high-confidence responses trigger LangGraph interrupt workflows,
   pausing until a clinician explicitly approves or rejects.

6. **Dual-Layer Safety Filtering** — Every input is screened for
   harmful content, prompt injection, and non-medical requests. Every
   output is verified for medical disclaimers and dangerous advice
   before delivery.

7. **Voice Interface** — Full Eleven Labs integration for speech
   transcription and synthesis, enabling hands-free clinical queries.

8. **Production-Ready Infrastructure** — FastAPI with async endpoints,
   Prometheus metrics, rate limiting, request tracing, structured
   logging, Docker Compose deployment, and comprehensive test coverage.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/clinevidence/clinevidence.git
cd clinevidence

# 2. Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI, Tavily, and Eleven Labs keys

# 3. Start with Docker Compose
docker compose up -d

# 4. Verify health
curl http://localhost:8000/health

# 5. Ingest clinical documents (optional)
clinevidence-ingest --path ./data/raw/
```

Visit `http://localhost:8000/docs` for the interactive API explorer.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   HTTP Clients (CLI/UI)                  │
└──────────────────────────┬───────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   FastAPI + SlowAPI  │
                │  Rate limiting, CORS │
                │  Prometheus metrics  │
                └──────────┬──────────┘
                           │
              ┌────────────▼────────────┐
              │   WorkflowOrchestrator  │
              │      (LangGraph)        │
              └────────────┬────────────┘
          ┌────────────────┼────────────────┐
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
   │ SafetyFilter│  │KnowledgeBase│  │  Evidence  │
   │  (guardrails│  │  (RAG + BM25│  │  Searcher  │
   │  in/out)    │  │   + Qdrant) │  │ (Tavily +  │
   └─────────────┘  └──────┬──────┘  │  PubMed)   │
                           │         └─────┬───────┘
                    ┌──────▼──────┐        │
                    │  ImagingRouter       │
                    │ BrainMRI /  │◄───────┘
                    │ ChestXray / │
                    │ SkinLesion  │
                    └─────────────┘
```

---

## Configuration

All settings are controlled via environment variables.
See [docs/configuration.md](docs/configuration.md) for the
full reference.

Minimum required variables:
```
deployment_name        # Azure OpenAI deployment
azure_endpoint         # Azure OpenAI endpoint
openai_api_key         # Azure OpenAI API key
embedding_deployment_name
embedding_azure_endpoint
embedding_openai_api_key
ELEVEN_LABS_API_KEY
TAVILY_API_KEY
SESSION_SECRET_KEY
```

---

## Usage Examples

### Text Query

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the recommended vasopressor \
                in refractory septic shock?",
    "session_id": "my-session-id"
  }'
```

### Image Upload

```bash
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@chest_xray.jpg"
```

### Voice Synthesis

```bash
curl -X POST http://localhost:8000/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Administer 30 mL/kg crystalloid fluid."}'
```

---

## Development

```bash
pip install -e ".[dev]"
make test
make lint
make type-check
```

See [docs/development.md](docs/development.md) for the
full developer guide including how to add new agents.

---

## Roadmap

- [x] Multi-agent LangGraph workflow
- [x] Hybrid RAG with Qdrant
- [x] Docling-powered document ingestion
- [x] Three imaging modalities (brain MRI, chest X-ray, skin)
- [x] Safety filtering (input + output)
- [x] Human-in-the-loop validation
- [x] Eleven Labs voice interface
- [x] Prometheus observability
- [x] Docker Compose deployment
- [ ] PostgreSQL-backed session persistence
- [ ] DICOM image format support
- [ ] Multi-language clinical guidelines
- [ ] Feedback loop for answer quality improvement
- [ ] Fine-tuned embeddings for clinical terminology
- [ ] FHIR integration for patient context

---

## Acknowledgments

Inspired by the challenges facing clinical teams who need
rapid access to evidence at the point of care. Built with
the goal of reducing cognitive load in high-stakes environments
while maintaining the primacy of human clinical judgement.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

# ClinEvidence Architecture

## System Overview

ClinEvidence is a multi-agent clinical decision-support system
designed for ICU and critical care teams. It provides rapid,
evidence-based answers to clinical questions by combining
retrieval-augmented generation (RAG) from indexed medical
literature, real-time web and PubMed evidence search, and
AI-assisted medical image analysis.

The system is built around LangGraph, a stateful graph execution
framework that coordinates multiple specialised AI agents. Each
agent is responsible for a distinct capability: safety filtering,
query routing, knowledge retrieval, web search, or imaging
analysis. The graph enforces a defined workflow with conditional
edges, enabling dynamic agent selection and human-in-the-loop
validation for high-stakes decisions.

The API layer is implemented with FastAPI, providing
asynchronous HTTP endpoints for text queries, image uploads,
speech transcription, and voice synthesis. Prometheus metrics,
rate limiting, and structured logging are built in from the start.

---

## Multi-Agent Workflow Diagram

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│  POST /v1/chat  │  POST /v1/upload  │  POST /v1/validate    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  WorkflowOrchestrator│  (LangGraph)
              └──────────┬───────────┘
                         │
                    ┌────▼────┐
                    │ assess  │  SafetyFilter (input check)
                    │ input   │  ModalityDetector (if image)
                    └────┬────┘
                         │ blocked?
                    ┌────▼────┐
                    │ select  │  AzureChatOpenAI
                    │ agent   │  JSON routing decision
                    └────┬────┘
           ┌─────────────┼──────────────┐
           │             │              │
      ┌────▼────┐  ┌─────▼─────┐  ┌────▼──────┐
      │  CONV-  │  │ KNOWLEDGE │  │   WEB     │
      │ERATION  │  │   BASE    │  │ EVIDENCE  │
      └────┬────┘  └─────┬─────┘  └────┬──────┘
           │             │              │
           │      ┌──────┴──────┐       │
           │      │ confidence  │       │
           │      │ < threshold │──────►│
           │      └──────┬──────┘       │
           │             │              │
           └─────────────┴──────────────┘
                         │
                    ┌────▼────┐
           ┌────────┤  check  ├──────────────────┐
           │        │ valid.  │                  │
           │        └─────────┘                  │
           │                                     │
      ┌────▼──────┐              ┌───────────────▼─────────────┐
      │  apply    │              │         await               │
      │  output   │◄─────────────│       validation            │
      │  safety   │              │  (human interrupt/resume)   │
      └────┬──────┘              └─────────────────────────────┘
           │
      ┌────▼──────┐
      │  IMAGING  │  (BRAIN_MRI / CHEST_XRAY / SKIN_LESION)
      │  AGENTS   │  → require_validation = True
      └───────────┘
```

---

## Component Descriptions

### WorkflowOrchestrator (`agents/orchestrator.py`)

The central controller that compiles and runs the LangGraph
state graph. It holds references to all agent components and
defines the node functions and routing logic. The MemorySaver
checkpointer persists conversation state across requests within
a session.

### SafetyFilter (`agents/safety_filter.py`)

A dual-layer content moderation system. Input safety checks
block harmful requests, prompt injections, and non-medical
content before any processing. Output safety checks verify that
AI-generated responses contain appropriate disclaimers and no
dangerous advice. Fails open on input (allows on LLM error),
fails closed on output (blocks on LLM error).

### KnowledgeBase (`agents/rag/pipeline.py`)

Orchestrates the full RAG pipeline:
1. **DocumentExtractor**: Uses Docling to parse PDFs with OCR,
   table structure recognition, and formula enrichment.
2. **DocumentFormatter**: Uses GPT-4o vision to summarise
   extracted images; applies semantic chunking with LLM
   assistance.
3. **KnowledgeStore**: Hybrid BM25 + dense vector search via
   Qdrant, with support for both local file and remote instances.
4. **QueryEnricher**: Expands medical queries with synonyms
   and related clinical terminology.
5. **ResultRanker**: Cross-encoder reranking using
   `ms-marco-TinyBERT-L-6` for precise relevance scoring.
6. **AnswerSynthesizer**: Generates cited, evidence-based
   answers with mandatory medical disclaimers.

### EvidenceSearchProcessor (`agents/search/search_processor.py`)

Real-time clinical evidence retrieval combining Tavily web
search and PubMed NCBI E-utilities. Falls back to PubMed when
Tavily returns few results. Synthesises findings using GPT-4o.

### ModalityDetector (`agents/imaging/modality_detector.py`)

Uses GPT-4o vision to classify an uploaded image into one of
five categories: BRAIN_MRI, CHEST_XRAY, SKIN_LESION, OTHER,
or NON_MEDICAL. Returns a structured result with confidence.

### ImagingRouter (`agents/imaging/router.py`)

Routes classified images to the appropriate specialist model:
- **BrainMRIAnalyzer**: DenseNet-121, 4-class tumor classifier
- **ChestXrayAnalyzer**: DenseNet-121, COVID-19/normal
- **SkinLesionAnalyzer**: U-Net based, 7-class dermatology

### ConversationAgent (`agents/conversation.py`)

Handles general medical questions, concept explanations, and
educational queries using GPT-4o with a clinical system prompt.
Maintains conversation history within session token limits.

---

## Data Flow: HTTP Request to Response

1. Client sends POST /v1/chat with `message` and `session_id`
2. FastAPI validates the request with Pydantic models
3. The session cookie is validated by `get_session()`
4. `orchestrator.process()` is called with the query
5. LangGraph begins execution at `assess_input` node:
   - SafetyFilter evaluates the input text
   - If an image_path is set, ModalityDetector classifies it
6. `select_agent` node routes to the best agent via LLM
7. Selected agent node executes and populates `response`
8. For imaging results, `check_validation` may trigger
   `await_validation` which calls `interrupt()` and pauses
9. `apply_output_safety` validates the final response
10. FastAPI serialises the result to ChatResponse schema

---

## RAG Pipeline Detailed Breakdown

### Ingestion Phase (`KnowledgeBase.ingest`)

```
PDF File
  ↓
DocumentExtractor.extract()
  → Docling PDF conversion with OCR + table + formula
  → Returns (markdown_text, [image_paths])
  ↓
DocumentFormatter.replace_image_placeholders()
  → GPT-4o vision summarises each extracted image
  → Image placeholders replaced with text summaries
  ↓
DocumentFormatter.semantic_chunk()
  → LLM identifies semantic split points (~512 chars)
  → Character-level overlap applied between chunks
  ↓
KnowledgeStore.add_documents()
  → Chunks embedded with AzureOpenAIEmbeddings
  → Stored in Qdrant (local or remote)
```

### Query Phase (`KnowledgeBase.query`)

```
User Question
  ↓
QueryEnricher.enrich()
  → LLM classifies as medical/general
  → Adds synonyms, ICD codes, related terms
  ↓
KnowledgeStore.hybrid_search()
  → BM25 sparse + dense vector retrieval
  → Returns top-k=5 candidates
  ↓
ResultRanker.rerank()
  → CrossEncoder scores each (query, doc) pair
  → Returns top-k=3 by relevance
  ↓
AnswerSynthesizer.synthesize()
  → Builds numbered context from ranked docs
  → GPT-4o generates cited, structured answer
  → Confidence = query-term coverage ratio
  → Returns (answer, sources, confidence)
```

---

## Technology Decisions

| Technology | Rationale |
|---|---|
| LangGraph | Stateful graph execution with persistence, conditional routing, and human-in-the-loop interrupts |
| Qdrant | Supports hybrid BM25+dense search natively; runs locally or as a managed service |
| Docling | Best-in-class PDF extraction with OCR, table structure, and formula support |
| AzureOpenAI | Enterprise-grade GPT-4o with data residency controls suitable for healthcare |
| DenseNet-121 | Well-validated architecture for medical imaging classification |
| FastAPI | High-performance async Python API with automatic OpenAPI docs |
| Pydantic | Runtime type validation ensures data integrity at API boundaries |
| CrossEncoder | More accurate reranking than bi-encoders for long-form clinical text |

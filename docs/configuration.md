# ClinEvidence Configuration Reference

All configuration is loaded from environment variables (or a
`.env` file). Copy `.env.example` to `.env` and fill in your
values. Never commit `.env` to version control.

---

## Azure OpenAI — Language Model

| Variable | Type | Default | Description |
|---|---|---|---|
| `deployment_name` | str | required | Azure OpenAI deployment name for GPT-4o |
| `model_name` | str | `gpt-4o` | Model name (informational) |
| `azure_endpoint` | str | required | Azure OpenAI endpoint URL |
| `openai_api_key` | str | required | Azure OpenAI API key |
| `openai_api_version` | str | `2024-02-15` | API version string |

---

## Azure OpenAI — Embeddings

| Variable | Type | Default | Description |
|---|---|---|---|
| `embedding_deployment_name` | str | required | Deployment name for text-embedding-ada-002 |
| `embedding_model_name` | str | `text-embedding-ada-002` | Embedding model name |
| `embedding_azure_endpoint` | str | required | Azure OpenAI endpoint for embeddings |
| `embedding_openai_api_key` | str | required | API key for embedding model |
| `embedding_openai_api_version` | str | `2024-02-15` | API version for embeddings |

---

## Eleven Labs — Speech

| Variable | Type | Default | Description |
|---|---|---|---|
| `ELEVEN_LABS_API_KEY` | str | required | Eleven Labs API key |
| `elevenlabs_voice_id` | str | `21m00Tcm4TlvDq8ikWAM` | Voice ID for TTS (Rachel voice) |
| `elevenlabs_model_id` | str | `eleven_monolingual_v1` | TTS model ID |

---

## Tavily — Web Search

| Variable | Type | Default | Description |
|---|---|---|---|
| `TAVILY_API_KEY` | str | required | Tavily search API key |
| `search_max_results` | int | `5` | Maximum results per search |

---

## HuggingFace

| Variable | Type | Default | Description |
|---|---|---|---|
| `HUGGINGFACE_TOKEN` | str | `""` | HuggingFace access token (for gated models) |

---

## Qdrant — Vector Database

| Variable | Type | Default | Description |
|---|---|---|---|
| `QDRANT_URL` | str | `None` | Remote Qdrant URL. If not set, uses local file storage |
| `QDRANT_API_KEY` | str | `None` | API key for remote Qdrant (cloud) |

When `QDRANT_URL` is not set, Qdrant runs in local file mode
using the path set by `kb_qdrant_path`.

---

## Session Security

| Variable | Type | Default | Description |
|---|---|---|---|
| `SESSION_SECRET_KEY` | str | `change-me-in-production` | Secret for signing session cookies. Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |

---

## Knowledge Base

| Variable | Type | Default | Description |
|---|---|---|---|
| `kb_chunk_size` | int | `512` | Target characters per document chunk |
| `kb_chunk_overlap` | int | `50` | Overlap characters between adjacent chunks |
| `kb_embedding_dim` | int | `1536` | Embedding vector dimension (must match model) |
| `kb_retrieval_top_k` | int | `5` | Number of candidates to retrieve from vector store |
| `kb_reranker_top_k` | int | `3` | Number of results to keep after reranking |
| `kb_min_confidence` | float | `0.40` | Minimum RAG confidence before escalating to web search |
| `kb_max_context_tokens` | int | `8192` | Maximum context tokens for synthesis prompt |
| `kb_qdrant_path` | str | `./data/qdrant_db` | Local Qdrant storage directory |
| `kb_docs_path` | str | `./data/docs_db` | Document metadata storage path |
| `kb_collection_name` | str | `clinevidence_kb` | Qdrant collection name |
| `kb_parsed_docs_path` | str | `./data/parsed_docs` | Directory for parsed document output |

---

## Medical Imaging

| Variable | Type | Default | Description |
|---|---|---|---|
| `brain_mri_model_path` | str | `./models/brain_mri.pth` | Path to brain MRI PyTorch model weights |
| `chest_xray_model_path` | str | `./models/chest_xray.pth` | Path to chest X-ray PyTorch model weights |
| `skin_lesion_model_path` | str | `./models/skin_lesion.pth.tar` | Path to skin lesion checkpoint file |
| `max_upload_bytes` | int | `5242880` | Maximum image upload size (5 MB) |
| `allowed_image_formats` | list | `["png","jpg","jpeg"]` | Accepted image file extensions |

---

## Server

| Variable | Type | Default | Description |
|---|---|---|---|
| `HOST` | str | `0.0.0.0` | Server bind address |
| `PORT` | int | `8000` | Server port |
| `rate_limit_per_minute` | int | `30` | Requests per minute per IP (slowapi) |
| `upload_dir` | str | `./uploads` | Base directory for uploaded files |

---

## Agent Temperatures

| Variable | Type | Default | Description |
|---|---|---|---|
| `decision_temperature` | float | `0.1` | Temperature for agent routing decisions |
| `conversation_temperature` | float | `0.7` | Temperature for conversational responses |
| `rag_temperature` | float | `0.3` | Temperature for RAG synthesis |
| `search_temperature` | float | `0.3` | Temperature for web evidence synthesis |
| `chunking_temperature` | float | `0.0` | Temperature for semantic chunking |

---

## Routing

| Variable | Type | Default | Description |
|---|---|---|---|
| `routing_confidence_threshold` | float | `0.85` | Minimum confidence for agent selection |
| `max_conversation_messages` | int | `20` | Maximum history messages kept per session |

---

## Generating a Secure Session Key

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Use the output as `SESSION_SECRET_KEY` in production.

# ClinEvidence Troubleshooting Guide

---

## Missing Model Weights

**Symptom:**
```
FileNotFoundError: Brain MRI model weights not found at:
./models/brain_mri.pth
```

**Cause:** The PyTorch model checkpoint file is not present
at the configured path.

**Resolution:**
1. Download the appropriate model weights for your task.
2. Place the file at the path specified in your `.env`
   (e.g. `brain_mri_model_path=./models/brain_mri.pth`).
3. Ensure the file has read permissions.
4. Text-based endpoints (chat, search) work without model
   weights — only imaging analysis requires them.

---

## Qdrant Connection Errors

**Symptom:**
```
RuntimeError: KnowledgeStore not initialised.
ConnectionRefusedError: [Errno 111] Connection refused
```

**Cause A:** Remote Qdrant URL is set but the service is
not running.

**Resolution A:**
- If using Docker Compose: `docker compose up qdrant`
- Check `QDRANT_URL` in your `.env` points to the correct host
- Verify Qdrant is healthy: `curl http://your-qdrant:6333/healthz`

**Cause B:** Local Qdrant path is not writable.

**Resolution B:**
```bash
mkdir -p ./data/qdrant_db
chmod 755 ./data/qdrant_db
```

**Cause C:** Qdrant version incompatibility.

**Resolution C:**
```bash
pip install qdrant-client>=1.13.0 langchain-qdrant>=0.2.0
```

---

## LLM API Errors

**Symptom:**
```
openai.AuthenticationError: Incorrect API key
openai.NotFoundError: The model deployment does not exist
```

**Resolution:**
1. Verify `openai_api_key` is correct in `.env`
2. Verify `deployment_name` matches exactly what is deployed
   in your Azure OpenAI resource
3. Verify `azure_endpoint` ends with `/` and matches your
   resource URL
4. Check the API version: `openai_api_version=2024-02-15`

**Symptom:**
```
openai.RateLimitError: Rate limit exceeded
```

**Resolution:**
- Check your Azure OpenAI quota and token rate limits
- Reduce `search_max_results` in settings
- Consider upgrading your Azure OpenAI tier

---

## Speech API Issues

**Symptom:**
```
502 Bad Gateway: Transcription service unavailable
```

**Cause:** Eleven Labs API key is invalid or the service is
unavailable.

**Resolution:**
1. Verify `ELEVEN_LABS_API_KEY` in `.env`
2. Check Eleven Labs service status
3. Verify your account has the Speech-to-Text feature enabled
   (requires Eleven Labs Scribe access)

**Symptom:** Audio file is uploaded but synthesis fails.

**Resolution:**
- Check that `uploads/audio/` directory is writable
- Verify the voice ID is valid for your Eleven Labs account
- Default voice ID: `21m00Tcm4TlvDq8ikWAM` (Rachel)

---

## Memory Issues with Large PDFs

**Symptom:**
```
MemoryError during document extraction
Process killed (OOM)
```

**Cause:** Very large PDFs (>100 pages) with many images can
exhaust memory during Docling extraction.

**Resolution:**
1. Split large PDFs into smaller chunks before ingestion
2. Reduce `images_scale` in `DocumentExtractor._build_converter`
   from 2.0 to 1.0
3. Set `generate_page_images=False` if page images are not
   needed
4. Increase Docker container memory limit:
   ```yaml
   services:
     app:
       mem_limit: 8g
   ```
5. Process PDFs one at a time rather than batching

---

## CUDA / CPU Device Issues

**Symptom:**
```
RuntimeError: CUDA out of memory
RuntimeError: Expected all tensors to be on the same device
```

**Resolution:**
- The imaging models automatically use GPU if available
  and fall back to CPU
- To force CPU: set environment variable
  `CUDA_VISIBLE_DEVICES=""` before starting the server
- For CUDA OOM: batch size is always 1 (single image),
  so this typically indicates a model loading issue
- Restart the service to free CUDA memory:
  ```bash
  docker compose restart app
  ```

**Symptom:**
```
torch.cuda.is_available() returns False unexpectedly
```

**Resolution:**
- In Docker: ensure NVIDIA Container Toolkit is installed
- Add GPU resources to `docker-compose.yml`:
  ```yaml
  services:
    app:
      deploy:
        resources:
          reservations:
            devices:
              - capabilities: [gpu]
  ```

---

## Session Cookie Issues

**Symptom:**
```
401 Unauthorized: Invalid or tampered session cookie
403 Forbidden: Session ID does not match the session cookie
```

**Resolution:**
1. Ensure `SESSION_SECRET_KEY` is consistent across restarts
   (not randomly generated at startup)
2. If testing with curl, include cookies: `--cookie-jar cookies.txt`
3. Clear browser cookies and start a new session
4. For API clients, store and resend the `clinevidence_session`
   cookie

---

## Tavily Search Returns No Results

**Symptom:** Chat responses for recent clinical events lack
current data.

**Resolution:**
1. Verify `TAVILY_API_KEY` is valid
2. Check Tavily API quota usage
3. The system automatically falls back to PubMed when Tavily
   returns fewer than 2 results
4. Increase `search_max_results` in settings

---

## Knowledge Base Returns Low Confidence

**Symptom:** Responses frequently fall back to web search
even after document ingestion.

**Resolution:**
1. Verify documents were ingested: check `data/qdrant_db/`
   has content
2. Increase `kb_retrieval_top_k` and `kb_reranker_top_k`
3. Lower `kb_min_confidence` threshold (default 0.40)
4. Re-ingest with a smaller `kb_chunk_size` (e.g. 256)
   for more granular retrieval
5. Verify the embedding model dimensions match
   `kb_embedding_dim=1536` for `text-embedding-ada-002`

# ClinEvidence API Reference

Base URL: `http://localhost:8000`

Interactive documentation available at `/docs` (Swagger UI)
and `/redoc` (ReDoc).

---

## GET /health

Returns service health status and readiness.

**Response: 200 OK**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "knowledge_base_ready": true
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

## POST /v1/chat

Submit a clinical query for evidence-based response.

**Request Body:**
```json
{
  "message": "What is the first-line vasopressor in septic shock?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `message` | string | yes | 1–4096 chars, non-blank |
| `session_id` | string | yes | 1–128 chars |

**Response: 200 OK**
```json
{
  "message": "Norepinephrine is the first-line vasopressor...",
  "agent_used": "KNOWLEDGE_BASE",
  "sources": [
    {
      "title": "Surviving Sepsis Campaign 2021",
      "url": null,
      "page": 12,
      "source_type": "document"
    }
  ],
  "confidence": 0.87,
  "processing_time_ms": 1423,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "requires_validation": false
}
```

**Error Codes:**
- `422 Unprocessable Entity` — invalid request body
- `403 Forbidden` — session ID mismatch
- `500 Internal Server Error` — orchestration failure

**Example:**
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain SOFA score components.",
       "session_id": "my-session-123"}'
```

---

## POST /v1/upload

Upload and analyse a medical image.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | Image file (PNG, JPG, JPEG, max 5 MB) |

Session cookie must be present (from a prior `/v1/chat` call
or set via `create_session`).

**Response: 200 OK**
```json
{
  "filename": "chest_ap.jpg",
  "image_type": "CHEST_XRAY",
  "analysis": "**CHEST XRAY Analysis**\n\n**Diagnosis:** normal...",
  "confidence": 0.91,
  "requires_validation": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Codes:**
- `413 Request Entity Too Large` — file exceeds 5 MB
- `415 Unsupported Media Type` — unsupported file format
- `500 Internal Server Error` — analysis failure

**Example:**
```bash
curl -X POST http://localhost:8000/v1/upload \
  -F "file=@/path/to/chest_xray.jpg"
```

---

## POST /v1/validate

Submit a human validation decision for a pending analysis.

**Request Body:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "approved": true,
  "feedback": "Confirmed by radiologist Dr. Smith."
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | yes | Session with pending validation |
| `approved` | boolean | yes | True to approve, False to reject |
| `feedback` | string | no | Optional clinician feedback (max 2048 chars) |

**Response: 200 OK**
```json
{
  "status": "approved",
  "message": "Analysis approved. Feedback recorded: Confirmed by radiologist."
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session", "approved": true}'
```

---

## POST /v1/transcribe

Transcribe an audio file to text.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | Audio file (mp3, wav, m4a, webm, etc.) |

**Response: 200 OK**
```json
{
  "text": "Patient presents with acute onset chest pain...",
  "duration_ms": null
}
```

**Error Codes:**
- `400 Bad Request` — empty audio file
- `415 Unsupported Media Type` — unsupported audio format
- `502 Bad Gateway` — Eleven Labs API unavailable

**Example:**
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@/path/to/clinical_note.mp3"
```

---

## POST /v1/synthesize

Convert text to speech audio.

**Request Body:**
```json
{
  "text": "Norepinephrine is the first-line vasopressor in septic shock.",
  "voice_id": "21m00Tcm4TlvDq8ikWAM"
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `text` | string | yes | 1–5000 chars |
| `voice_id` | string | no | Overrides default voice |

**Response: 200 OK**
```json
{
  "audio_url": "/uploads/audio/a1b2c3d4-5678.mp3",
  "duration_ms": null
}
```

**Error Codes:**
- `422 Unprocessable Entity` — invalid request
- `502 Bad Gateway` — Eleven Labs API unavailable

**Example:**
```bash
curl -X POST http://localhost:8000/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Early antibiotic administration is critical."}'
```

---

## Common Response Headers

| Header | Description |
|---|---|
| `X-Request-ID` | Unique request identifier (UUID) |
| `X-Processing-Time` | Request processing time in milliseconds |

---

## Agent Selection Values

The `agent_used` field in `ChatResponse` will be one of:

| Value | Meaning |
|---|---|
| `CONVERSATION` | General medical Q&A |
| `KNOWLEDGE_BASE` | Evidence from indexed clinical documents |
| `WEB_EVIDENCE` | Real-time web/PubMed search |
| `BRAIN_MRI` | Brain MRI tumor analysis |
| `CHEST_XRAY` | Chest X-ray COVID-19/normal analysis |
| `SKIN_LESION` | Dermoscopy lesion classification |
| `SAFETY_BLOCK` | Request was blocked by safety filter |

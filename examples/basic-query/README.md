# Basic Query Example

This example shows how to send a clinical question to the
ClinEvidence API and display the evidence-based response.

## Prerequisites

- ClinEvidence server running at `http://localhost:8000`
- Python 3.11+ with `httpx` installed: `pip install httpx`

## What This Example Does

1. Creates a session ID
2. Sends a clinical question to `POST /v1/chat`
3. Prints the response, agent used, sources, and confidence
4. Demonstrates session reuse for a follow-up question

## Running

```bash
python query_example.py
```

## Expected Output

```
Clinical Query Response
=======================
Question: What is the recommended fluid resuscitation...

Agent: KNOWLEDGE_BASE
Confidence: 0.87
Processing time: 1423ms

Response:
Based on the Surviving Sepsis Campaign 2021 guidelines...

Sources:
  [1] Surviving Sepsis Campaign 2021 (document)
```

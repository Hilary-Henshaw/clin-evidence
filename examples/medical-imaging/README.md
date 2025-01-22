# Medical Imaging Example

This example demonstrates how to upload a medical image to
ClinEvidence for AI-assisted analysis and how to submit a
validation decision for the result.

## Prerequisites

- ClinEvidence server running at `http://localhost:8000`
- Model weights placed at the configured paths
- Python 3.11+ with `httpx` installed: `pip install httpx`
- A sample medical image (PNG or JPEG)

## What This Example Does

1. Uploads a medical image to `POST /v1/upload`
2. Displays the modality detection and diagnosis
3. Submits a validation decision via `POST /v1/validate`
4. Shows the validation acknowledgement

## Running

```bash
# With an actual image file
python imaging_example.py --image ./chest_xray.jpg

# With the built-in test pattern
python imaging_example.py --demo
```

## Expected Output

```
ClinEvidence Medical Imaging Example
=====================================

Uploading image: chest_xray.jpg
Image type detected: CHEST_XRAY
Confidence: 91%
Requires validation: Yes

Analysis:
---------
**CHEST XRAY Analysis**

**Diagnosis:** normal
**Confidence:** 91.0%

Features consistent with a normal chest X-ray...

Submitting validation: APPROVED
Status: approved
Message: Analysis approved.
```

## Notes

- If model weights are not present, the server returns a
  descriptive error explaining where to place the model files
- All imaging analyses require human validation before use
- The `requires_validation: true` flag in the response
  indicates the workflow is paused awaiting your decision

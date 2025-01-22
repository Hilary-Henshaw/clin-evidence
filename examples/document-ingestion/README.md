# Document Ingestion Example

This example demonstrates how to ingest clinical documents
(PDF files) into the ClinEvidence knowledge base using the
ingestion script.

## Prerequisites

- ClinEvidence installed: `pip install -e .`
- Environment configured: `.env` file with API keys
- One or more PDF files to ingest

## What This Example Does

1. Creates a sample clinical text file
2. Calls the ingestion script programmatically
3. Verifies the knowledge base is ready after ingestion
4. Demonstrates querying the ingested content

## Running

```bash
# Using the CLI script directly
clinevidence-ingest --path ./my_guidelines.pdf

# Using this example script
python ingest_example.py --path ./data/raw/

# Using Make
make ingest PATH_ARG=./data/raw/my_guidelines.pdf
```

## Expected Output

```
Found 3 PDF file(s) to ingest

  [1/3] sepsis_guidelines.pdf: 47 chunks, 3 images (8.2s)
  [2/3] ards_protocol.pdf: 31 chunks, 1 images (5.1s)
  [3/3] ventilator_management.pdf: 52 chunks, 5 images (9.7s)

==================================================
Ingestion complete:
  Files processed : 3
  Files failed    : 0
  Total chunks    : 130
  Total images    : 9
```

## Notes

- Only PDF files are currently supported
- Large PDFs (>100 pages) may take several minutes
- Images are summarised by GPT-4o during ingestion
- Chunks are deduplicated by Qdrant automatically

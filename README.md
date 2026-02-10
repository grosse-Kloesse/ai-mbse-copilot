# AI MBSE Copilot (Demo)

This repo is a small, runnable demo that:
1) turns an MBSE/EA-like XML export into **nodes / edges / text chunks** (JSONL),
2) indexes chunks into **Qdrant (vector DB)** using a **dummy embedding**,
3) queries top-k chunks back from Qdrant (end-to-end pipeline verified).

> Dummy embedding = deterministic hash-based vectors, used only to validate the pipeline.
> Next step is switching to a real embedding model for semantic retrieval.

---

## What’s included (what we built)
- Generate a toy MBSE XML: `data/raw/sample_mbse.xml`
- Parse XML → JSONL:
  - `data/processed/nodes.jsonl`
  - `data/processed/edges.jsonl`
  - `data/processed/chunks.jsonl`
- Trace demo (graph traversal via edges)
- Qdrant indexing + querying demo (vector search)
- Minimal FastAPI skeleton: `GET /health`

---

## Quickstart

### 0) Create Python env
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt


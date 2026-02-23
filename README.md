# AI MBSE Copilot (Demo)

A small, runnable **MBSE Copilot** demo that turns an EA/XMI-like XML export into:

- **graph data** (nodes/edges) for traceability, and
- **text chunks** indexed into a **vector database (Qdrant)** for semantic search.

It exposes a minimal **FastAPI** service (`/search`, `/status`) so you can demo the full end-to-end pipeline.

---

## What this demo does (end-to-end)

1. Generate a toy MBSE XML (EA/XMI-like)
2. Parse XML → JSONL (`nodes/edges/chunks`)
3. (Optional) Graph traversal demo (k-hop trace over edges)
4. Embed chunks with **SentenceTransformer** (sentence-transformers/all-MiniLM-L6-v2, 384-dim) and **upsert** into **Qdrant**
5. Query top-k chunks back from Qdrant (semantic retrieval)
6. Call the API:
   - `GET /health`
   - `GET /status` (checks Qdrant + collection + points_count)
   - `POST /search` (semantic search)

---

## Repo structure (important files)

- `tools/generate_toy_mbse_xml.py` — generate `data/raw/sample_mbse.xml`
- `ingest/parse_toy_mbse.py` — parse XML → JSONL outputs:
  - `data/processed/nodes.jsonl`
  - `data/processed/edges.jsonl`
  - `data/processed/chunks.jsonl`
- `ingest/trace_one_hop.py`, `ingest/trace_k_hops.py` — traceability demo over `edges.jsonl`
- `ingest/index_chunks_st.py` — embed chunks (SentenceTransformer) → Qdrant
- `ingest/search_chunks_st.py` — query Qdrant from CLI
- `api/main.py` — FastAPI app (`/health`, `/status`, `/search`)
- `docker-compose.yml` — runs `qdrant` (6333) + `api` (8000)
- `eval/queries.jsonl`, `eval/run_eval.py` — quick evaluation

> Note: `data/processed/` is generated output and is typically **not committed**.

---

## Requirements

- Python **3.11**
- Docker Desktop (for Qdrant + API via compose)

---

## Quickstart (Python + Qdrant)

If you only want a CLI demo (index + search), follow **Quickstart**.  
If you want the full service demo (FastAPI + Swagger UI), jump to **Run the API (Docker Compose)**.

> **Note:** All Python commands below assume the venv is active. If not, run `source .venv/bin/activate` first.

### 0) Create Python env
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 1) Start Qdrant (Docker)
```bash
docker compose up -d qdrant
docker compose ps
curl -s http://127.0.0.1:6333 | head
```

### 2) Generate toy XML and parse → JSONL
```bash
python tools/generate_toy_mbse_xml.py
python ingest/parse_toy_mbse.py
```

### 3) Index chunks into Qdrant (SentenceTransformer embeddings)
```bash
python ingest/index_chunks_st.py
curl -s http://127.0.0.1:6333/collections | head
```

### 4) Query from CLI
```bash
python ingest/search_chunks_st.py "power module protection logic"
python ingest/search_chunks_st.py "detect over-voltage"
```

---

## Run the API (Docker Compose)

### 1) Start API + Qdrant
```bash
docker compose up -d --build
docker compose ps
```

### 2) Verify health/status
```bash
curl -s http://127.0.0.1:8000/health && echo
curl -s http://127.0.0.1:8000/status && echo
```

Swagger UI: http://127.0.0.1:8000/docs

### 3) Search via API
```bash
curl -s -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"power module protection logic","top_k":3}' | head
```

Expected: top hit should be `BLK-100 (Power_Module)` for this query.

---

## Qdrant persistence (volume)

This project uses a Docker volume (e.g. `mbse-copilot_qdrant_data`) so Qdrant collections survive `docker compose down` and restarts.

Verify:
```bash
docker volume ls | grep qdrant
curl -s http://127.0.0.1:6333/collections | head
curl -s http://127.0.0.1:6333/collections/mbse_chunks_st | head
```

---

## Evaluation (quick check)

A tiny evaluation set is stored in `eval/queries.jsonl`.

Run evaluation (top-k configurable):
```bash
python eval/run_eval.py 3
python eval/run_eval.py 1
```

Optional: configure Qdrant URL via env var (only affects the current terminal session):
```bash
export QDRANT_URL="http://127.0.0.1:6333"
python eval/run_eval.py 3
```

---

## Troubleshooting

### `/docs` (or API behavior) not updating after code changes

If you changed code but `/docs` (or responses) still look old, restart/rebuild the API container:
```bash
docker compose restart api
# if you changed dependencies / Dockerfile:
docker compose up -d --build api
```

### API container exits / import errors

If the API is not reachable (connection refused) or the container keeps restarting, check logs:
```bash
docker compose ps
docker compose logs -n 200 api
```

Common cause: missing Python dependency inside the container. Fix `requirements.txt`/`Dockerfile` and rebuild:
```bash
docker compose up -d --build api
```

### API returns "Qdrant not reachable"

Check containers:
```bash
docker compose ps
docker compose logs -n 80 api
```

Ensure Qdrant is running:
```bash
curl -s http://127.0.0.1:6333 | head
```

### Collections empty

If `GET /collections` is empty, it usually means either:
- you haven't indexed yet, or
- Qdrant was started without the persistence volume (so data was lost after container recreation).

Fix: (re)index:
```bash
python ingest/index_chunks_st.py
```

---

## Next steps (roadmap)

- Add `POST /ask` to turn retrieval into full RAG (retrieve → LLM generate + citations)
- Expand evaluation queries and compare embedding settings
- Support real EA/XMI exports (beyond toy XML)
# AI MBSE Copilot (Demo)

A small, runnable **MBSE Copilot** demo that turns an EA/XMI-like XML export into:

- **graph data** (nodes/edges) for traceability, and
- **text chunks** indexed into a **vector database (Qdrant)** for semantic search.

It exposes a minimal **FastAPI** service (`/search`, `/trace`, `/search_trace`, `/status`) so you can demo the full end-to-end pipeline.

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
   - `POST /trace` (graph k-hop traceability)
   - `POST /search_trace` (semantic retrieval + graph traversal combined)

---

## Repo structure (important files)

- `tools/generate_toy_mbse_xml.py` — generate `data/raw/sample_mbse.xml`
- `tools/generate_toy_mbse_xml_many_v2.py` — generate v2 XML (multi-theme, realistic wording)
- `ingest/parse_toy_mbse.py` — parse XML → JSONL outputs:
  - `data/processed/nodes.jsonl`
  - `data/processed/edges.jsonl`
  - `data/processed/chunks.jsonl`
- `ingest/parse_toy_mbse_many.py` — parse XML with output directory arg (supports v2/many)
- `ingest/trace_one_hop.py`, `ingest/trace_k_hops.py` — traceability demo over `edges.jsonl`
- `ingest/index_chunks_st.py` — embed chunks → Qdrant (`mbse_chunks_st`)
- `ingest/index_chunks_many_st.py` — embed many dataset → Qdrant (`mbse_chunks_many`)
- `ingest/index_chunks_v2_st.py` — embed v2 dataset → Qdrant (`mbse_chunks_v2`)
- `ingest/search_chunks_st.py` — query Qdrant from CLI
- `api/main.py` — FastAPI app
- `api/trace.py` — graph traversal logic (adjacency, k-hop)
- `docker-compose.yml` — runs `qdrant` (6333) + `api` (8000)
- `eval/queries.jsonl`, `eval/queries_v2.jsonl`, `eval/queries_v2_typed.jsonl`
- `eval/run_eval.py` — evaluation (Recall@k, MRR, latency)

> Note: `data/processed/` is generated output and is typically **not committed**.

---

## Datasets / Collections

This repo contains multiple synthetic datasets for different purposes:

| Dataset | Size | Purpose | Qdrant Collection |
|---------|------|---------|-------------------|
| **sample** | 3 chunks | Quick sanity checks | `mbse_chunks_st` |
| **many** | 60 chunks, 40 edges | Pipeline scaling baseline | `mbse_chunks_many` |
| **v2** | 180 chunks, 120 edges | Multi-theme, realistic wording — evaluation baseline | `mbse_chunks_v2` |

v2 covers: Over/Under-voltage, Thermal, Vibration, Cooling, Insulation — more representative of real engineering descriptions.

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

### 2a) Generate sample XML and parse → JSONL (minimal)
```bash
python tools/generate_toy_mbse_xml.py
python ingest/parse_toy_mbse.py
```

### 2b) Build + index v2 dataset (recommended)

Generate v2 XML:
```bash
python tools/generate_toy_mbse_xml_many_v2.py
```

Parse to `data/processed/v2`:
```bash
python ingest/parse_toy_mbse_many.py data/raw/sample_mbse_v2.xml data/processed/v2
```

Index to Qdrant collection `mbse_chunks_v2`:
```bash
python ingest/index_chunks_v2_st.py
```

### 3) Query from CLI
```bash
python ingest/search_chunks_st.py "power module protection logic"
python ingest/search_chunks_st.py "detect over-voltage"
```

> Note: `search_chunks_st.py` uses the baseline collection (`mbse_chunks_st`).  
> For v2 results, use `ingest/search_chunks_v2_st.py` (or update the collection name in the script).

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

Expected: for the sample dataset, top hit should be `BLK-100 (Power_Module)`.

### 4) Traceability via API (graph k-hop)

```bash
curl -s -X POST "http://127.0.0.1:8000/trace" \
  -H "Content-Type: application/json" \
  -d '{"start_id":"REQ-001","k":2}' && echo
```

Expected: `paths` contains `["REQ-001","FUNC-010","BLK-100"]` for the toy dataset.

### 5) Search + Trace (semantic retrieval + graph traversal)

```bash
curl -s -X POST "http://127.0.0.1:8000/search_trace" \
  -H "Content-Type: application/json" \
  -d '{"query":"power module protection logic","top_k":3,"k":2}' && echo
```

Returns:
- `hits`: top-k semantic matches from Qdrant
- `trace_paths`: k-hop paths starting from the top hit's `source_id`

---

## Qdrant persistence (volume)

This project uses a Docker volume (e.g. `mbse-copilot_qdrant_data`) so Qdrant collections survive `docker compose down` and restarts.

Verify:
```bash
docker volume ls | grep qdrant
curl -s http://127.0.0.1:6333/collections | head
curl -s http://127.0.0.1:6333/collections/mbse_chunks_v2 | head  # (after indexing v2)
```

---

## Evaluation

A small evaluation set is in `eval/`. Metrics: **Recall@1**, **Recall@3**, **MRR@3**, **avg latency (ms)**.

### Sample dataset (sanity check)
```bash
python eval/run_eval.py 3
python eval/run_eval.py 1
```

### V2 dataset — baseline vs type-filtered

Baseline (no filter):
```bash
python eval/run_eval.py 3 eval/queries_v2.jsonl mbse_chunks_v2
```

Type-filtered (uses `expected_type` to filter by `meta.type`, reducing cross-type noise):
```bash
python eval/run_eval.py 3 eval/queries_v2_typed.jsonl mbse_chunks_v2
```

Example results on v2 (24 queries, no ID hints):

| Mode | Recall@1 | Recall@3 | MRR@3 | Latency |
|------|----------|----------|-------|---------|
| Baseline | 0.208 | 0.542 | 0.333 | ~22 ms |
| Type-filtered | 0.208 | 0.750 | 0.431 | ~25 ms |

> Type filtering significantly reduces cross-type noise and improves top-k recall — a natural fit for MBSE's structured type system (REQ / FUNC / BLK).

Optional: configure Qdrant URL via env var:
```bash
export QDRANT_URL="http://127.0.0.1:6333"
python eval/run_eval.py 3 eval/queries_v2.jsonl mbse_chunks_v2
```

---

## Troubleshooting

### `/docs` (or API behavior) not updating after code changes

```bash
docker compose restart api
# if you changed dependencies / Dockerfile:
docker compose up -d --build api
```

### API container exits / import errors

```bash
docker compose ps
docker compose logs -n 200 api
```

Common cause: missing Python dependency inside the container. Fix `requirements.txt`/`Dockerfile` and rebuild:
```bash
docker compose up -d --build api
```

### API returns "Qdrant not reachable"

```bash
docker compose ps
docker compose logs -n 80 api
curl -s http://127.0.0.1:6333 | head
```

### Collections empty

If `GET /collections` is empty:
- You haven't indexed yet, or
- Qdrant was started without the persistence volume (data lost after container recreation).

Fix: re-index:
```bash
python ingest/index_chunks_v2_st.py
```

---

## Next steps (roadmap)

- [ ] Add `POST /ask` for full RAG (retrieve → LLM generate + citations)
- [ ] Support real EA/XMI exports (beyond synthetic XML)
- [ ] Expand evaluation queries (10–50), add latency benchmarks
- [ ] CI/CD (GitHub Actions) + demo video
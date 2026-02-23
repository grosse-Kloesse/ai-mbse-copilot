# Project Notes — AI MBSE Copilot (Demo)

This is a single-file note for my **AI MBSE Copilot** demo:
- end-to-end commands,
- key concepts (CN + EN),
- troubleshooting,
- and a handoff section for continuing in a new chat.

---

## 1) Current status (what I built)

### Data foundation (Toy MBSE)
- Toy MBSE XML input: `data/raw/sample_mbse.xml`
- Parser (XML → JSONL):
  - `data/processed/nodes.jsonl`
  - `data/processed/edges.jsonl`
  - `data/processed/chunks.jsonl`
- Graph traversal demo (traceability):
  - `ingest/trace_one_hop.py`, `ingest/trace_k_hops.py`

### Vector retrieval (Qdrant + embeddings)
- Vector DB: **Qdrant** via Docker Compose (host port `6333`)
- **Persistence**: Qdrant uses a Docker **volume** (data survives `docker compose down` / restart)
- Embeddings: SentenceTransformer `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Collection: `mbse_chunks_st`
- Verified: `points_count = 3` for toy chunks

### API (FastAPI)
- FastAPI service (host port `8000`) with:
  - `GET /health`
  - `GET /status` (checks Qdrant connectivity + collection + points_count)
  - `POST /search` (semantic search in Qdrant)
  - Swagger UI: `GET /docs`

### Evaluation
- Tiny evaluation set: `eval/queries.jsonl`
- Evaluation runner: `eval/run_eval.py` (TOP_K via argv)

---

## 2) End-to-end demo (commands)

> **Note:** All Python commands below assume the venv is active. If not, run `source .venv/bin/activate` first.

### 2.1 Start services
```bash
docker compose up -d --build
docker compose ps
```

### 2.2 Check Qdrant (host side)
```bash
curl -s http://127.0.0.1:6333 | head
curl -s http://127.0.0.1:6333/collections | head
curl -s http://127.0.0.1:6333/collections/mbse_chunks_st | head
```

### 2.3 Generate toy XML and parse → JSONL
```bash
python tools/generate_toy_mbse_xml.py
python ingest/parse_toy_mbse.py
```

### 2.4 Index into Qdrant (real embeddings)
```bash
python ingest/index_chunks_st.py
curl -s http://127.0.0.1:6333/collections | head
```

### 2.5 Search (CLI)
```bash
python ingest/search_chunks_st.py "power module protection logic"
python ingest/search_chunks_st.py "detect over-voltage"
```

### 2.6 API demo
```bash
curl -s http://127.0.0.1:8000/health && echo
curl -s http://127.0.0.1:8000/status && echo

curl -s -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"power module protection logic","top_k":3}' | head
```

### 2.7 Persistence verification (proof)
```bash
docker volume ls | grep qdrant
docker compose down
docker compose up -d
curl -s http://127.0.0.1:6333/collections | head
curl -s http://127.0.0.1:6333/collections/mbse_chunks_st | head
```

### 2.8 Evaluation run
```bash
python eval/run_eval.py 3
python eval/run_eval.py 1
```

---

## 3) Key concepts (CN + EN keywords)

### Docker & networking
- **Docker image (镜像)**: template for building a container
- **Docker container (容器)**: a running instance of an image
- **Docker Compose (多服务编排)**: run multiple services together (api + qdrant)
- **Port mapping (端口映射)**: host:container, e.g. `6333:6333`
- **Docker internal DNS / service name**:
  - Host (my Mac) talks to Qdrant via `127.0.0.1:6333`
  - Containers talk to each other via service names like `http://qdrant:6333`

### Persistence (持久化)
- **Volume (卷)**: Docker-managed storage that survives container deletion
- Qdrant stores data under `/qdrant/storage` inside the container
- With a volume mapping, collections survive `docker compose down + up`

### Data formats
- **JSONL (JSON Lines)**: one JSON object per line
- **nodes/edges/chunks**:
  - `nodes.jsonl`: MBSE elements (Requirement/Function/Block…)
  - `edges.jsonl`: relationships (src_id, dst_id, rel_type)
  - `chunks.jsonl`: retrievable text chunks (chunk_id, text, source_id, meta)

### Vector search
- **Embedding (向量表示)**: map text → vector
- **Vector dimension (维度)**: 384 for sentence-transformers/all-MiniLM-L6-v2
- **Cosine similarity (余弦相似度)**: similarity metric used by Qdrant
- **Qdrant point** = vector + payload (payload = 元数据/文本等)
- **Upsert (更新插入)**: update if exists, insert if not

### FastAPI & schemas
- Decorators like `@app.post("/search")` bind a URL to a function (路由触发)
- **Pydantic BaseModel**: request/response schema (请求/响应结构)
- **Swagger UI**: FastAPI auto docs at `/docs`

### Environment variables (环境变量)
- `export VAR=...`: set environment variable in current terminal session
- `os.getenv("VAR", default)`: read environment variable in Python

---

## 4) Troubleshooting playbook (issues I debugged)

### 4.1 Docker daemon not running

**Symptom:**
- `Cannot connect to the Docker daemon ... docker.sock`

**Fix:**
- Start Docker Desktop (Docker engine must be running)

### 4.2 API connection refused on port 8000

**Diagnosis:**
```bash
docker compose ps        # api container may not be running
docker compose logs -n 200 api  # look for import errors
```

**Common root cause:** Missing Python dependency inside the container

**Fix:**
```bash
docker compose up -d --build api
```

### 4.3 Qdrant rejects point id

**Cause:** point id must be unsigned integer or UUID (not arbitrary strings)

**Fix:** use `int` id or stable `uuid5(chunk_id)`

### 4.4 Code changed but /docs not updated

**Fix:**
```bash
docker compose restart api
# or rebuild if needed
docker compose up -d --build api
```
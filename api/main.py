import os
import httpx
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field
from api.trace import trace_k 
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

app = FastAPI(title="AI MBSE Copilot")

# ---- runtime singletons (load once) ----
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mbse_chunks_v2")

model = SentenceTransformer(MODEL_NAME)

# IMPORTANT:
# - inside Docker network, reach Qdrant via service name "qdrant"
# - if you run API locally (not in Docker), you can switch to 127.0.0.1
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
client = QdrantClient(url=QDRANT_URL)

def qdrant_collection_status():
    try:
        cols = client.get_collections().collections
        names = [c.name for c in cols]
        exists = COLLECTION_NAME in names
        if not exists:
            return True, False, None, None
        
        info = client.get_collection(collection_name=COLLECTION_NAME)
        return True, True, int(info.points_count), None
    except Exception as e:
        return False, False, None, str(e)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    qdrant_ok, exists, points_count, err = qdrant_collection_status()
    return {
        "qdrant_ok": qdrant_ok,
        "collection": COLLECTION_NAME,
        "collection_exists": exists,
        "points_count": points_count,
        "error": err,
    }

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class SearchHit(BaseModel):
    score: float
    source_id: str | None = None
    chunk_id: str | None = None
    text: str | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]

def build_type_filter(expected_type: str | None):
    if not expected_type:
        return None
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    return Filter(
        must=[
            FieldCondition(
                key="meta.type",
                match=MatchValue(value=expected_type),
            )
        ]
    )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

def call_llm(prompt: str) -> str:
    if not OPENAI_API_KEY:
        # 没 key 就回退到 stub（方便你没配 key 也能 demo）
        return "LLM not configured (missing OPENAI_API_KEY)."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "input": prompt,
    }

    with httpx.Client(timeout=30.0) as h:
        r = h.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    # Responses API 的输出在 response 对象里（我们取文本即可）
    # 为了兼容不同返回结构，这里做个稳健提取
    text_parts = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text" and "text" in c:
                text_parts.append(c["text"])
    return "".join(text_parts).strip() or "(empty response)"

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    qdrant_ok, exists, points_count, err = qdrant_collection_status()
    if not qdrant_ok:
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {err}")
    if not exists:
        raise HTTPException(
            status_code=409,
            detail=f"Collection '{COLLECTION_NAME}' not ready. Run: python ingest/index_chunks_st.py",
        )

    qvec = model.encode(req.query, normalize_embeddings=True).tolist()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=req.top_k,
        with_payload=True,
    )

    hits: list[SearchHit] = []
    for p in res.points:
        payload = p.payload or {}
        hits.append(
            SearchHit(
                score=float(p.score),
                source_id=payload.get("source_id"),
                chunk_id=payload.get("chunk_id"),
                text=payload.get("text"),
            )
        )

    return SearchResponse(query=req.query, hits=hits)

class TraceRequest(BaseModel):
    start_id: str
    k: int = Field(2, ge=1, le=10)  # 默认2跳，限制 1~10


class TraceResponse(BaseModel):
    start_id: str
    k: int
    paths: list[list[str]]


@app.post("/trace", response_model=TraceResponse)
def trace(req: TraceRequest):
    qdrant_ok, exists, _, err = qdrant_collection_status()
    if not qdrant_ok:
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {err}")
    if not exists:
        raise HTTPException(status_code=409, detail=f"Collection '{COLLECTION_NAME}' not ready. Run the index script first.")
    paths = trace_k(req.start_id, req.k)
    return TraceResponse(start_id=req.start_id, k=req.k, paths=paths)

class SearchTraceRequest(BaseModel):
    query: str
    top_k: int = Field(3, ge=1, le=20)
    k: int = Field(2, ge=1, le=10)


class SearchTraceResponse(BaseModel):
    query: str
    top_hit_source_id: str | None
    hits: list[SearchHit]
    trace_paths: list[list[str]]


@app.post("/search_trace", response_model=SearchTraceResponse)
def search_trace(req: SearchTraceRequest):
    qdrant_ok, exists, _, err = qdrant_collection_status()
    if not qdrant_ok:
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {err}")
    if not exists:
        raise HTTPException(status_code=409, detail=f"Collection '{COLLECTION_NAME}' not ready. Run the index script first.")
    # 1) vector search (reuse the same logic as /search)
    qvec = model.encode(req.query, normalize_embeddings=True).tolist()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=req.top_k,
        with_payload=True,
    )

    hits: list[SearchHit] = []
    for p in res.points:
        payload = p.payload or {}
        hits.append(
            SearchHit(
                score=float(p.score),
                source_id=payload.get("source_id"),
                chunk_id=payload.get("chunk_id"),
                text=payload.get("text"),
            )
        )

    # 2) trace from the top hit's source_id
    top_source_id = hits[0].source_id if hits and hits[0].source_id else None
    trace_paths = trace_k(top_source_id, req.k) if top_source_id else []

    return SearchTraceResponse(
        query=req.query,
        top_hit_source_id=top_source_id,
        hits=hits,
        trace_paths=trace_paths,
    )

class AskRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    trace_k: int = Field(2, ge=0, le=10)  # 0 = no trace
    type_filter: str | None = None        # "Requirement" / "Function" / "Block" or None


class Citation(BaseModel):
    rank: int
    score: float
    source_id: str | None
    chunk_id: str | None
    type: str | None
    text: str | None


class AskResponse(BaseModel):
    query: str
    answer: str
    trace_paths: list[list[str]]
    citations: list[Citation]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    qdrant_ok, exists, _, err = qdrant_collection_status()
    if not qdrant_ok:
        raise HTTPException(status_code=503, detail=f"Qdrant not reachable: {err}")
    if not exists:
        raise HTTPException(status_code=409, detail=f"Collection '{COLLECTION_NAME}' not ready. Index first.")
    # 1) retrieve top-k
    qvec = model.encode(req.query, normalize_embeddings=True).tolist()
    qfilter = build_type_filter(req.type_filter)

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=req.top_k,
        with_payload=True,
        query_filter=qfilter,
    )

    # 2) build citations
    citations: list[Citation] = []
    hits: list[SearchHit] = []
    for i, p in enumerate(res.points, start=1):
        payload = p.payload or {}
        hits.append(
            SearchHit(
                score=float(p.score),
                source_id=payload.get("source_id"),
                chunk_id=payload.get("chunk_id"),
                text=payload.get("text"),
            )
        )
        citations.append(
            Citation(
                rank=i,
                score=float(p.score),
                source_id=payload.get("source_id"),
                chunk_id=payload.get("chunk_id"),
                type=(payload.get("meta") or {}).get("type"),
                text=payload.get("text"),
            )
        )

    if not citations:
        return AskResponse(
            query=req.query,
            answer="UNKNOWN (no evidence retrieved).",
            trace_paths=[],
            citations=[],
        )
    # 3) trace (optional)
    top_source_id = hits[0].source_id if hits and hits[0].source_id else None
    trace_paths = trace_k(top_source_id, req.trace_k) if (top_source_id and req.trace_k > 0) else []

    # 4) stub "answer" (for now): show what would be sent to LLM
    # keep it short
    rules = (
        "Rules:\n"
        "- Use ONLY the evidence below. If insufficient, answer 'UNKNOWN'.\n"
        "- Add citations like [1][2] after each claim.\n"
        "- Keep the answer concise and engineering-oriented.\n"
    )

    evidence = "\n".join(
        [f"[{c.rank}] ({c.source_id}) {((c.text or '').strip())}" for c in citations[: min(8, len(citations))]]
    )

    trace_txt = "\n".join([" -> ".join(p) for p in trace_paths]) if trace_paths else "(none)"

    prompt = (
        f"Question:\n{req.query}\n\n"
        f"{rules}\n"
        f"Trace paths (if any):\n{trace_txt}\n\n"
        f"Evidence:\n{evidence}\n"
    )

    answer = call_llm(prompt)
    return AskResponse(query=req.query, answer=answer, trace_paths=trace_paths, citations=citations)
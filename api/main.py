import os
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel, Field
from api.trace import trace_k 
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

app = FastAPI(title="AI MBSE Copilot")

# ---- runtime singletons (load once) ----
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "mbse_chunks_st"

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
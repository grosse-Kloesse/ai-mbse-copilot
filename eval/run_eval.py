import sys
import os
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")  # 本机访问 qdrant（端口映射）
COLLECTION = "mbse_chunks_st"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
if len(sys.argv) > 1:
    TOP_K = int(sys.argv[1])
else:
    TOP_K = 3

QUERIES_FILE = Path("eval/queries.jsonl")


def load_queries(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)

    rows = load_queries(QUERIES_FILE)

    hit_count = 0
    mrr_sum = 0.0

    for r in rows:
        query = r["query"]
        expected = r["expected_source_id"]

        qvec = model.encode(query, normalize_embeddings=True).tolist()
        res = client.query_points(
            collection_name=COLLECTION,
            query=qvec,
            limit=TOP_K,
            with_payload=True,
        )

        # 提取 top-k 的 source_id 列表
        top_ids = []
        for p in res.points:
            payload = p.payload or {}
            top_ids.append(payload.get("source_id"))

        # Recall@K
        hit = expected in top_ids
        hit_count += 1 if hit else 0

        # MRR@K
        rr = 0.0
        if hit:
            rank = top_ids.index(expected) + 1  # rank 从 1 开始
            rr = 1.0 / rank
        mrr_sum += rr

        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        print(f"Top{TOP_K}: {top_ids}")
        print(f"Hit@{TOP_K}: {hit} | RR: {rr:.3f}")

    recall = hit_count / len(rows) if rows else 0.0
    mrr = mrr_sum / len(rows) if rows else 0.0

    print("\n==== Summary ====")
    print(f"Queries: {len(rows)}")
    print(f"Recall@{TOP_K}: {recall:.3f}")
    print(f"MRR@{TOP_K}: {mrr:.3f}")


if __name__ == "__main__":
    main()

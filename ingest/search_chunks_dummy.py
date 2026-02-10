import hashlib
from qdrant_client import QdrantClient

COLLECTION = "mbse_chunks_dummy"
DIM = 64

def dummy_embed(text: str) -> list[float]:
    d = hashlib.sha256(text.encode("utf-8")).digest()
    return [((d[i % len(d)] / 255.0) * 2.0 - 1.0) for i in range(DIM)]

def main():
    client = QdrantClient(url="http://127.0.0.1:6333")

    query = "detect over-voltage"
    res = client.query_points(
        collection_name=COLLECTION,
        query=dummy_embed(query),
        limit=3,
        with_payload=True,
    )

    print(type(res))
    print(res)

    hits = res.points
    print(hits)


    print("Query:", query)
    for h in hits:
        p = getattr(h, "payload", None) or {}
        score = getattr(h, "score", None)
        if score is None:
            score = getattr(h, "distance", None)
        print("score=", score, "source_id=", p.get("source_id"), "chunk_id=", p.get("chunk_id"))

if __name__ == "__main__":
    main()

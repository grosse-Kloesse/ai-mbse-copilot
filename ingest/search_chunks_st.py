import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION_NAME = "mbse_chunks_st"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

query = sys.argv[1] if len(sys.argv) > 1 else "detect over-voltage"

def main():

    model = SentenceTransformer(MODEL_NAME)
    qvec = model.encode(query, normalize_embeddings=True).tolist()

    client = QdrantClient(url=QDRANT_URL)

    # Newer client versions: query_points (returns QueryResponse with .points)
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=3,
        with_payload=True,
    )

    print("Query:", query)
    for p in res.points:
        payload = p.payload or {}
        text = (payload.get("text") or "").replace("\n", " ")
        print(f"score={p.score:.4f} source_id={payload.get('source_id')} chunk_id={payload.get('chunk_id')}")
        print("  text:", text[:80])

if __name__ == "__main__":
    main()

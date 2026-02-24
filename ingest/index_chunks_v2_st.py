import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


QDRANT_URL = "http://127.0.0.1:6333"
CHUNKS_FILE = Path("data/processed/v2/chunks.jsonl")
COLLECTION_NAME = "mbse_chunks_v2"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
DIM = model.get_sentence_embedding_dimension()


def stable_uuid(chunk_id: str) -> str:
    # 生成一个稳定的 UUID（基于文本内容）
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
def load_chunks():
    chunks = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    print("Index script starting...")

    # 1) connect qdrant
    client = QdrantClient(url=QDRANT_URL)
    print("Connected to Qdrant:", QDRANT_URL)

    chunks = load_chunks()
    print("Loaded chunks:", len(chunks))
    print("First chunk_id:", chunks[0].get("chunk_id") if chunks else None)

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    print(f"Collection ready: {COLLECTION_NAME} (dim={DIM})")

    texts = [c.get("text", "") for c in chunks]
    vectors = model.encode(texts,normalize_embeddings=True).tolist()
    points = []
    for i,c in enumerate(chunks):
        chunk_id = c["chunk_id"]          # chunk 标识符（字符串）
        point_id = stable_uuid(chunk_id) # qqdrant point id (UUID string, stable)
        vector = vectors[i]

        payload = {
            "chunk_id": chunk_id,
            "source_id": c.get("source_id"),
            "source_type": c.get("source_type"),
            "source": c.get("source"),
            "meta": c.get("meta"),
            "text": texts[i],
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    BATCH = 64
    for i in range(0, len(points), BATCH):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + BATCH],
        )

    print(f"Upsert done: {len(points)} points")

    info = client.get_collection(COLLECTION_NAME)
    print("Collection points_count:", info.points_count)

if __name__ == "__main__":
    main()

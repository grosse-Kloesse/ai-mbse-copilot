import json
import hashlib
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
COLLECTION_NAME = "mbse_chunks_dummy"
DIM = 64  # 向量维度（dimension）——必须固定


def dummy_embed(text: str, dim: int = DIM) -> list[float]:
    """
    把文本变成一个“可重复”的假向量（deterministic dummy embedding）
    思路：用 sha256 产生稳定字节序列，然后映射到 [-1, 1] 的浮点数。
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes
    vec = []
    # 为了凑够 dim 个数，我们重复利用 digest 的字节
    for i in range(dim):
        b = digest[i % len(digest)]
        # 把 0..255 映射到 -1..1
        vec.append((b / 255.0) * 2.0 - 1.0)
    return vec


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {CHUNKS_FILE}")

    chunks = load_chunks(CHUNKS_FILE)
    print(f"Loaded chunks: {len(chunks)}")

    # 连接本机 Qdrant（因为 docker-compose 把 6333 映射到本机）
    client = QdrantClient(url="http://127.0.0.1:6333")

    # 重新创建 collection（recreate = 删除旧的再建新的，保证可重复）
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    print(f"Collection ready: {COLLECTION_NAME} (dim={DIM})")

    # 组装 points 并分批 upsert
    points = []
    for i,c in enumerate(chunks):
        chunk_id = c["chunk_id"]          # 我们用 chunk_id 做 point id（字符串也可以）
        point_id = i # 也可以用数字 id，但要保证唯一且稳定（这里简单用行号）
        text = c.get("text", "")
        vector = dummy_embed(text, DIM)

        payload = {
            "chunk_id": chunk_id,
            "source_id": c.get("source_id"),
            "source_type": c.get("source_type"),
            "source": c.get("source"),
            "meta": c.get("meta"),
            "text": text,
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

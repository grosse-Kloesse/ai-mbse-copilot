import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CHUNKS = Path("data/processed/bike/chunks.jsonl")
DEFAULT_EMBED_DIR = Path("data/processed/bike/embeddings_e5_small")


CONTEXT_QUERY_KEYWORDS = [
    "diagramm",
    "diagram",
    "package",
    "paket",
    "modelliert",
    "erscheint",
    "appears",
    "appear",
    "kontext",
    "context",
    "pfad",
    "path",
    "gehört",
    "gehort",
    "zu welchem",
    "in welchem",
    "wo befindet",
    "where",
    "which diagram",
    "which package",
]

DEFAULT_TYPE_WEIGHTS = {
    "node": 1.00,
    "relation": 1.00,
    "diagram": 0.92,
    "package": 0.90,
}

CONTEXT_TYPE_WEIGHTS = {
    "node": 1.00,
    "relation": 1.00,
    "diagram": 1.15,
    "package": 1.10,
}

STOPWORDS = {
    "wie", "was", "wo", "wann", "warum", "welche", "welcher", "welchem", "welchen",
    "ist", "sind", "im", "in", "am", "an", "auf", "vom", "von", "der", "die", "das",
    "dem", "den", "des", "ein", "eine", "einem", "einer", "einen",
    "maximale", "maximal", "groß", "gross", "größe", "grosse",
    "erscheint", "diagramm", "diagram", "package", "paket",
    "which", "where", "what", "is", "the", "a", "an", "of", "in",
    "context", "path", "appears", "appear",
}


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def load_embeddings(embed_dir: Path) -> np.ndarray:
    emb_path = embed_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    return np.load(emb_path)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    return s


def is_context_query(query: str) -> bool:
    if not query:
        return False
    q = query.lower().strip()
    return any(keyword in q for keyword in CONTEXT_QUERY_KEYWORDS)


def get_type_weights(query: str) -> Dict[str, float]:
    if is_context_query(query):
        return CONTEXT_TYPE_WEIGHTS
    return DEFAULT_TYPE_WEIGHTS


def tokenize_query(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9_\-]+", query.lower())
    tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]
    return tokens


def entity_overlap_bonus(query: str, chunk: Dict[str, Any]) -> float:
    """
    与 test_real_embedding.py 保持一致：
    - 只对 diagram/package 做额外 bonus
    - 只在 context query 时启用
    """
    chunk_type = chunk.get("chunk_type")
    if chunk_type not in {"diagram", "package"}:
        return 1.0

    if not is_context_query(query):
        return 1.0

    tokens = tokenize_query(query)
    if not tokens:
        return 1.0

    text = (chunk.get("text") or "").lower()

    matched = 0
    for token in tokens:
        if token in text:
            matched += 1

    if matched == 0:
        return 0.97
    if matched == 1:
        return 1.05
    return 1.10


def get_adjusted_score(raw_score: float, chunk: Dict[str, Any], query: str) -> float:
    chunk_type = chunk.get("chunk_type", "")
    weights = get_type_weights(query)
    type_weight = weights.get(chunk_type, 1.0)
    bonus = entity_overlap_bonus(query, chunk)
    return raw_score * type_weight * bonus


def search(
    model: SentenceTransformer,
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int = 10,
) -> Tuple[List[Dict[str, Any]], bool, Dict[str, float], List[str]]:
    q = model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]

    raw_scores = embeddings @ q

    scored_results = []
    for idx, ch in enumerate(chunks):
        raw_score = float(raw_scores[idx])
        adjusted_score = get_adjusted_score(raw_score, ch, query)

        scored_results.append(
            {
                "raw_score": raw_score,
                "adjusted_score": adjusted_score,
                "chunk": ch,
            }
        )

    scored_results.sort(key=lambda x: x["adjusted_score"], reverse=True)
    top_results = scored_results[:top_k]

    results: List[Dict[str, Any]] = []
    for rank, item in enumerate(top_results, start=1):
        ch = item["chunk"]
        results.append(
            {
                "rank": rank,
                "score": item["adjusted_score"],
                "raw_score": item["raw_score"],
                "chunk_id": ch.get("chunk_id"),
                "chunk_type": ch.get("chunk_type"),
                "name": ch.get("name"),
                "type": ch.get("type"),
                "source_id": ch.get("source_id"),
                "text": ch.get("text"),
            }
        )

    return results, is_context_query(query), get_type_weights(query), tokenize_query(query)


def print_results(
    results: List[Dict[str, Any]],
    context_mode: bool,
    type_weights: Dict[str, float],
    query_tokens: List[str],
) -> None:
    print(f"Context query mode: {context_mode}")
    print(f"Type weights      : {type_weights}")
    print(f"Query tokens      : {query_tokens}")

    for r in results:
        print("=" * 100)
        print(f"rank      : {r['rank']}")
        print(f"score     : {r['score']:.4f}")
        print(f"raw_score : {r['raw_score']:.4f}")
        print(f"chunk_id  : {r['chunk_id']}")
        print(f"chunk_type: {r['chunk_type']}")
        print(f"name      : {r['name']}")
        print(f"type      : {r['type']}")
        print(f"source_id : {r['source_id']}")
        print("-" * 100)
        print(r["text"])
        print()


def pick_best_evidence(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    关键修正：
    不再做二次重排，直接沿用 search() 的结果顺序。
    """
    return results[:3]


def extract_property_line(text: str) -> str | None:
    lines = text.splitlines()
    for line in lines:
        s = line.strip()
        if s.startswith("- ") and " = " in s:
            return s[2:]
    return None


def extract_requirement_text(text: str) -> str | None:
    m = re.search(r"^Text:\s*(.+)$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def extract_diagrams(text: str) -> List[str]:
    lines = text.splitlines()
    out: List[str] = []
    in_section = False
    for line in lines:
        s = line.strip()
        if s.startswith("Appears in diagrams:"):
            in_section = True
            continue
        if in_section:
            if not s.startswith("- "):
                break
            out.append(s[2:].strip())
    return out


def extract_package_path(text: str) -> str | None:
    m = re.search(r"^Path:\s*(.+)$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    m = re.search(r"^Package:\s*(.+)$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def answer_query(query: str, evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "没有检索到可用证据。"

    best = evidence[0]
    best_text = best.get("text", "") or ""
    best_name = best.get("name") or "Unbekannt"
    best_chunk_type = best.get("chunk_type")

    # 1) 图/包/出现位置类
    if is_context_query(query):
        diagrams = extract_diagrams(best_text)

        if best_chunk_type == "diagram":
            return f"答案：`{best_name}` 是最相关的图，检索结果表明目标最可能出现在这个 diagram 中。"

        if diagrams:
            joined = "; ".join(f"`{d}`" for d in diagrams)
            return f"答案：该实体出现在这些 diagram 中：{joined}。"

        pkg = extract_package_path(best_text)
        if pkg:
            return f"答案：该实体位于包路径 `{pkg}` 中。"

        return f"答案：最相关证据指向 `{best_name}`。"

    # 2) 属性类
    property_line = extract_property_line(best_text)
    if property_line:
        return f"答案：`{best_name}` 的相关属性为 `{property_line}`。"

    req_text = extract_requirement_text(best_text)
    if req_text:
        return f"答案：`{best_name}` 的要求是：{req_text}"

    # 3) relation 类
    if best_chunk_type == "relation":
        first_line = best_text.splitlines()[0].strip()
        return f"答案：最相关关系为：{first_line}"

    return f"答案：最相关结果是 `{best_name}`。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--embed_dir", type=Path, default=DEFAULT_EMBED_DIR)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    chunks = load_chunks(args.chunks)
    print(f"Loaded chunks: {len(chunks)}")

    print(f"Loading existing embeddings from: {args.embed_dir / 'embeddings.npy'}")
    embeddings = load_embeddings(args.embed_dir)
    print(f"Embeddings shape: {embeddings.shape}")

    results, context_mode, type_weights, query_tokens = search(
        model=model,
        query=args.query,
        chunks=chunks,
        embeddings=embeddings,
        top_k=args.top_k,
    )

    print_results(results, context_mode, type_weights, query_tokens)

    evidence = pick_best_evidence(args.query, results)

    print("#" * 100)
    print("SELECTED EVIDENCE")
    print("#" * 100)
    for i, e in enumerate(evidence, start=1):
        print(f"{i}. {e['chunk_type']} | {e['name']} | score={e['score']:.4f}")

    print("#" * 100)
    print("FINAL ANSWER")
    print("#" * 100)
    print(answer_query(args.query, evidence))


if __name__ == "__main__":
    main()
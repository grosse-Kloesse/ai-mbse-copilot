#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# =========================
# Config
# =========================

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_CHUNKS = "data/processed/bike/chunks.jsonl"
DEFAULT_EMBEDDINGS = "data/processed/bike/embeddings_e5_small/embeddings.npy"

TYPE_WEIGHTS_BY_MODE = {
    "default": {
        "node": 1.00,
        "relation": 1.00,
        "diagram": 1.00,
        "package": 1.00,
    },
    "diagram": {
        "node": 0.98,
        "relation": 0.98,
        "diagram": 1.22,
        "package": 0.98,
    },
    "package": {
        "node": 1.00,
        "relation": 0.98,
        "diagram": 0.95,
        "package": 1.18,
    },
    "relation": {
        "node": 1.00,
        "relation": 1.10,
        "diagram": 0.95,
        "package": 1.00,
    },
    "belongs_to_relation": {
        "node": 1.00,
        "relation": 1.18,
        "diagram": 0.90,
        "package": 0.95,
    },
}

# 不同 mode 下，不同 chunk_type 的基础 lexical bonus
LEXICAL_BONUS_BY_MODE = {
    "default": {
        "node": 0.05,
        "relation": 0.05,
        "diagram": 0.04,
        "package": 0.04,
    },
    "diagram": {
        "node": 0.01,
        "relation": 0.01,
        "diagram": 0.08,
        "package": 0.02,
    },
    "package": {
        "node": 0.04,
        "relation": 0.02,
        "diagram": 0.02,
        "package": 0.16,
    },
    "relation": {
        "node": 0.04,
        "relation": 0.08,
        "diagram": 0.01,
        "package": 0.02,
    },
    "belongs_to_relation": {
        "node": 0.05,
        "relation": 0.14,
        "diagram": 0.01,
        "package": 0.02,
    },
}

# 每种 mode 的 lexical bonus 上限
LEXICAL_CAP_BY_MODE = {
    "default": 0.16,
    "diagram": 0.20,
    "package": 0.26,
    "relation": 0.20,
    "belongs_to_relation": 0.32,
}

QUESTION_STOPWORDS = {
    "in", "welchem", "welcher", "welche", "welches",
    "diagramm", "diagram", "package", "paket",
    "liegt", "erscheint", "enthaelt", "enthält",
    "zu", "system", "gehoert", "gehört", "das",
    "der", "die", "dem", "den", "ein", "eine", "einem",
    "ist", "sind", "von", "im", "am", "an", "auf",
    "was", "wer", "wo", "wie", "warum", "wieso",
    "enthalt", "enthaelt", "erscheinen",
}

BELONGS_PATTERNS = [
    r"\bzu welchem system gehoert\b",
    r"\bzu welchem system gehört\b",
    r"\bwelchem system gehoert\b",
    r"\bwelchem system gehört\b",
    r"\bgehoert .* zu\b",
    r"\bgehört .* zu\b",
]

PACKAGE_PATTERNS = [
    r"\bin welchem package\b",
    r"\bwelches package\b",
    r"\bpackage\b",
    r"\bpaket\b",
    r"\benthaelt .* package\b",
    r"\benthält .* package\b",
]

DIAGRAM_PATTERNS = [
    r"\bin welchem diagramm\b",
    r"\bwelches diagramm\b",
    r"\bdiagramm\b",
    r"\bdiagram\b",
    r"\berscheint\b",
]

RELATION_PATTERNS = [
    r"\bwelche relation\b",
    r"\bwelcher zusammenhang\b",
    r"\bwie haengt .* zusammen\b",
    r"\bwie hängt .* zusammen\b",
]


# =========================
# Utilities
# =========================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = (
        text.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
    )
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    return [t for t in norm.split() if t]


def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec) + 1e-12
    doc_norms = np.linalg.norm(doc_matrix, axis=1) + 1e-12
    return (doc_matrix @ query_vec) / (doc_norms * query_norm)


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_query(model, tokenizer, query: str, device: str = "cpu") -> np.ndarray:
    encoded_input = tokenizer(
        [f"query: {query}"],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = average_pool(model_output.last_hidden_state, encoded_input["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy()


def load_chunks(path: str) -> List[Dict[str, Any]]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def get_chunk_type(chunk: Dict[str, Any]) -> str:
    return chunk.get("chunk_type", "unknown")


def get_chunk_name(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("name", "") or "")


def get_meta(chunk: Dict[str, Any], key: str, default=None):
    if key in chunk:
        return chunk.get(key, default)
    meta = chunk.get("metadata", {})
    if isinstance(meta, dict):
        return meta.get(key, default)
    return default


def chunk_text_for_match(chunk: Dict[str, Any]) -> str:
    parts = [
        str(chunk.get("name", "") or ""),
        str(chunk.get("type", "") or ""),
        str(chunk.get("chunk_type", "") or ""),
        str(chunk.get("text", "") or ""),
        str(chunk.get("content", "") or ""),
        str(chunk.get("package_path", "") or ""),
        str(chunk.get("path", "") or ""),
    ]

    for key in ["source_name", "target_name", "source_type", "target_type"]:
        if key in chunk:
            parts.append(str(chunk.get(key, "") or ""))

    meta = chunk.get("metadata", {})
    if isinstance(meta, dict):
        for key in [
            "package", "package_path", "source_name", "target_name",
            "source_type", "target_type", "name", "type"
        ]:
            if key in meta:
                parts.append(str(meta.get(key, "") or ""))

    return normalize_text(" ".join(parts))


def contains_full_phrase(text_norm: str, phrase_norm: str) -> bool:
    if not text_norm or not phrase_norm:
        return False
    return phrase_norm in text_norm


def contains_any_token(text_norm: str, tokens: List[str]) -> bool:
    if not text_norm or not tokens:
        return False
    text_tokens = set(text_norm.split())
    return any(t in text_tokens for t in tokens if t)


# =========================
# Mode / Entity Detection
# =========================

def detect_mode(query: str) -> str:
    q = normalize_text(query)

    for p in BELONGS_PATTERNS:
        if re.search(p, q):
            return "belongs_to_relation"

    for p in PACKAGE_PATTERNS:
        if re.search(p, q):
            return "package"

    for p in DIAGRAM_PATTERNS:
        if re.search(p, q):
            return "diagram"

    for p in RELATION_PATTERNS:
        if re.search(p, q):
            return "relation"

    return "default"


def build_name_catalog(chunks: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for c in chunks:
        name = get_chunk_name(c)
        if name:
            names.add(name)
        for key in ["source_name", "target_name"]:
            value = get_meta(c, key, "")
            if value:
                names.add(str(value))
    return sorted(names, key=lambda x: len(normalize_text(x)), reverse=True)


def extract_target_entity(query: str, name_catalog: List[str]) -> Optional[str]:
    q_norm = normalize_text(query)

    for name in name_catalog:
        n_norm = normalize_text(name)
        if not n_norm:
            continue
        if n_norm in q_norm:
            return name

    toks = [t for t in tokenize(query) if t not in QUESTION_STOPWORDS]
    if not toks:
        return None

    candidate = " ".join(toks).strip()
    if candidate:
        cand_norm = normalize_text(candidate)
        best_name = None
        best_score = 0.0
        cand_set = set(cand_norm.split())

        for name in name_catalog:
            n_norm = normalize_text(name)
            n_set = set(n_norm.split())
            if not n_set:
                continue
            inter = len(cand_set & n_set)
            union = len(cand_set | n_set)
            score = inter / union if union else 0.0
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= 0.5:
            return best_name

    return None


# =========================
# Bonus Logic
# =========================

def lexical_overlap_score(query_tokens: List[str], chunk_text_norm: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = set(chunk_text_norm.split())
    if not text_tokens:
        return 0.0

    matched = 0
    total = 0

    for token in query_tokens:
        if not token:
            continue
        total += 1
        if " " in token:
            if token in chunk_text_norm:
                matched += 1
        else:
            if token in text_tokens:
                matched += 1

    return matched / max(total, 1)


def exact_name_match_bonus(
    chunk: Dict[str, Any],
    target_entity: Optional[str],
    mode: str,
) -> float:
    if not target_entity:
        return 0.0

    chunk_type = get_chunk_type(chunk)
    name_norm = normalize_text(get_chunk_name(chunk))
    target_norm = normalize_text(target_entity)

    source_name = normalize_text(str(get_meta(chunk, "source_name", "") or ""))
    target_name = normalize_text(str(get_meta(chunk, "target_name", "") or ""))

    bonus = 0.0

    # ===== diagram mode =====
    if mode == "diagram":
        if chunk_type == "diagram":
            text_norm = chunk_text_for_match(chunk)

            # 完整实体短语命中最重要，例如 "e motor"
            if target_norm and contains_full_phrase(text_norm, target_norm):
                bonus += 0.20

            # diagram 名称刚好等于实体名，少见，但保留
            if name_norm == target_norm:
                bonus += 0.03

        elif chunk_type == "package":
            text_norm = chunk_text_for_match(chunk)
            if target_norm and contains_full_phrase(text_norm, target_norm):
                bonus += 0.02

        elif chunk_type == "node":
            if name_norm == target_norm:
                bonus += 0.02

        elif chunk_type == "relation":
            if source_name == target_norm or target_name == target_norm:
                bonus += 0.02

        return bonus

    # ===== package mode =====
    if mode == "package":
        if chunk_type == "package":
            if name_norm == target_norm:
                bonus += 0.22
            else:
                text_norm = chunk_text_for_match(chunk)
                if target_norm and contains_full_phrase(text_norm, target_norm):
                    bonus += 0.12

        elif chunk_type == "node":
            if name_norm == target_norm:
                bonus += 0.10

        elif chunk_type == "relation":
            if source_name == target_norm or target_name == target_norm:
                bonus += 0.05

        elif chunk_type == "diagram":
            text_norm = chunk_text_for_match(chunk)
            if target_norm and contains_full_phrase(text_norm, target_norm):
                bonus += 0.02

        return bonus

    # ===== belongs_to_relation mode =====
    if mode == "belongs_to_relation":
        if chunk_type == "relation":
            if source_name == target_norm:
                bonus += 0.24
            elif target_name == target_norm:
                bonus += 0.10

        elif chunk_type == "node":
            if name_norm == target_norm:
                bonus += 0.14

        elif chunk_type == "package":
            text_norm = chunk_text_for_match(chunk)
            if target_norm and contains_full_phrase(text_norm, target_norm):
                bonus += 0.04

        elif chunk_type == "diagram":
            text_norm = chunk_text_for_match(chunk)
            if target_norm and contains_full_phrase(text_norm, target_norm):
                bonus += 0.02

        return bonus

    # ===== relation mode =====
    if mode == "relation":
        if chunk_type == "relation":
            if source_name == target_norm or target_name == target_norm:
                bonus += 0.12
        elif chunk_type == "node":
            if name_norm == target_norm:
                bonus += 0.06
        return bonus

    # ===== default =====
    if chunk_type == "node" and name_norm == target_norm:
        bonus += 0.08
    elif chunk_type == "relation" and (source_name == target_norm or target_name == target_norm):
        bonus += 0.08
    elif chunk_type == "package" and name_norm == target_norm:
        bonus += 0.08
    elif chunk_type == "diagram":
        text_norm = chunk_text_for_match(chunk)
        if target_norm and contains_full_phrase(text_norm, target_norm):
            bonus += 0.05

    return bonus


def token_bonus(query_tokens: List[str], chunk: Dict[str, Any], mode: str) -> float:
    chunk_type = get_chunk_type(chunk)
    base = LEXICAL_BONUS_BY_MODE.get(mode, LEXICAL_BONUS_BY_MODE["default"]).get(chunk_type, 0.0)

    if base <= 0:
        return 0.0

    text_norm = chunk_text_for_match(chunk)
    overlap = lexical_overlap_score(query_tokens, text_norm)

    # 限制 token 数放大效应
    token_factor = min(len(set(query_tokens)), 2)

    return base * overlap * token_factor


def compute_lexical_bonus(
    query_tokens: List[str],
    chunk: Dict[str, Any],
    mode: str,
    target_entity: Optional[str],
) -> float:
    b1 = token_bonus(query_tokens, chunk, mode)
    b2 = exact_name_match_bonus(chunk, target_entity, mode)
    cap = LEXICAL_CAP_BY_MODE.get(mode, 0.18)
    return min(b1 + b2, cap)


# =========================
# Ranking
# =========================

@dataclass
class RankedItem:
    rank: int
    score: float
    raw_score: float
    type_weight: float
    lexical_bonus: float
    chunk: Dict[str, Any]


def build_query_tokens(query: str, mode: str, target_entity: Optional[str]) -> List[str]:
    if target_entity:
        entity_norm = normalize_text(target_entity)
        entity_tokens = [t for t in tokenize(target_entity) if t not in QUESTION_STOPWORDS]

        # 同时保留完整短语和拆词
        # E-Motor -> ["e motor", "e", "motor"]
        query_tokens: List[str] = []
        if entity_norm:
            query_tokens.append(entity_norm)
        query_tokens.extend(entity_tokens)

        if mode == "belongs_to_relation":
            query_tokens.extend(["gehoert", "system"])

        # 去重保持顺序
        dedup = []
        seen = set()
        for t in query_tokens:
            if t and t not in seen:
                dedup.append(t)
                seen.add(t)
        return dedup

    return [t for t in tokenize(query) if t not in QUESTION_STOPWORDS]


def rank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    model,
    tokenizer,
    top_k: int = 10,
    device: str = "cpu",
) -> Tuple[List[RankedItem], str, List[str], Optional[str]]:
    query_vec = encode_query(model, tokenizer, query, device=device)
    raw_scores = cosine_similarity_matrix(query_vec, embeddings)

    mode = detect_mode(query)
    weights = TYPE_WEIGHTS_BY_MODE.get(mode, TYPE_WEIGHTS_BY_MODE["default"])

    name_catalog = build_name_catalog(chunks)
    target_entity = extract_target_entity(query, name_catalog)
    query_tokens = build_query_tokens(query, mode, target_entity)

    ranked: List[RankedItem] = []

    for idx, chunk in enumerate(chunks):
        raw = float(raw_scores[idx])
        ctype = get_chunk_type(chunk)
        type_weight = weights.get(ctype, 1.0)
        lexical_bonus = compute_lexical_bonus(query_tokens, chunk, mode, target_entity)

        score = raw * type_weight + lexical_bonus

        ranked.append(
            RankedItem(
                rank=0,
                score=score,
                raw_score=raw,
                type_weight=type_weight,
                lexical_bonus=lexical_bonus,
                chunk=chunk,
            )
        )

    ranked.sort(key=lambda x: x.score, reverse=True)

    for i, item in enumerate(ranked[:top_k], start=1):
        item.rank = i

    return ranked[:top_k], mode, query_tokens, target_entity


# =========================
# Printing
# =========================

def pretty_chunk_header(chunk: Dict[str, Any]) -> str:
    ctype = get_chunk_type(chunk)
    name = get_chunk_name(chunk) or "None"
    typ = str(chunk.get("type", "") or get_meta(chunk, "type", "") or "None")
    cid = str(chunk.get("chunk_id", "") or "None")
    sid = str(chunk.get("source_id", "") or "None")
    lines = [
        f"chunk_id     : {cid}",
        f"chunk_type   : {ctype}",
        f"name         : {name}",
        f"type         : {typ}",
        f"source_id    : {sid}",
    ]
    return "\n".join(lines)


def chunk_text_preview(chunk: Dict[str, Any]) -> str:
    text = str(chunk.get("text", "") or chunk.get("content", "") or "").strip()
    if not text:
        fields = []
        for key in ["package_path", "path", "source_name", "target_name"]:
            val = get_meta(chunk, key, "")
            if val:
                fields.append(f"{key}: {val}")
        return "\n".join(fields) if fields else "(no text)"
    return text


def print_ranked_results(
    ranked: List[RankedItem],
    mode: str,
    query_tokens: List[str],
    target_entity: Optional[str],
):
    print(f"Detected mode     : {mode}")
    print(f"Query tokens      : {query_tokens}")
    if target_entity:
        print(f"Target entity     : {target_entity}")

    print("=" * 100)

    for item in ranked:
        print(f"rank         : {item.rank}")
        print(f"score        : {item.score:.4f}")
        print(f"raw_score    : {item.raw_score:.4f}")
        print(f"type_weight  : {item.type_weight:.2f}")
        print(f"lexical_bonus: {item.lexical_bonus:.4f}")
        print(pretty_chunk_header(item.chunk))
        print("-" * 100)
        print(chunk_text_preview(item.chunk))
        print()
        print("=" * 100)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--top_k", type=int, default=10, help="Top K results")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS, help="Path to chunks.jsonl")
    parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS, help="Path to embeddings.npy")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    args = parser.parse_args()

    if not os.path.exists(args.chunks):
        raise FileNotFoundError(f"Chunks file not found: {args.chunks}")
    if not os.path.exists(args.embeddings):
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    chunks = load_chunks(args.chunks)
    print(f"Loaded chunks: {len(chunks)}")

    print(f"Loading existing embeddings from: {args.embeddings}")
    embeddings = np.load(args.embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

    ranked, mode, query_tokens, target_entity = rank_chunks(
        query=args.query,
        chunks=chunks,
        embeddings=embeddings,
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device=args.device,
    )

    print_ranked_results(ranked, mode, query_tokens, target_entity)


if __name__ == "__main__":
    main()
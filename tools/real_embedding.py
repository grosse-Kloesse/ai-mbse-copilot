#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import io
import json
import logging
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from transformers.utils import logging as hf_transformers_logging
    hf_transformers_logging.set_verbosity_error()
except Exception:
    pass

try:
    from huggingface_hub.utils import logging as hf_hub_logging
    hf_hub_logging.set_verbosity_error()
except Exception:
    pass

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_CHUNKS_PATH = "data/processed/bike/chunks.jsonl"
DEFAULT_EMBEDDINGS_PATH = "data/processed/bike/embeddings_e5_small/embeddings.npy"


@dataclass
class QueryFrame:
    raw_query: str
    intent: str = "unknown"
    target_entity: Optional[str] = None
    asked_slot: Optional[str] = None
    expected_answer_type: str = "text"
    confidence: float = 0.0
    rewritten_queries: List[str] = field(default_factory=list)
    reasoning_mode: str = "semantic_fallback"


@dataclass
class RetrievalStep:
    name: str
    description: str


@dataclass
class RetrievalPlan:
    intent: str
    steps: List[RetrievalStep] = field(default_factory=list)


@dataclass
class ScoredResult:
    rank: int
    score: float
    raw_score: float
    type_weight: float
    lexical_bonus: float
    source: str
    chunk: Dict[str, Any]


@dataclass
class AnswerBundle:
    final_answer: str
    evidence: List[ScoredResult] = field(default_factory=list)


STOPWORDS = {
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einem", "einer", "eines",
    "und", "oder", "ist", "sind", "war", "waren",
    "in", "im", "am", "an", "auf", "zu", "zum", "zur",
    "von", "für", "mit", "ohne", "aus", "bei", "nach",
    "welche", "welcher", "welches", "welchem", "welchen",
    "was", "wie", "wo", "wann", "warum",
    "enthält", "enthaelt", "liegt", "erscheint", "gehört", "gehoert",
    "system", "package", "diagramm", "elemente",
    "hat", "haben",
    "hoch", "gross", "groß",
}

PROPERTY_KEYWORDS = {
    "spannung": ["spannung", "volt", "voltage", "u", "u_max", "u_nom"],
    "strom": ["strom", "ampere", "current", "i", "i_max", "i_nom"],
    "leistung": ["leistung", "power", "watt", "p"],
    "drehmoment": ["drehmoment", "torque", "nm"],
    "geschwindigkeit": ["geschwindigkeit", "speed", "maximalgeschwindigkeit", "höchstgeschwindigkeit", "hoechstgeschwindigkeit"],
    "masse": ["masse", "gewicht", "mass", "weight", "kg"],
    "beschleunigung": ["beschleunigung", "m/s2", "m/s²"],
    "beschleunigungszeit": ["beschleunigungszeit", "zeit", "s"],
}

INTENT_PATTERNS = {
    "locate_in_diagram": [
        r"\bin welchem diagramm erscheint\b",
        r"\bwelches diagramm enthält\b",
        r"\bwelches diagramm enthaelt\b",
        r"\bwo erscheint\b",
    ],
    "locate_in_package": [
        r"\bin welchem package liegt\b",
        r"\bwelches package enthält\b",
        r"\bwelches package enthaelt\b",
        r"\bwo liegt\b",
    ],
    "find_parent_or_owner": [
        r"\bzu welchem system gehört\b",
        r"\bzu welchem system gehoert\b",
        r"\bzu welchem block gehört\b",
        r"\bzu welchem block gehoert\b",
        r"\bgehört zu welchem\b",
        r"\bgehoert zu welchem\b",
    ],
    "list_members": [
        r"\bwelche elemente enthält\b",
        r"\bwelche elemente enthaelt\b",
        r"\bwas ist in .* package\b",
        r"\bwelche blöcke enthält\b",
        r"\bwelche bloecke enthaelt\b",
    ],
    "ask_property_value": [
        r"\bwie hoch ist\b",
        r"\bwie groß ist\b",
        r"\bwie gross ist\b",
        r"\bwas ist die spannung\b",
        r"\bwas ist der strom\b",
        r"\bwas ist die leistung\b",
        r"\bwelche properties hat\b",
        r"\bwelche eigenschaften hat\b",
    ],
    "find_relations": [
        r"\bwelche relationen hat\b",
        r"\bmit wem verbunden\b",
        r"\bwelche beziehungen hat\b",
        r"\bwelche relation hat\b",
    ],
}

TYPE_WEIGHTS_BY_INTENT = {
    "locate_in_diagram": {"node": 1.00, "relation": 0.98, "diagram": 1.18, "package": 0.95},
    "locate_in_package": {"node": 1.00, "relation": 0.98, "diagram": 0.92, "package": 1.18},
    "find_parent_or_owner": {"node": 1.00, "relation": 1.18, "diagram": 0.90, "package": 0.95},
    "list_members": {"node": 0.98, "relation": 0.95, "diagram": 0.90, "package": 1.20},
    "ask_property_value": {"node": 1.18, "relation": 0.98, "diagram": 0.90, "package": 0.92},
    "find_relations": {"node": 1.00, "relation": 1.18, "diagram": 0.92, "package": 0.90},
    "unknown": {"node": 1.00, "relation": 1.00, "diagram": 1.00, "package": 1.00},
}


def suppress_model_load_noise():
    class _SilentStderr(io.StringIO):
        def write(self, s):
            blocked_keywords = [
                "Warning: You are sending unauthenticated requests to the HF Hub",
                "BertModel LOAD REPORT",
                "embeddings.position_ids",
                "UNEXPECTED",
                "Loading weights:",
                "Materializing param=",
                "can be ignored when loading from different task/architecture",
            ]
            if any(k in s for k in blocked_keywords):
                return len(s)
            return super().write(s)

    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(_SilentStderr())


def load_model_quietly(model_name: str) -> SentenceTransformer:
    stdout_cm, stderr_cm = suppress_model_load_noise()
    with stdout_cm, stderr_cm:
        model = SentenceTransformer(model_name)
    return model


def load_chunks(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks file not found: {p}")

    chunks = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_embeddings(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embeddings file not found: {p}")

    emb = np.load(p)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    return emb


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text).lower().strip())
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = (
        text.replace("ä", "ae")
            .replace("ö", "oe")
            .replace("ü", "ue")
            .replace("ß", "ss")
    )
    text = re.sub(r"[^a-z0-9\s/=.:]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_query_tokens(query: str) -> List[str]:
    q = normalize_text(query)
    return [t for t in q.split() if t and t not in STOPWORDS and len(t) > 1]


def get_chunk_text(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("text", "") or "")


def get_chunk_name(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("name", "") or "")


def get_chunk_type(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("chunk_type", "") or "")


def safe_get_type_weights(intent: str) -> Dict[str, float]:
    return TYPE_WEIGHTS_BY_INTENT.get(intent, TYPE_WEIGHTS_BY_INTENT["unknown"])


def build_entity_catalog(chunks: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for ch in chunks:
        name = get_chunk_name(ch)
        if name:
            names.add(name)
    return sorted(names, key=lambda x: len(normalize_text(x)), reverse=True)


def extract_target_entity(query: str, chunks: List[Dict[str, Any]]) -> Optional[str]:
    raw = query.strip()
    lower = raw.lower()

    patterns = [
        r"in welchem diagramm erscheint (?:der|die|das)\s+(.+?)\??$",
        r"in welchem package liegt (?:der|die|das)\s+(.+?)\??$",
        r"zu welchem system gehört (?:der|die|das)\s+(.+?)\??$",
        r"zu welchem system gehoert (?:der|die|das)\s+(.+?)\??$",
        r"zu welchem block gehört (?:der|die|das)\s+(.+?)\??$",
        r"zu welchem block gehoert (?:der|die|das)\s+(.+?)\??$",
        r"welche elemente enthält (?:das|der|die)\s+(.+?)\s+package\??$",
        r"welche elemente enthaelt (?:das|der|die)\s+(.+?)\s+package\??$",
        r"welche properties hat (?:der|die|das)\s+(.+?)\??$",
        r"welche eigenschaften hat (?:der|die|das)\s+(.+?)\??$",
        r"was ist (?:der|die|das)\s+.+?\s+(?:des|der)\s+(.+?)\??$",
        r"wie hoch ist (?:der|die|das)\s+(.+?)\??$",
        r"wie groß ist (?:der|die|das)\s+(.+?)\??$",
        r"wie gross ist (?:der|die|das)\s+(.+?)\??$",
    ]

    for pat in patterns:
        m = re.search(pat, lower, flags=re.IGNORECASE)
        if m:
            start, end = m.span(1)
            return raw[start:end].strip(" ?\"'`")

    catalog = build_entity_catalog(chunks)
    q_norm = normalize_text(query)

    for name in catalog:
        if normalize_text(name) in q_norm:
            return name

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', raw)
    flat = [x for pair in quoted for x in pair if x]
    if flat:
        return flat[0].strip()

    return None


def detect_asked_slot(query: str) -> Optional[str]:
    q = normalize_text(query)
    best_slot = None
    best_len = 0
    for slot, kws in PROPERTY_KEYWORDS.items():
        for kw in kws:
            kw_norm = normalize_text(kw)
            if kw_norm in q and len(kw_norm) > best_len:
                best_slot = slot
                best_len = len(kw_norm)
    return best_slot


def detect_intent(query: str) -> Tuple[str, float]:
    q = normalize_text(query)

    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return intent, 0.95

    asked_slot = detect_asked_slot(query)
    if asked_slot:
        return "ask_property_value", 0.75

    return "unknown", 0.20


def rewrite_queries(frame: QueryFrame) -> List[str]:
    target = frame.target_entity or frame.raw_query

    if frame.intent == "locate_in_package":
        return [target, f"{target} package", f"{target} path"]

    if frame.intent == "locate_in_diagram":
        return [target, f"{target} appears in diagrams", f"{target} diagram"]

    if frame.intent == "find_parent_or_owner":
        return [target, f"{target} source target relation", f"{target} connected target"]

    if frame.intent == "list_members":
        return [target, f"{target} package elements", f"{target} contains elements"]

    if frame.intent == "ask_property_value":
        if frame.asked_slot:
            return [target, f"{target} {frame.asked_slot}", f"{target} properties"]
        return [target, f"{target} properties"]

    if frame.intent == "find_relations":
        return [target, f"{target} relations", f"{target} connected relations"]

    return [frame.raw_query]


def expected_answer_type(intent: str) -> str:
    mapping = {
        "locate_in_package": "path",
        "locate_in_diagram": "diagram_list",
        "find_parent_or_owner": "relation_target",
        "list_members": "entity_list",
        "ask_property_value": "property_value",
        "find_relations": "relation_list",
        "unknown": "text",
    }
    return mapping.get(intent, "text")


def reasoning_mode(intent: str) -> str:
    if intent in {"locate_in_package", "locate_in_diagram", "list_members", "ask_property_value"}:
        return "direct_field_lookup"
    if intent in {"find_parent_or_owner", "find_relations"}:
        return "relation_traversal"
    return "semantic_fallback"


def build_query_frame(query: str, chunks: List[Dict[str, Any]]) -> QueryFrame:
    intent, conf = detect_intent(query)
    target_entity = extract_target_entity(query, chunks)
    asked_slot = detect_asked_slot(query) if intent == "ask_property_value" else None

    frame = QueryFrame(
        raw_query=query,
        intent=intent,
        target_entity=target_entity,
        asked_slot=asked_slot,
        expected_answer_type=expected_answer_type(intent),
        confidence=conf,
        reasoning_mode=reasoning_mode(intent),
    )
    frame.rewritten_queries = rewrite_queries(frame)
    return frame


def build_retrieval_plan(frame: QueryFrame) -> RetrievalPlan:
    steps = []

    if frame.intent == "locate_in_package":
        steps = [
            RetrievalStep("lookup_entity_node", "先定位目标元素节点"),
            RetrievalStep("lookup_package_from_node", "从节点中抽取 package/path"),
            RetrievalStep("semantic_fallback", "若失败则语义兜底"),
        ]
    elif frame.intent == "locate_in_diagram":
        steps = [
            RetrievalStep("lookup_entity_node", "先定位目标元素节点"),
            RetrievalStep("lookup_diagram_from_node", "从节点中抽取 Appears in diagrams"),
            RetrievalStep("lookup_diagram_chunks_by_entity", "再直接找包含该实体的 diagram chunk"),
            RetrievalStep("semantic_fallback", "若失败则语义兜底"),
        ]
    elif frame.intent == "find_parent_or_owner":
        steps = [
            RetrievalStep("lookup_relation_by_source", "先找以目标实体为 source 的 relation"),
            RetrievalStep("lookup_entity_node", "补充节点证据"),
            RetrievalStep("semantic_fallback", "若失败则语义兜底"),
        ]
    elif frame.intent == "list_members":
        steps = [
            RetrievalStep("lookup_package_chunk", "先定位 package chunk"),
            RetrievalStep("extract_members_from_package", "从 package 中抽 Elements / Child packages"),
            RetrievalStep("expand_members_from_nodes", "若 package 被截断，则按 node.package 全量补全"),
            RetrievalStep("semantic_fallback", "若失败则语义兜底"),
        ]
    elif frame.intent == "ask_property_value":
        steps = [
            RetrievalStep("lookup_entity_node", "先定位目标元素节点"),
            RetrievalStep("extract_properties", "从节点中抽取 Properties / Attributes"),
            RetrievalStep("semantic_fallback", "若失败则语义兜底"),
        ]
    elif frame.intent == "find_relations":
        steps = [
            RetrievalStep("lookup_relations_touching_entity", "找所有与目标实体相关的 relation"),
            RetrievalStep("lookup_entity_node", "补充节点证据"),
            RetrievalStep("semantic_fallback", "若失败则语义兜底"),
        ]
    else:
        steps = [RetrievalStep("semantic_fallback", "直接做语义检索")]

    return RetrievalPlan(intent=frame.intent, steps=steps)


def chunk_text_norm(chunk: Dict[str, Any]) -> str:
    return normalize_text(get_chunk_text(chunk))


def name_norm(chunk: Dict[str, Any]) -> str:
    return normalize_text(get_chunk_name(chunk))


def find_exact_node_by_name(chunks: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    target_norm = normalize_text(target)
    return [ch for ch in chunks if get_chunk_type(ch) == "node" and name_norm(ch) == target_norm]


def find_exact_package_by_name(chunks: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    target_norm = normalize_text(target)
    return [ch for ch in chunks if get_chunk_type(ch) == "package" and name_norm(ch) == target_norm]


def find_relations_by_source(chunks: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    target_norm = normalize_text(target)
    out = []
    for ch in chunks:
        if get_chunk_type(ch) != "relation":
            continue
        txt = chunk_text_norm(ch)
        if f"source element: {target_norm}" in txt:
            out.append(ch)
    return out


def find_relations_touching_entity(chunks: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    target_norm = normalize_text(target)
    out = []
    for ch in chunks:
        if get_chunk_type(ch) != "relation":
            continue
        txt = chunk_text_norm(ch)
        if target_norm in txt:
            out.append(ch)
    return out


def find_diagram_chunks_containing_entity(chunks: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    target_norm = normalize_text(target)
    return [ch for ch in chunks if get_chunk_type(ch) == "diagram" and target_norm in chunk_text_norm(ch)]


def extract_package_path_from_node_text(text: str) -> Optional[str]:
    m = re.search(r"Package:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return None


def extract_diagrams_from_node_text(text: str) -> List[str]:
    lines = text.splitlines()
    out = []
    in_section = False
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("Appears in diagrams:"):
            in_section = True
            continue
        if in_section:
            if not line_strip.startswith("-"):
                break
            out.append(line_strip[1:].strip())
    return out


def extract_properties_from_node_text(text: str) -> List[str]:
    lines = text.splitlines()
    out = []

    def read_section(header: str):
        items = []
        in_section = False
        for line in lines:
            s = line.strip()
            if s.startswith(header):
                in_section = True
                continue
            if in_section:
                if not s.startswith("-"):
                    break
                items.append(s[1:].strip())
        return items

    out.extend(read_section("Properties:"))
    out.extend(read_section("Attributes:"))
    return out


def extract_package_members(text: str) -> List[str]:
    lines = text.splitlines()
    out = []

    def read_section(start_key: str) -> List[str]:
        items = []
        in_section = False
        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith(start_key):
                in_section = True
                continue
            if in_section:
                if not line_strip.startswith("-"):
                    break
                items.append(line_strip[1:].strip())
        return items

    out.extend(read_section("Elements:"))
    out.extend(read_section("Child packages:"))
    return out


def property_matches_slot(prop: str, asked_slot: str) -> int:
    prop_norm = normalize_text(prop)
    slot_norm = normalize_text(asked_slot)

    if ":" in prop:
        left = normalize_text(prop.split(":", 1)[0])
    elif "=" in prop:
        left = normalize_text(prop.split("=", 1)[0])
    else:
        left = prop_norm

    if left == slot_norm:
        return 100

    for kw in PROPERTY_KEYWORDS.get(asked_slot, []):
        kw_norm = normalize_text(kw)
        if left == kw_norm:
            return 95
        if kw_norm in left:
            return 85

    if asked_slot == "strom":
        if " a" in f" {prop_norm}" or "ampere" in prop_norm or "i_max" in prop_norm or "i =" in prop_norm:
            return 70

    if asked_slot == "spannung":
        if " v" in f" {prop_norm}" or "volt" in prop_norm or "u_max" in prop_norm or "u =" in prop_norm:
            return 70

    if asked_slot == "masse":
        if " kg" in f" {prop_norm}" or "gewicht" in prop_norm:
            return 70

    if asked_slot == "beschleunigung":
        if "m/s2" in prop_norm or "m/s²" in prop or "beschleunigung" in prop_norm:
            return 70

    if asked_slot == "beschleunigungszeit":
        if " s" in f" {prop_norm}" or "sek" in prop_norm or "zeit" in prop_norm:
            return 70

    return 0


def choose_property(properties: List[str], asked_slot: Optional[str]) -> Tuple[Optional[str], bool]:
    if not properties:
        return None, False

    if not asked_slot:
        return properties[0], True

    scored = []
    for p in properties:
        score = property_matches_slot(p, asked_slot)
        scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_prop = scored[0]

    if best_score <= 0:
        return None, False

    return best_prop, True


def lexical_bonus(frame: QueryFrame, chunk: Dict[str, Any]) -> float:
    bonus = 0.0
    ctype = get_chunk_type(chunk)
    text_norm = chunk_text_norm(chunk)
    cname_norm = name_norm(chunk)

    target_norm = normalize_text(frame.target_entity) if frame.target_entity else ""
    q_tokens = extract_query_tokens(frame.raw_query)
    target_tokens = [t for t in target_norm.split() if len(t) >= 2]

    overlap = sum(1 for t in q_tokens if t in text_norm or t in cname_norm)
    bonus += min(overlap * 0.02, 0.08)

    if target_norm:
        if cname_norm == target_norm:
            bonus += 0.18
        elif target_norm in text_norm:
            bonus += 0.12

    entity_hits = sum(1 for t in target_tokens if t in text_norm or t in cname_norm)

    if frame.intent == "list_members" and ctype == "package":
        if cname_norm == target_norm:
            bonus += 0.20
        elif target_norm in text_norm:
            bonus += 0.10
        bonus += min(entity_hits * 0.03, 0.08)

    elif frame.intent == "locate_in_package" and ctype == "node":
        if cname_norm == target_norm:
            bonus += 0.16

    elif frame.intent == "locate_in_diagram" and ctype == "diagram":
        if target_norm in text_norm:
            bonus += 0.16

    elif frame.intent == "find_parent_or_owner" and ctype == "relation":
        if f"source element: {target_norm}" in text_norm:
            bonus += 0.20
        elif f"target element: {target_norm}" in text_norm:
            bonus += 0.06

    elif frame.intent == "ask_property_value" and ctype == "node":
        if cname_norm == target_norm:
            bonus += 0.18
        if frame.asked_slot and normalize_text(frame.asked_slot) in text_norm:
            bonus += 0.08

    elif frame.intent == "find_relations" and ctype == "relation":
        if target_norm in text_norm:
            bonus += 0.18

    return min(bonus, 0.32)


def semantic_search(
    model: SentenceTransformer,
    frame: QueryFrame,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int = 10,
) -> List[ScoredResult]:
    q = frame.rewritten_queries[0] if frame.rewritten_queries else frame.raw_query

    q_emb = model.encode(
        [f"query: {q}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]

    raw_scores = embeddings @ q_emb
    weights = safe_get_type_weights(frame.intent)

    scored = []
    for i, ch in enumerate(chunks):
        ctype = get_chunk_type(ch)
        type_weight = weights.get(ctype, 1.0)
        lex = lexical_bonus(frame, ch)
        score = float(raw_scores[i]) * type_weight + lex
        scored.append((i, score, float(raw_scores[i]), type_weight, lex))

    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    for rank, (idx, score, raw, tw, lex) in enumerate(scored[:top_k], start=1):
        out.append(
            ScoredResult(
                rank=rank,
                score=score,
                raw_score=raw,
                type_weight=tw,
                lexical_bonus=lex,
                source="semantic_fallback",
                chunk=chunks[idx],
            )
        )
    return out


def make_scored_result(
    rank: int,
    chunk: Dict[str, Any],
    source: str,
    score: float = 1.10,
    raw_score: float = 0.90,
    type_weight: float = 1.00,
    lexical_bonus: float = 0.20,
) -> ScoredResult:
    return ScoredResult(
        rank=rank,
        score=score,
        raw_score=raw_score,
        type_weight=type_weight,
        lexical_bonus=lexical_bonus,
        source=source,
        chunk=chunk,
    )


def dedupe_evidence(items: List[ScoredResult], top_k: int = 10) -> List[ScoredResult]:
    seen = set()
    out = []
    for item in items:
        cid = item.chunk.get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(item)
        if len(out) >= top_k:
            break

    for i, item in enumerate(out, start=1):
        item.rank = i
    return out


def execute_locate_in_package(frame, chunks, semantic_hits) -> AnswerBundle:
    evidence = []
    if frame.target_entity:
        nodes = find_exact_node_by_name(chunks, frame.target_entity)
        for idx, node in enumerate(nodes, start=1):
            evidence.append(make_scored_result(idx, node, "lookup_entity_node", 1.15, 0.90, 1.00, 0.25))

        if nodes:
            path = extract_package_path_from_node_text(get_chunk_text(nodes[0]))
            if path:
                evidence.extend(semantic_hits)
                return AnswerBundle(
                    final_answer=f"{frame.target_entity} 位于 package：{path}",
                    evidence=dedupe_evidence(evidence, 10),
                )

    return AnswerBundle(
        final_answer="未能通过结构化字段直接定位 package，以下是最相关证据。",
        evidence=dedupe_evidence(semantic_hits[:10], 10),
    )


def execute_locate_in_diagram(frame, chunks, semantic_hits) -> AnswerBundle:
    evidence = []
    diagrams = []
    seen_diagrams = set()

    def add_diagram(diagram_str: str):
        key = normalize_text(diagram_str)
        if key not in seen_diagrams:
            seen_diagrams.add(key)
            diagrams.append(diagram_str)

    if frame.target_entity:
        direct_diagrams = find_diagram_chunks_containing_entity(chunks, frame.target_entity)
        for idx, dg in enumerate(direct_diagrams, start=1):
            evidence.append(make_scored_result(idx, dg, "lookup_diagram_chunks_by_entity", 1.27, 0.84, 1.18, 0.28))
            chunk_id = str(dg.get("chunk_id", ""))
            name = str(dg.get("name", ""))
            if "::" in chunk_id:
                middle = chunk_id.split("::")[1]
                middle = middle.replace("EAID_", "")
                add_diagram(f"{name} [{middle}]")
            else:
                add_diagram(name)

        nodes = find_exact_node_by_name(chunks, frame.target_entity)
        if nodes:
            node = nodes[0]
            evidence.append(make_scored_result(len(evidence) + 1, node, "lookup_entity_node", 1.14, 0.90, 1.00, 0.24))
            extracted = extract_diagrams_from_node_text(get_chunk_text(node))
            for d in extracted:
                add_diagram(d)

        if diagrams:
            evidence.extend(semantic_hits)
            return AnswerBundle(
                final_answer=f"{frame.target_entity} 出现在这些 diagram 中：\n" + "\n".join(f"- {d}" for d in diagrams),
                evidence=dedupe_evidence(evidence, 10),
            )

    return AnswerBundle(
        final_answer="未能通过结构化字段直接提取 diagram，以下是最相关证据。",
        evidence=dedupe_evidence(semantic_hits[:10], 10),
    )


def execute_find_parent_or_owner(frame, chunks, semantic_hits) -> AnswerBundle:
    evidence = []

    if frame.target_entity:
        relations = find_relations_by_source(chunks, frame.target_entity)
        for idx, rel in enumerate(relations, start=1):
            evidence.append(make_scored_result(idx, rel, "lookup_relation_by_source", 1.32, 0.88, 1.18, 0.28))

        if relations:
            txt = get_chunk_text(relations[0])
            m = re.search(r"Target element:\s*(.+)", txt)
            if m:
                target_line = m.group(1).strip()
                evidence.extend(semantic_hits)
                return AnswerBundle(
                    final_answer=f"{frame.target_entity} 的直接输出关系目标是：{target_line}",
                    evidence=dedupe_evidence(evidence, 10),
                )

    return AnswerBundle(
        final_answer="未能直接提取该元素的直接关系目标，以下是最相关证据。",
        evidence=dedupe_evidence(semantic_hits[:10], 10),
    )


def collect_nodes_under_package(chunks: List[Dict[str, Any]], package_path: str) -> List[str]:
    out = []
    target_norm = normalize_text(package_path)

    for ch in chunks:
        if get_chunk_type(ch) != "node":
            continue
        text = get_chunk_text(ch)
        m = re.search(r"Package:\s*(.+)", text)
        if not m:
            continue
        node_pkg = m.group(1).strip()
        if normalize_text(node_pkg) == target_norm:
            name = ch.get("name")
            node_id = ch.get("node_id")
            if name and node_id:
                out.append(f"{name} [{str(node_id).replace('EAID_', '')}]")
            elif name:
                out.append(str(name))
    return sorted(set(out), key=lambda x: normalize_text(x))


def execute_list_members(frame, chunks, semantic_hits) -> AnswerBundle:
    evidence = []

    if frame.target_entity:
        packages = find_exact_package_by_name(chunks, frame.target_entity)
        for idx, pkg in enumerate(packages, start=1):
            evidence.append(make_scored_result(idx, pkg, "lookup_package_chunk", 1.40, 0.90, 1.20, 0.32))

        if packages:
            pkg_text = get_chunk_text(packages[0])
            members = extract_package_members(pkg_text)

            path_match = re.search(r"Path:\s*(.+)", pkg_text)
            pkg_path = path_match.group(1).strip() if path_match else None

            if "Additional elements omitted." in pkg_text and pkg_path:
                expanded = collect_nodes_under_package(chunks, pkg_path)
                if expanded:
                    members = expanded

            if members:
                evidence.extend(semantic_hits)
                return AnswerBundle(
                    final_answer=f"{frame.target_entity} package 中包含以下元素：\n" + "\n".join(f"- {m}" for m in members),
                    evidence=dedupe_evidence(evidence, 10),
                )

    return AnswerBundle(
        final_answer="未能从 package chunk 中直接提取成员，以下是最相关证据。",
        evidence=dedupe_evidence(semantic_hits[:10], 10),
    )


def execute_ask_property_value(frame, chunks, semantic_hits) -> AnswerBundle:
    evidence = []

    if frame.target_entity:
        nodes = find_exact_node_by_name(chunks, frame.target_entity)
        for idx, node in enumerate(nodes, start=1):
            evidence.append(make_scored_result(idx, node, "lookup_entity_node", 1.18, 0.90, 1.00, 0.28))

        if nodes:
            props = extract_properties_from_node_text(get_chunk_text(nodes[0]))
            chosen, matched = choose_property(props, frame.asked_slot)

            evidence.extend(semantic_hits)

            if matched and chosen:
                if frame.asked_slot:
                    final = f"{frame.target_entity} 的 {frame.asked_slot}：{chosen}"
                else:
                    final = f"{frame.target_entity} 的属性之一：{chosen}"
                return AnswerBundle(final_answer=final, evidence=dedupe_evidence(evidence, 10))

            if props:
                return AnswerBundle(
                    final_answer=(
                        f"未在 {frame.target_entity} 中找到与“{frame.asked_slot}”对应的属性。\n"
                        f"{frame.target_entity} 当前可提取到的属性有：\n" +
                        "\n".join(f"- {p}" for p in props)
                    ),
                    evidence=dedupe_evidence(evidence, 10),
                )

    return AnswerBundle(
        final_answer="未能通过结构化 properties / attributes 直接提取属性值，以下是最相关证据。",
        evidence=dedupe_evidence(semantic_hits[:10], 10),
    )


def execute_find_relations(frame, chunks, semantic_hits) -> AnswerBundle:
    evidence = []

    if frame.target_entity:
        rels = find_relations_touching_entity(chunks, frame.target_entity)
        for idx, rel in enumerate(rels[:5], start=1):
            evidence.append(make_scored_result(idx, rel, "lookup_relations_touching_entity", 1.24, 0.90, 1.18, 0.18))

        if rels:
            lines = []
            for rel in rels[:10]:
                text = get_chunk_text(rel).splitlines()[0].strip()
                lines.append(f"- {text}")

            evidence.extend(semantic_hits)
            return AnswerBundle(
                final_answer=f"{frame.target_entity} 相关的 relations 包括：\n" + "\n".join(lines),
                evidence=dedupe_evidence(evidence, 10),
            )

    return AnswerBundle(
        final_answer="未能直接提取 relations，以下是最相关证据。",
        evidence=dedupe_evidence(semantic_hits[:10], 10),
    )


def execute_query(frame, chunks, semantic_hits) -> AnswerBundle:
    if frame.intent == "locate_in_package":
        return execute_locate_in_package(frame, chunks, semantic_hits)
    if frame.intent == "locate_in_diagram":
        return execute_locate_in_diagram(frame, chunks, semantic_hits)
    if frame.intent == "find_parent_or_owner":
        return execute_find_parent_or_owner(frame, chunks, semantic_hits)
    if frame.intent == "list_members":
        return execute_list_members(frame, chunks, semantic_hits)
    if frame.intent == "ask_property_value":
        return execute_ask_property_value(frame, chunks, semantic_hits)
    if frame.intent == "find_relations":
        return execute_find_relations(frame, chunks, semantic_hits)

    return AnswerBundle(
        final_answer="未识别出明确 intent，以下为语义检索结果。",
        evidence=dedupe_evidence(semantic_hits[:10], 10),
    )


def maybe_use_llm_parser(frame: QueryFrame, use_llm_parser: bool = False) -> QueryFrame:
    if not use_llm_parser:
        return frame
    if frame.confidence >= 0.80:
        return frame
    return frame


def print_plan(frame: QueryFrame, plan: RetrievalPlan) -> None:
    print(f"Detected intent    : {frame.intent}")
    print(f"Target entity      : {frame.target_entity}")
    print(f"Asked slot         : {frame.asked_slot}")
    print(f"Expected answer    : {frame.expected_answer_type}")
    print(f"Reasoning mode     : {frame.reasoning_mode}")
    print(f"Query tokens       : {extract_query_tokens(frame.raw_query)}")
    print(f"Rewritten queries  : {frame.rewritten_queries}")
    print("Retrieval plan:")
    for i, step in enumerate(plan.steps, start=1):
        print(f"  {i}. {step.name} - {step.description}")


def print_answer_bundle(bundle: AnswerBundle) -> None:
    print("=" * 100)
    print("Final Answer:")
    print(bundle.final_answer)
    print("=" * 100)

    for item in bundle.evidence:
        print("=" * 100)
        print(f"rank         : {item.rank}")
        print(f"score        : {item.score:.4f}")
        print(f"raw_score    : {item.raw_score:.4f}")
        print(f"type_weight  : {item.type_weight:.2f}")
        print(f"lexical_bonus: {item.lexical_bonus:.4f}")
        print(f"source       : {item.source}")
        print(f"chunk_id     : {item.chunk.get('chunk_id')}")
        print(f"chunk_type   : {item.chunk.get('chunk_type')}")
        print(f"name         : {item.chunk.get('name')}")
        print(f"type         : {item.chunk.get('type')}")
        print(f"source_id    : {item.chunk.get('source_id')}")
        print("-" * 100)
        print(item.chunk.get("text", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structured MBSE query retriever")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--top_k", type=int, default=10, help="Top K fallback results")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model")
    parser.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS_PATH, help="Path to chunks.jsonl")
    parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS_PATH, help="Path to embeddings.npy")
    parser.add_argument("--show_plan", action="store_true", help="Show query frame and retrieval plan")
    parser.add_argument("--answer_only", action="store_true", help="Only print final answer")
    parser.add_argument("--use_llm_parser", action="store_true", help="Enable optional LLM parser hook")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = load_model_quietly(args.model)

    chunks = load_chunks(args.chunks)
    print(f"Loaded chunks: {len(chunks)}")

    print(f"Loading existing embeddings from: {args.embeddings}")
    embeddings = load_embeddings(args.embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")

    frame = build_query_frame(args.query, chunks)
    frame = maybe_use_llm_parser(frame, use_llm_parser=args.use_llm_parser)
    plan = build_retrieval_plan(frame)

    semantic_hits = semantic_search(
        model=model,
        frame=frame,
        chunks=chunks,
        embeddings=embeddings,
        top_k=args.top_k,
    )

    bundle = execute_query(frame, chunks, semantic_hits)

    if args.answer_only:
        print(bundle.final_answer)
        return

    if args.show_plan:
        print_plan(frame, plan)

    print_answer_bundle(bundle)


if __name__ == "__main__":
    main()
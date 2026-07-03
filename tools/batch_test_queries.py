#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Tuple

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")

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

from sentence_transformers import SentenceTransformer
from tools.real_embedding import (
    DEFAULT_MODEL_NAME,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EMBEDDINGS_PATH,
    load_chunks,
    load_embeddings,
    build_query_frame,
    build_retrieval_plan,
    maybe_use_llm_parser,
    semantic_search,
    execute_query,
    extract_query_tokens,
)

DEFAULT_QUERIES = [
    "In welchem Package liegt der Akku",
    "In welchem Diagramm erscheint der E-Motor",
    "Zu welchem System gehört die Energieversorgung",
    "Welche Elemente enthält das bdd_Blocks Package",
    "Was ist der Strom des Akku",
    "Was ist die Spannung des Akku",
    "Wie hoch ist die Beschleunigung",
    "Wie hoch ist die Beschleunigungszeit",
    "Wie hoch ist die Gesamtmasse_Fahrrad",
]


def load_model_quietly(model_name: str):
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

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(_SilentStderr()):
        model = SentenceTransformer(model_name)
    return model


def read_queries_from_txt(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Queries file not found: {p}")

    queries = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q and not q.startswith("#"):
                queries.append(q)
    return queries


def read_queries_with_expected(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Expected queries file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON must be a list.")
    return data


def evaluate_answer(answer: str, must_contain: List[str]) -> Tuple[str, List[str]]:
    missing = [x for x in must_contain if x not in answer]
    if not must_contain:
        return "NO_EXPECTATION", []
    if not missing:
        return "PASS", []
    return "FAIL", missing


def format_plan_block(frame, plan) -> str:
    lines = []
    lines.append(f"Detected intent    : {frame.intent}")
    lines.append(f"Target entity      : {frame.target_entity}")
    lines.append(f"Asked slot         : {frame.asked_slot}")
    lines.append(f"Expected answer    : {frame.expected_answer_type}")
    lines.append(f"Reasoning mode     : {frame.reasoning_mode}")
    lines.append(f"Query tokens       : {extract_query_tokens(frame.raw_query)}")
    lines.append(f"Rewritten queries  : {frame.rewritten_queries}")
    lines.append("Retrieval plan:")
    for i, step in enumerate(plan.steps, start=1):
        lines.append(f"  {i}. {step.name} - {step.description}")
    return "\n".join(lines)


def format_bundle_block(bundle, answer_only: bool = False, top_k: int = 10) -> str:
    lines = []
    lines.append("=" * 100)
    lines.append("Final Answer:")
    lines.append(bundle.final_answer)
    lines.append("=" * 100)

    if answer_only:
        return "\n".join(lines)

    for item in bundle.evidence[:top_k]:
        lines.append("=" * 100)
        lines.append(f"rank         : {item.rank}")
        lines.append(f"score        : {item.score:.4f}")
        lines.append(f"raw_score    : {item.raw_score:.4f}")
        lines.append(f"type_weight  : {item.type_weight:.2f}")
        lines.append(f"lexical_bonus: {item.lexical_bonus:.4f}")
        lines.append(f"source       : {item.source}")
        lines.append(f"chunk_id     : {item.chunk.get('chunk_id')}")
        lines.append(f"chunk_type   : {item.chunk.get('chunk_type')}")
        lines.append(f"name         : {item.chunk.get('name')}")
        lines.append(f"type         : {item.chunk.get('type')}")
        lines.append(f"source_id    : {item.chunk.get('source_id')}")
        lines.append("-" * 100)
        lines.append(str(item.chunk.get("text", "")))

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test queries for structured MBSE retriever")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embeddings", type=str, default=DEFAULT_EMBEDDINGS_PATH)

    parser.add_argument("--queries_file", type=str, default="", help="txt: one query per line")
    parser.add_argument("--expected_file", type=str, default="", help="json: query + must_contain")

    parser.add_argument("--output", type=str, default="outputs/full_batch_report.txt")
    parser.add_argument("--summary_output", type=str, default="outputs/batch_summary.txt")
    parser.add_argument("--json_output", type=str, default="outputs/batch_results.json")

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--answer_only", action="store_true")
    parser.add_argument("--use_llm_parser", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.expected_file:
        query_specs = read_queries_with_expected(args.expected_file)
    elif args.queries_file:
        query_specs = [{"query": q, "must_contain": []} for q in read_queries_from_txt(args.queries_file)]
    else:
        query_specs = [{"query": q, "must_contain": []} for q in DEFAULT_QUERIES]

    print(f"[INFO] Loading model once: {args.model}")
    model = load_model_quietly(args.model)

    print(f"[INFO] Loading chunks: {args.chunks}")
    chunks = load_chunks(args.chunks)
    print(f"[INFO] Loaded chunks: {len(chunks)}")

    print(f"[INFO] Loading embeddings: {args.embeddings}")
    embeddings = load_embeddings(args.embeddings)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings")

    full_lines = []
    summary_lines = []
    json_results = []

    pass_count = 0
    fail_count = 0
    noexp_count = 0

    full_lines.append(f"MODEL: {args.model}")
    full_lines.append(f"CHUNKS: {args.chunks}")
    full_lines.append(f"EMBEDDINGS: {args.embeddings}")
    full_lines.append(f"TOTAL_QUERIES: {len(query_specs)}")
    full_lines.append("")

    for idx, spec in enumerate(query_specs, start=1):
        q = spec["query"]
        must_contain = spec.get("must_contain", [])

        print(f"[RUN] ({idx}/{len(query_specs)}) {q}")

        frame = build_query_frame(q, chunks)
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

        status, missing = evaluate_answer(bundle.final_answer, must_contain)

        if status == "PASS":
            pass_count += 1
        elif status == "FAIL":
            fail_count += 1
        else:
            noexp_count += 1

        full_lines.append("=" * 120)
        full_lines.append(f"QUERY #{idx}")
        full_lines.append("=" * 120)
        full_lines.append(q)
        full_lines.append(f"STATUS: {status}")
        if missing:
            full_lines.append(f"MISSING: {missing}")
        full_lines.append("")

        if not args.answer_only:
            full_lines.append(format_plan_block(frame, plan))
            full_lines.append("")

        full_lines.append(format_bundle_block(bundle, answer_only=args.answer_only, top_k=args.top_k))
        full_lines.append("")

        summary_lines.append("=" * 80)
        summary_lines.append(f"QUERY #{idx}")
        summary_lines.append(q)
        summary_lines.append(f"STATUS: {status}")
        summary_lines.append("ANSWER:")
        summary_lines.append(bundle.final_answer)
        if missing:
            summary_lines.append(f"MISSING: {missing}")
        summary_lines.append("")

        json_results.append({
            "index": idx,
            "query": q,
            "status": status,
            "must_contain": must_contain,
            "missing": missing,
            "intent": frame.intent,
            "target_entity": frame.target_entity,
            "asked_slot": frame.asked_slot,
            "expected_answer_type": frame.expected_answer_type,
            "reasoning_mode": frame.reasoning_mode,
            "query_tokens": extract_query_tokens(frame.raw_query),
            "rewritten_queries": frame.rewritten_queries,
            "final_answer": bundle.final_answer,
            "top_evidence": [
                {
                    "rank": item.rank,
                    "score": item.score,
                    "raw_score": item.raw_score,
                    "type_weight": item.type_weight,
                    "lexical_bonus": item.lexical_bonus,
                    "source": item.source,
                    "chunk_id": item.chunk.get("chunk_id"),
                    "chunk_type": item.chunk.get("chunk_type"),
                    "name": item.chunk.get("name"),
                    "type": item.chunk.get("type"),
                    "source_id": item.chunk.get("source_id"),
                }
                for item in bundle.evidence[:args.top_k]
            ]
        })

    total = len(query_specs)
    eval_total = pass_count + fail_count
    accuracy = (pass_count / eval_total * 100.0) if eval_total > 0 else 0.0

    footer = [
        "",
        "=" * 120,
        "SUMMARY",
        "=" * 120,
        f"TOTAL           : {total}",
        f"PASS            : {pass_count}",
        f"FAIL            : {fail_count}",
        f"NO_EXPECTATION  : {noexp_count}",
        f"ACCURACY        : {accuracy:.2f}%",
        "",
    ]

    full_lines.extend(footer)
    summary_lines.extend(footer)

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    json_path = Path(args.json_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text("\n".join(full_lines), encoding="utf-8")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE]")
    print(f"  Full report : {output_path}")
    print(f"  Summary     : {summary_path}")
    print(f"  JSON        : {json_path}")
    print(f"  PASS        : {pass_count}")
    print(f"  FAIL        : {fail_count}")
    print(f"  ACCURACY    : {accuracy:.2f}%")


if __name__ == "__main__":
    main()
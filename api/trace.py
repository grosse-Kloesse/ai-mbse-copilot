import json
import os
from pathlib import Path
from typing import Dict, List

EDGES_FILE = Path(os.getenv("EDGES_FILE", "/app/data/processed/v2/edges.jsonl"))

def load_edges() -> List[dict]:
    edges: List[dict] = []
    with EDGES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            edges.append(json.loads(line))
    return edges


def build_adjacency(edges: List[dict]) -> Dict[str, List[dict]]:
    """src_id -> list of outgoing edges"""
    adj: Dict[str, List[dict]] = {}
    for e in edges:
        adj.setdefault(e["src_id"], []).append(e)
    return adj


def trace_paths(adj: Dict[str, List[dict]], cur: str, k: int, path: List[str]) -> List[List[str]]:
    """
    Return paths up to exactly k hops (or earlier if no outgoing edges).
    Each path is a list of node ids: [start, ..., end]
    """
    if k == 0:
        return [path]

    out_edges = adj.get(cur, [])
    if not out_edges:
        return [path]

    paths: List[List[str]] = []
    for e in out_edges:
        nxt = e["dst_id"]
        paths.extend(trace_paths(adj, nxt, k - 1, path + [nxt]))
    return paths


def trace_k(start_id: str, k: int) -> List[List[str]]:
    edges = load_edges()
    adj = build_adjacency(edges)
    return trace_paths(adj, start_id, k, [start_id])

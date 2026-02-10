import json
from pathlib import Path

EDGES_FILE = Path("data/processed/edges.jsonl")

def load_edges():
    """Read edges.jsonl -> list[dict]."""
    edges = []
    with EDGES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            edges.append(json.loads(line))
    return edges

def build_adjacency(edges):
    """Build adjacency: src_id -> list of edges (outgoing)."""
    adj = {}
    for e in edges:
        adj.setdefault(e["src_id"], []).append(e)
    return adj

def trace_paths(adj, cur, k, path):
    # path: 当前已走过的节点列表
    if k == 0:
        return [path]

    out_edges = adj.get(cur, [])
    if not out_edges:
        return [path]  # 没路了，就返回当前路径（到此为止）

    paths = []
    for e in out_edges:
        nxt = e["dst_id"]
        paths.extend(trace_paths(adj, nxt, k - 1, path + [e["rel_type"]] + [nxt]))
    return paths


def main():
    edges = load_edges()
    adj = build_adjacency(edges)

    start_id = "REQ-001"
    k = 2

    paths = trace_paths(adj, start_id, k, [start_id])
    print("Paths:", paths)

if __name__ == "__main__":
    main()

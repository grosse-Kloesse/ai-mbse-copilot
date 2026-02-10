import json
from pathlib import Path

EDGES_FILE = Path("data/processed/edges.jsonl")

def load_edges():
    edges = []
    with EDGES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            edges.append(json.loads(line))
    return edges

def main(start_id: str = "REQ-001"):
    edges = load_edges()

    # 找：start_id 作为 src 的“出边”（outgoing edges）
    outgoing = [e for e in edges if e["src_id"] == start_id]

    print(f"Start: {start_id}")
    if not outgoing:
        print("No outgoing edges found.")
        return

    for e in outgoing:
        outgoing2 = [a for a in edges if a["src_id"] == e["dst_id"]]
        if not outgoing2:
            print(f"{e['src_id']} -({e['rel_type']})-> {e['dst_id']} (no 2nd hop)")
            continue
        for a in outgoing2:

            print(f"{e['src_id']} -({e['rel_type']})-> {e['dst_id']} -({a['rel_type']})-> {a['dst_id']}")

if __name__ == "__main__":
    import sys
    start = sys.argv[1] if len(sys.argv) > 1 else "REQ-001"
    main(start)

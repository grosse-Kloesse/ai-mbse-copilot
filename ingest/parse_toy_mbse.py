import json
from pathlib import Path
from xml.etree import ElementTree as ET

RAW = Path("data/raw/sample_mbse.xml")
OUT_DIR = Path("data/processed")

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Missing input file: {RAW}")

    tree = ET.parse(RAW)
    root = tree.getroot()

    nodes = []
    edges = []
    chunks = []
    source = RAW.name

    # 1) Elements -> nodes + chunks
    elements = root.find("Elements")
    if elements is None:
        raise ValueError("XML missing <Elements> section")

    for el in elements.findall("Element"):
        node_id = el.attrib["id"]
        node_type = el.attrib.get("type", "Unknown")
        name = el.attrib.get("name", "")
        path = el.attrib.get("path", "")
        text = (el.findtext("Text") or "").strip()

        nodes.append({
            "id": node_id,
            "type": node_type,
            "name": name,
            "text": text,
            "path": path,
            "source": source,
        })

        if text:
            chunks.append({
                "chunk_id": f"node::{node_id}::0",
                "text": f"[{node_type}] {name}\n{text}",
                "source_type": "xmi_node",
                "source_id": node_id,
                "source": source,
                "meta": {"type": node_type, "path": path},
            })

    # 2) Relations -> edges
    relations = root.find("Relations")
    if relations is None:
        raise ValueError("XML missing <Relations> section")

    for rel in relations.findall("Relation"):
        edges.append({
            "src_id": rel.attrib["src"],
            "dst_id": rel.attrib["dst"],
            "rel_type": rel.attrib.get("type", "unknown"),
            "source": source,
        })

    write_jsonl(OUT_DIR / "nodes.jsonl", nodes)
    write_jsonl(OUT_DIR / "edges.jsonl", edges)
    write_jsonl(OUT_DIR / "chunks.jsonl", chunks)

    print(f"OK: wrote {len(nodes)} nodes, {len(edges)} edges, {len(chunks)} chunks")

if __name__ == "__main__":
    main()

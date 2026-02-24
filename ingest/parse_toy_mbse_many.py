import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/sample_mbse_many.xml")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/processed/many")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if not RAW.exists():
        raise FileNotFoundError(f"Input not found: {RAW}")

    tree = ET.parse(RAW)
    root = tree.getroot()

    nodes = []
    chunks = []
    edges = []

    # Elements -> nodes + chunks
    for el in root.findall(".//Element"):
        _id = el.attrib.get("id")
        _type = el.attrib.get("type")
        name = el.attrib.get("name", "")
        path = el.attrib.get("path", "")
        text = (el.text or "").strip()

        nodes.append(
            {
                "id": _id,
                "type": _type,
                "name": name,
                "path": path,
                "source": RAW.name,
            }
        )

        chunks.append(
            {
                "chunk_id": f"node::{_id}::0",
                "text": f"[{_type}] {name}\n{text}".strip(),
                "source_type": "xmi_node",
                "source_id": _id,
                "source": RAW.name,
                "meta": {"type": _type, "path": path},
            }
        )

    # Relations -> edges
    for rel in root.findall(".//Relation"):
        src = rel.attrib.get("src")
        dst = rel.attrib.get("dst")
        rtype = rel.attrib.get("type", "trace")
        edges.append(
            {
                "src_id": src,
                "dst_id": dst,
                "rel_type": rtype,
                "source": RAW.name,
            }
        )

    write_jsonl(OUT_DIR / "nodes.jsonl", nodes)
    write_jsonl(OUT_DIR / "chunks.jsonl", chunks)
    write_jsonl(OUT_DIR / "edges.jsonl", edges)

    print("Parsed:", RAW)
    print("nodes:", len(nodes), "edges:", len(edges), "chunks:", len(chunks))
    print("Wrote:", OUT_DIR / "nodes.jsonl")
    print("Wrote:", OUT_DIR / "edges.jsonl")
    print("Wrote:", OUT_DIR / "chunks.jsonl")


if __name__ == "__main__":
    main()

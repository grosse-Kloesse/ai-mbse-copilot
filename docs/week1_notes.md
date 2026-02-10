# Week 1 - Data fundation (Nodes/Edges/Chunks)

- G enerated toy XML: data/raw/sample_mbse.xml
- Parse to:
  - data/processed/nodes.jsonl
  - data/processed/edges.jsonl
  - data/processed/chunks.jsonl
- Trace tool (2-hop):
  python3.11 ingest/trace_one_hop.py REQ-001
  Output: REQ-001 -> FUNC-010 -> BLK-100
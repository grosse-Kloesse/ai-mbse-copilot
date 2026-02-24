from pathlib import Path

OUT = Path("data/raw/sample_mbse_many.xml")

N_REQ = 20
N_FUNC = 20
N_BLK = 20


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<MBSEModel name="ToyMBSE-Many">')

    # Requirements
    for i in range(1, N_REQ + 1):
        rid = f"REQ-{i:03d}"
        lines.append(
            f'  <Element id="{rid}" type="Requirement" name="Requirement {i}" path="Model::Safety">'
            f'System shall satisfy requirement {i} for safe operation.</Element>'
        )

    # Functions
    for i in range(1, N_FUNC + 1):
        fid = f"FUNC-{i:03d}"
        lines.append(
            f'  <Element id="{fid}" type="Function" name="Function {i}" path="Model::Functions">'
            f'Function {i} detects condition {i} and triggers protection.</Element>'
        )

    # Blocks
    for i in range(1, N_BLK + 1):
        bid = f"BLK-{i:03d}"
        lines.append(
            f'  <Element id="{bid}" type="Block" name="Block {i}" path="Model::Architecture">'
            f'Block {i} implements protection and control logic.</Element>'
        )

    # Relations: REQ-i -> FUNC-i -> BLK-i
    for i in range(1, min(N_REQ, N_FUNC, N_BLK) + 1):
        rid = f"REQ-{i:03d}"
        fid = f"FUNC-{i:03d}"
        bid = f"BLK-{i:03d}"
        lines.append(f'  <Relation src="{rid}" dst="{fid}" type="refine" />')
        lines.append(f'  <Relation src="{fid}" dst="{bid}" type="satisfy" />')

    lines.append("</MBSEModel>")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", OUT, "lines:", len(lines))


if __name__ == "__main__":
    main()

from xml.etree.ElementTree import Element, SubElement, ElementTree
from pathlib import Path

def gen(out_path="data/raw/sample_mbse.xml"):
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    root = Element("MBSE")
    elements = SubElement(root, "Elements")
    relations = SubElement(root, "Relations")

    # Nodes
    req = SubElement(elements, "Element", id="REQ-001", type="Requirement",
                     name="Over-voltage protection", path="Model::Safety")
    SubElement(req, "Text").text = "System shall prevent over-voltage conditions."

    func = SubElement(elements, "Element", id="FUNC-010", type="Function",
                      name="Detect over-voltage", path="Model::Functions")
    SubElement(func, "Text").text = "Detect voltage above threshold and trigger protection."

    block = SubElement(elements, "Element", id="BLK-100", type="Block",
                       name="Power_Module", path="Model::Architecture")
    SubElement(block, "Text").text = "Implements protection logic and control."

    # Edges
    SubElement(relations, "Relation", src="REQ-001", dst="FUNC-010", type="refine")
    SubElement(relations, "Relation", src="FUNC-010", dst="BLK-100", type="satisfy")
    SubElement(relations, "Relation", src="BLK-100", dst="REQ-001", type="trace")

    ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)

if __name__ == "__main__":
    gen()
    print("Wrote data/raw/sample_mbse.xml")

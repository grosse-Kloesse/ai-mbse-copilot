from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import xml.etree.ElementTree as ET
import json
import html
import re


NS = {
    "xmi": "http://schema.omg.org/spec/XMI/2.1",
    "uml": "http://schema.omg.org/spec/UML/2.1",
    "SysML": "http://www.omg.org/spec/SysML/20161101/SysML",
}


# =========================
# Data model
# =========================

@dataclass
class PropertyRecord:
    id: Optional[str] = None
    name: str = ""
    type_ref: Optional[str] = None
    default_value: Optional[str] = None
    unit_ref: Optional[str] = None
    stereotype: Optional[str] = None
    visibility: Optional[str] = None
    multiplicity: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ElementView:
    diagram_id: str
    element_id: str
    view_kind: Optional[str] = None
    geometry: Optional[str] = None
    style: Optional[str] = None
    seqno: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationView:
    diagram_id: str
    relation_id: str
    view_kind: Optional[str] = None
    geometry: Optional[str] = None
    style: Optional[str] = None
    seqno: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PackageRecord:
    id: str
    name: str
    xmi_type: Optional[str] = None
    visibility: Optional[str] = None

    parent_id: Optional[str] = None
    child_package_ids: list[str] = field(default_factory=list)

    element_ids: list[str] = field(default_factory=list)
    diagram_ids: list[str] = field(default_factory=list)

    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagramRecord:
    id: str
    name: str
    diagram_type: Optional[str] = None
    package_id: Optional[str] = None

    element_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)

    element_views: list[ElementView] = field(default_factory=list)
    relation_views: list[RelationView] = field(default_factory=list)

    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ElementRecord:
    id: str
    name: str

    xmi_type: Optional[str] = None
    semantic_type: Optional[str] = None

    package_id: Optional[str] = None
    package_path: list[str] = field(default_factory=list)
    package_path_names: list[str] = field(default_factory=list)

    diagram_refs: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)

    source_id: Optional[str] = None
    text: Optional[str] = None
    notes: Optional[str] = None

    stereotype: Optional[str] = None
    visibility: Optional[str] = None
    is_stub: bool = False

    properties: list[PropertyRecord] = field(default_factory=list)

    tags: dict[str, Any] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)

    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationRecord:
    id: str
    source_id: Optional[str] = None
    target_id: Optional[str] = None

    xmi_type: Optional[str] = None
    semantic_type: Optional[str] = None
    stereotype: Optional[str] = None
    direction: Optional[str] = None

    package_id: Optional[str] = None
    diagram_refs: list[str] = field(default_factory=list)

    tags: dict[str, Any] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)

    raw: dict[str, Any] = field(default_factory=dict)


# =========================
# Utils
# =========================

def clean_text(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    text = html.unescape(value)
    text = text.replace("<memo>", "")
    text = text.replace("#NOTES#", "")
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def append_unique(seq: list[str], value: Optional[str]) -> None:
    if value and value not in seq:
        seq.append(value)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize_bad_encoding(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    replacements = {
        "ï¿½": "²",
        "�": "²",
    }
    fixed = text
    for bad, good in replacements.items():
        fixed = fixed.replace(bad, good)
    return fixed


def trimmed_list(items: list[str], limit: int = 15) -> list[str]:
    if len(items) <= limit:
        return items
    return items[:limit]


def id_tail(ref_id: Optional[str]) -> Optional[str]:
    if not ref_id:
        return None
    if "_" not in ref_id:
        return ref_id
    return ref_id.split("_", 1)[1]


def short_id(ref_id: Optional[str]) -> Optional[str]:
    if not ref_id:
        return ref_id
    return re.sub(r"^(EAID|EAPK|EAIDREF|EAREF)_", "", ref_id)


def normalize_display_name(name: Optional[str]) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"(?:EAID|EAPK|EAIDREF|EAREF)_[A-Za-z0-9_]+", raw):
        return short_id(raw) or raw
    return raw


def display_ref(name: Optional[str], ref_id: Optional[str]) -> str:
    name = normalize_display_name(name)
    sid = (short_id(ref_id) or "").strip()
    if name and sid:
        if name == sid:
            return sid
        return f"{name} [{sid}]"
    return name or sid or "<unknown>"


# =========================
# Parsing helpers
# =========================

def infer_xmi_type_from_stype(s_type: Optional[str]) -> Optional[str]:
    if not s_type:
        return None
    mapping = {
        "Class": "uml:Class",
        "Requirement": "uml:Class",
        "Actor": "uml:Actor",
        "UseCase": "uml:UseCase",
        "Activity": "uml:Activity",
        "StateMachine": "uml:StateMachine",
        "State": "uml:State",
        "StateNode": "uml:State",
        "Pseudostate": "uml:Pseudostate",
        "Action": "uml:Action",
        "ActionPin": "uml:Pin",
        "ObjectNode": "uml:ObjectNode",
        "Object": "uml:ObjectNode",
        "ActivityPartition": "uml:ActivityPartition",
        "Lifeline": "uml:Lifeline",
        "Interaction": "uml:Interaction",
        "InteractionFragment": "uml:InteractionFragment",
        "InteractionOccurrence": "uml:InteractionOccurrence",
        "CombinedFragment": "uml:CombinedFragment",
        "Decision": "uml:DecisionNode",
        "DataType": "uml:DataType",
        "Part": "uml:Property",
    }
    return mapping.get(s_type, s_type)


def should_stub_from_stype(s_type: Optional[str], name: Optional[str]) -> bool:
    if not s_type:
        return False
    if s_type == "Part":
        return False
    if s_type == "Package":
        return False
    if not name and s_type not in {"Decision", "StateNode", "Pseudostate"}:
        return False
    return True


def ensure_stub_element(
    elements: dict[str, ElementRecord],
    element_id: str,
    name: Optional[str] = None,
    xmi_type: Optional[str] = None,
    semantic_type: Optional[str] = None,
) -> ElementRecord:
    if element_id in elements:
        ele = elements[element_id]
        if name and (not ele.name or ele.name == ele.id):
            ele.name = name
        if xmi_type and not ele.xmi_type:
            ele.xmi_type = xmi_type
        if semantic_type and not ele.semantic_type:
            ele.semantic_type = semantic_type
        return ele

    ele = ElementRecord(
        id=element_id,
        name=name or element_id,
        xmi_type=xmi_type,
        semantic_type=semantic_type,
        is_stub=True,
    )
    elements[element_id] = ele
    return ele


# =========================
# Parse packages
# =========================

def parse_packages(root: ET.Element, packages: dict[str, PackageRecord]) -> None:
    model = root.find("./uml:Model", NS)
    if model is None:
        return

    def walk_package(pkg_elem: ET.Element, parent_id: Optional[str]) -> None:
        pkg_id = pkg_elem.attrib.get(f"{{{NS['xmi']}}}id")
        pkg_name = pkg_elem.attrib.get("name", "") or ""

        if not pkg_id:
            return

        record = packages.get(pkg_id)
        if record is None:
            record = PackageRecord(
                id=pkg_id,
                name=pkg_name,
                xmi_type=pkg_elem.attrib.get(f"{{{NS['xmi']}}}type"),
                visibility=pkg_elem.attrib.get("visibility"),
                parent_id=parent_id,
                raw={"basic_attrs": dict(pkg_elem.attrib)},
            )
            packages[pkg_id] = record
        else:
            record.name = record.name or pkg_name
            record.parent_id = record.parent_id or parent_id

        if parent_id and parent_id in packages:
            append_unique(packages[parent_id].child_package_ids, pkg_id)

        for child in pkg_elem.findall("./packagedElement"):
            child_type = child.attrib.get(f"{{{NS['xmi']}}}type")
            child_id = child.attrib.get(f"{{{NS['xmi']}}}id")

            if child_type == "uml:Package":
                walk_package(child, pkg_id)
            elif child_id:
                append_unique(record.element_ids, child_id)

    for pe in model.findall("./packagedElement"):
        if pe.attrib.get(f"{{{NS['xmi']}}}type") == "uml:Package":
            walk_package(pe, None)


# =========================
# Parse diagrams
# =========================

def parse_diagrams(root: ET.Element, diagrams: dict[str, DiagramRecord], packages: dict[str, PackageRecord]) -> None:
    ext = root.find("./xmi:Extension", NS)
    if ext is None:
        return

    diagrams_parent = ext.find("./diagrams")
    if diagrams_parent is None:
        return

    for d in diagrams_parent.findall("./diagram"):
        d_id = d.attrib.get(f"{{{NS['xmi']}}}id")
        if not d_id:
            continue

        props = d.find("./properties")
        model = d.find("./model")
        elements_node = d.find("./elements")

        name = props.attrib.get("name", "") if props is not None else ""
        dtype = props.attrib.get("type") if props is not None else None
        package_id = model.attrib.get("package") if model is not None else None

        rec = DiagramRecord(
            id=d_id,
            name=name,
            diagram_type=dtype,
            package_id=package_id,
            raw={
                "diagram_attrs": dict(d.attrib),
                "properties_attrs": dict(props.attrib) if props is not None else {},
                "model_attrs": dict(model.attrib) if model is not None else {},
            },
        )

        if package_id and package_id in packages:
            append_unique(packages[package_id].diagram_ids, d_id)

        if elements_node is not None:
            for element in elements_node.findall("./element"):
                subject = element.attrib.get("subject")
                geometry = element.attrib.get("geometry")
                style = element.attrib.get("style")
                seqno = element.attrib.get("seqno")
                view_kind = element.attrib.get("type")

                if not subject:
                    continue

                if subject.startswith("EAID_"):
                    rec.element_views.append(
                        ElementView(
                            diagram_id=d_id,
                            element_id=subject,
                            view_kind=view_kind,
                            geometry=geometry,
                            style=style,
                            seqno=seqno,
                            raw={"view_attrs": dict(element.attrib)},
                        )
                    )
                else:
                    rec.relation_views.append(
                        RelationView(
                            diagram_id=d_id,
                            relation_id=subject,
                            view_kind=view_kind,
                            geometry=geometry,
                            style=style,
                            seqno=seqno,
                            raw={"view_attrs": dict(element.attrib)},
                        )
                    )

        diagrams[d_id] = rec


# =========================
# Parse elements
# =========================

def parse_basic_elements(root: ET.Element, elements: dict[str, ElementRecord], packages: dict[str, PackageRecord]) -> None:
    model = root.find("./uml:Model", NS)
    if model is None:
        return

    def walk_package(pkg_elem: ET.Element, package_id: Optional[str], path_names: list[str]) -> None:
        pkg_name = pkg_elem.attrib.get("name", "")
        next_path = path_names + ([pkg_name] if pkg_name else [])

        for child in pkg_elem.findall("./packagedElement"):
            child_type = child.attrib.get(f"{{{NS['xmi']}}}type")
            child_id = child.attrib.get(f"{{{NS['xmi']}}}id")
            child_name = child.attrib.get("name", "") or ""

            if child_type == "uml:Package":
                walk_package(child, child_id, next_path)
                continue

            if not child_id:
                continue

            rec = elements.get(child_id)
            if rec is None:
                rec = ElementRecord(
                    id=child_id,
                    name=child_name or child_id,
                    xmi_type=child_type,
                    package_id=package_id,
                    package_path_names=next_path[:],
                    raw={"basic_attrs": dict(child.attrib)},
                )
                elements[child_id] = rec
            else:
                rec.name = rec.name or child_name or child_id
                rec.xmi_type = rec.xmi_type or child_type
                rec.package_id = rec.package_id or package_id

            for attr in child.findall("./ownedAttribute"):
                p = PropertyRecord(
                    id=attr.attrib.get(f"{{{NS['xmi']}}}id"),
                    name=attr.attrib.get("name", "") or "",
                    type_ref=attr.attrib.get("type"),
                    visibility=attr.attrib.get("visibility"),
                    multiplicity=attr.attrib.get("aggregation"),
                    raw={"basic_attrs": dict(attr.attrib)},
                )
                default_value = attr.find("./defaultValue")
                if default_value is not None:
                    p.default_value = default_value.attrib.get("value")
                rec.properties.append(p)

    for pe in model.findall("./packagedElement"):
        if pe.attrib.get(f"{{{NS['xmi']}}}type") == "uml:Package":
            walk_package(pe, pe.attrib.get(f"{{{NS['xmi']}}}id"), [])


def parse_sysml_semantics(root: ET.Element, elements: dict[str, ElementRecord], relations: dict[str, RelationRecord]) -> None:
    ext = root.find("./xmi:Extension", NS)
    if ext is None:
        return

    elems_parent = ext.find("./elements")
    if elems_parent is not None:
        for e in elems_parent.findall("./element"):
            elem_id = e.attrib.get(f"{{{NS['xmi']}}}idref")
            if not elem_id:
                continue

            model = e.find("./model")
            props = e.find("./properties")
            tags_parent = e.find("./tags")

            model_type = model.attrib.get("type") if model is not None else None
            model_name = model.attrib.get("name") if model is not None else None

            if should_stub_from_stype(model_type, model_name):
                rec = ensure_stub_element(
                    elements,
                    elem_id,
                    name=model_name,
                    xmi_type=infer_xmi_type_from_stype(model_type),
                    semantic_type=model_type,
                )
            else:
                rec = elements.get(elem_id)
                if rec is None:
                    rec = ensure_stub_element(elements, elem_id, name=model_name)
                if model_name and (not rec.name or rec.name == rec.id):
                    rec.name = model_name
                if model_type:
                    rec.semantic_type = model_type
                    rec.xmi_type = rec.xmi_type or infer_xmi_type_from_stype(model_type)

            rec.raw.setdefault("extension_element_attrs", dict(e.attrib))
            if model is not None:
                rec.raw["model_attrs"] = dict(model.attrib)
            if props is not None:
                rec.raw["properties_attrs"] = dict(props.attrib)

            if props is not None:
                rec.stereotype = rec.stereotype or props.attrib.get("stereotype")
                rec.visibility = rec.visibility or props.attrib.get("scope")

            if tags_parent is not None:
                for tag in tags_parent.findall("./tag"):
                    tag_name = tag.attrib.get("name")
                    tag_value = sanitize_bad_encoding(clean_text(tag.attrib.get("value")))
                    if tag_name:
                        rec.tags[tag_name] = tag_value

            if not rec.source_id and "id" in rec.tags:
                rec.source_id = rec.tags.get("id")
            if not rec.text and "text" in rec.tags:
                rec.text = rec.tags.get("text")

    connectors_parent = ext.find("./connectors")
    if connectors_parent is not None:
        for conn in connectors_parent.findall("./connector"):
            conn_id = conn.attrib.get(f"{{{NS['xmi']}}}idref")
            if not conn_id:
                continue

            rel = relations.get(conn_id)
            if rel is None:
                rel = RelationRecord(id=conn_id)
                relations[conn_id] = rel

            source = conn.find("./source")
            target = conn.find("./target")
            props = conn.find("./properties")

            if source is not None:
                rel.source_id = source.attrib.get(f"{{{NS['xmi']}}}idref")
            if target is not None:
                rel.target_id = target.attrib.get(f"{{{NS['xmi']}}}idref")
            if props is not None:
                rel.semantic_type = props.attrib.get("ea_type") or rel.semantic_type
                rel.stereotype = props.attrib.get("stereotype") or rel.stereotype
                rel.direction = props.attrib.get("direction") or rel.direction

            rel.raw.setdefault("connector_attrs", dict(conn.attrib))


def parse_extension_elements(root: ET.Element, elements: dict[str, ElementRecord]) -> None:
    ext = root.find("./xmi:Extension", NS)
    if ext is None:
        return

    elems_parent = ext.find("./elements")
    if elems_parent is None:
        return

    for e in elems_parent.findall("./element"):
        elem_id = e.attrib.get(f"{{{NS['xmi']}}}idref")
        if not elem_id:
            continue
        record = elements.get(elem_id)
        if record is None:
            record = ensure_stub_element(elements, elem_id)

        model = e.find("./model")
        props = e.find("./properties")
        tags_parent = e.find("./tags")
        project = e.find("./project")

        if model is not None:
            model_name = model.attrib.get("name")
            model_type = model.attrib.get("type")
            if model_name and (not record.name or record.name == record.id):
                record.name = model_name
            if model_type:
                record.semantic_type = record.semantic_type or model_type
                record.xmi_type = record.xmi_type or infer_xmi_type_from_stype(model_type)

        if props is not None:
            for k in ["stereotype", "style", "nType", "isAbstract"]:
                v = props.attrib.get(k)
                if v:
                    record.custom[k] = v

        if project is not None:
            for k in ["status", "priority", "difficulty", "author", "created", "modified"]:
                v = project.attrib.get(k)
                if v:
                    record.custom[k] = v

        if tags_parent is not None:
            for tag in tags_parent.findall("./tag"):
                tag_name = tag.attrib.get("name")
                tag_value = sanitize_bad_encoding(clean_text(tag.attrib.get("value")))
                if tag_name:
                    record.tags[tag_name] = tag_value

        if not record.source_id and "id" in record.tags:
            record.source_id = record.tags.get("id")
        if not record.text and "text" in record.tags:
            record.text = record.tags.get("text")


def parse_relations(root: ET.Element, relations: dict[str, RelationRecord], packages: dict[str, PackageRecord]) -> None:
    model = root.find("./uml:Model", NS)
    if model is None:
        return

    def walk_package(pkg_elem: ET.Element, package_id: Optional[str]) -> None:
        for child in pkg_elem.findall("./packagedElement"):
            child_type = child.attrib.get(f"{{{NS['xmi']}}}type")
            child_id = child.attrib.get(f"{{{NS['xmi']}}}id")

            if child_type == "uml:Package":
                walk_package(child, child_id)
                continue

            if child_type in {"uml:Association", "uml:Dependency"} and child_id:
                rel = relations.get(child_id)
                if rel is None:
                    rel = RelationRecord(id=child_id)
                    relations[child_id] = rel
                rel.xmi_type = child_type
                rel.package_id = rel.package_id or package_id
                rel.raw.setdefault("basic_attrs", dict(child.attrib))

    for pe in model.findall("./packagedElement"):
        if pe.attrib.get(f"{{{NS['xmi']}}}type") == "uml:Package":
            pkg_id = pe.attrib.get(f"{{{NS['xmi']}}}id")
            walk_package(pe, pkg_id)

    ext = root.find("./xmi:Extension", NS)
    if ext is None:
        return

    connectors_parent = ext.find("./connectors")
    if connectors_parent is None:
        return

    for conn in connectors_parent.findall("./connector"):
        rel_id = conn.attrib.get(f"{{{NS['xmi']}}}idref")
        if not rel_id:
            continue

        rel = relations.get(rel_id)
        if rel is None:
            rel = RelationRecord(id=rel_id)
            relations[rel_id] = rel

        source = conn.find("./source")
        target = conn.find("./target")
        props = conn.find("./properties")

        if source is not None:
            rel.source_id = source.attrib.get(f"{{{NS['xmi']}}}idref")
        if target is not None:
            rel.target_id = target.attrib.get(f"{{{NS['xmi']}}}idref")
        if props is not None:
            rel.xmi_type = rel.xmi_type or props.attrib.get("ea_type")
            rel.stereotype = rel.stereotype or props.attrib.get("stereotype")
            rel.direction = props.attrib.get("direction")

        rel.raw.setdefault("connector_attrs", dict(conn.attrib))


def resolve_property_units(elements: dict[str, ElementRecord]) -> None:
    for e in elements.values():
        for p in e.properties:
            if not p.type_ref:
                continue
            type_ele = elements.get(p.type_ref)
            if type_ele is None:
                continue

            unit_ref = type_ele.custom.get("unit_ref")
            if unit_ref and unit_ref in elements and elements[unit_ref].name:
                p.unit_ref = elements[unit_ref].name
            elif type_ele.name and type_ele.name != type_ele.id:
                p.unit_ref = type_ele.name


def build_property_id_index(elements: dict[str, ElementRecord]) -> dict[str, tuple[str, str]]:
    idx: dict[str, tuple[str, str]] = {}
    for e in elements.values():
        for p in e.properties:
            if p.id:
                idx[p.id] = (e.id, p.name)
    return idx


def build_relation_endpoint_set(relations: dict[str, RelationRecord]) -> set[str]:
    ids: set[str] = set()
    for r in relations.values():
        if r.source_id:
            ids.add(r.source_id)
        if r.target_id:
            ids.add(r.target_id)
    return ids


def build_package_alias_map(packages: dict[str, PackageRecord]) -> dict[str, str]:
    alias: dict[str, str] = {}
    for pkg_id in packages:
        tail = id_tail(pkg_id)
        if not tail:
            continue
        alias[pkg_id] = pkg_id
        alias[f"EAPK_{tail}"] = pkg_id
        alias[f"EAID_{tail}"] = pkg_id
    return alias


def is_package_frame_subject(subject_id: str, diagram: DiagramRecord, package_alias_map: dict[str, str]) -> bool:
    if subject_id in package_alias_map:
        return True

    if diagram.package_id:
        pkg_tail = id_tail(diagram.package_id)
        subj_tail = id_tail(subject_id)
        if pkg_tail and subj_tail and pkg_tail == subj_tail:
            return True

    model_package2 = diagram.raw.get("model_attrs", {}).get("package2")
    if model_package2:
        pkg2_tail = id_tail(model_package2)
        subj_tail = id_tail(subject_id)
        if pkg2_tail and subj_tail and pkg2_tail == subj_tail:
            return True

    return False


def link_refs(
    packages: dict[str, PackageRecord],
    diagrams: dict[str, DiagramRecord],
    elements: dict[str, ElementRecord],
    relations: dict[str, RelationRecord],
) -> None:
    def build_package_path(package_id: Optional[str]) -> list[str]:
        path = []
        current = package_id
        seen = set()
        while current and current in packages and current not in seen:
            seen.add(current)
            path.append(current)
            current = packages[current].parent_id
        path.reverse()
        return path

    for element in elements.values():
        element.package_path = build_package_path(element.package_id)
        element.package_path_names = [packages[pid].name if pid in packages else pid for pid in element.package_path]

    for rel in relations.values():
        if rel.source_id and rel.source_id not in elements:
            ensure_stub_element(elements, rel.source_id)
        if rel.target_id and rel.target_id not in elements:
            ensure_stub_element(elements, rel.target_id)

    property_idx = build_property_id_index(elements)
    relation_endpoint_ids = build_relation_endpoint_set(relations)
    package_alias_map = build_package_alias_map(packages)

    for diagram in diagrams.values():
        kept_element_views: list[ElementView] = []
        kept_relation_views: list[RelationView] = []

        package_subject_ids: list[str] = []
        property_subject_ids: list[str] = []
        ignored_container_subjects: list[str] = []
        truly_unresolved_subjects: list[str] = []

        for ev in diagram.element_views:
            sid = ev.element_id

            if sid in elements:
                append_unique(elements[sid].diagram_refs, diagram.id)
                kept_element_views.append(ev)
                continue

            if sid in relations:
                kept_relation_views.append(
                    RelationView(
                        diagram_id=diagram.id,
                        relation_id=sid,
                        view_kind=ev.view_kind,
                        geometry=ev.geometry,
                        style=ev.style,
                        seqno=ev.seqno,
                        raw=ev.raw,
                    )
                )
                continue

            if sid in property_idx:
                property_subject_ids.append(sid)
                continue

            if is_package_frame_subject(sid, diagram, package_alias_map):
                mapped = package_alias_map.get(sid)
                if mapped:
                    package_subject_ids.append(mapped)
                    if not diagram.package_id:
                        diagram.package_id = mapped
                    if mapped in packages:
                        append_unique(packages[mapped].diagram_ids, diagram.id)
                else:
                    package_subject_ids.append(sid)
                continue

            if sid not in relation_endpoint_ids:
                ignored_container_subjects.append(sid)
                continue

            truly_unresolved_subjects.append(sid)

        diagram.element_views = kept_element_views
        diagram.relation_views = kept_relation_views
        diagram.element_ids = [ev.element_id for ev in diagram.element_views]
        diagram.relation_ids = [rv.relation_id for rv in diagram.relation_views]

        if package_subject_ids:
            diagram.raw["package_subject_ids"] = list(dict.fromkeys(package_subject_ids))
        if property_subject_ids:
            diagram.raw["property_subject_ids"] = list(dict.fromkeys(property_subject_ids))
        if ignored_container_subjects:
            diagram.raw["ignored_container_subjects"] = list(dict.fromkeys(ignored_container_subjects))
        if truly_unresolved_subjects:
            diagram.raw["unresolved_subjects"] = list(dict.fromkeys(truly_unresolved_subjects))

        for rv in diagram.relation_views:
            if rv.relation_id in relations:
                append_unique(relations[rv.relation_id].diagram_refs, diagram.id)

    for rel in relations.values():
        if rel.source_id in elements:
            append_unique(elements[rel.source_id].relation_ids, rel.id)
        if rel.target_id in elements:
            append_unique(elements[rel.target_id].relation_ids, rel.id)

    resolve_property_units(elements)


# =========================
# Export helpers
# =========================

def package_path_names_by_id(package_id: Optional[str], packages: dict[str, PackageRecord]) -> list[str]:
    if not package_id:
        return []
    path = []
    current = package_id
    seen = set()
    while current and current in packages and current not in seen:
        seen.add(current)
        path.append(packages[current].name)
        current = packages[current].parent_id
    path.reverse()
    return path


def relation_display_type(rel: RelationRecord) -> str:
    raw = rel.semantic_type or rel.stereotype or rel.xmi_type or "Relation"
    mapping = {
        "uml:Actor": "Actor",
        "uml:UseCase": "UseCase",
        "uml:Association": "Association",
        "uml:Dependency": "Dependency",
    }
    return mapping.get(raw, raw)


def human_element_type(ele: ElementRecord) -> str:
    raw = ele.semantic_type or ele.xmi_type or "Element"
    mapping = {
        "uml:Actor": "Actor",
        "uml:UseCase": "UseCase",
        "uml:Class": "Class",
        "uml:DataType": "DataType",
        "uml:Activity": "Activity",
        "uml:StateMachine": "StateMachine",
        "uml:State": "State",
        "uml:Action": "Action",
        "uml:Interaction": "Interaction",
        "uml:ObjectNode": "ObjectNode",
        "uml:DecisionNode": "DecisionNode",
    }
    return mapping.get(raw, raw)


def element_display_type(ele: ElementRecord) -> str:
    return human_element_type(ele)


def build_node_rows(elements: dict[str, ElementRecord]) -> list[dict]:
    rows = []
    for e in elements.values():
        rows.append({
            "id": e.id,
            "name": e.name,
            "xmi_type": e.xmi_type,
            "semantic_type": e.semantic_type,
            "package_id": e.package_id,
            "package_path": e.package_path,
            "package_path_names": e.package_path_names,
            "diagram_refs": e.diagram_refs,
            "relation_ids": e.relation_ids,
            "source_id": e.source_id,
            "text": e.text,
            "notes": e.notes,
            "stereotype": e.stereotype,
            "visibility": e.visibility,
            "is_stub": e.is_stub,
            "properties": [p.__dict__ for p in e.properties],
            "tags": e.tags,
            "custom": e.custom,
        })
    return rows


def build_edge_rows(relations: dict[str, RelationRecord]) -> list[dict]:
    rows = []
    for r in relations.values():
        rows.append({
            "id": r.id,
            "source_id": r.source_id,
            "target_id": r.target_id,
            "xmi_type": r.xmi_type,
            "semantic_type": r.semantic_type,
            "stereotype": r.stereotype,
            "direction": r.direction,
            "package_id": r.package_id,
            "diagram_refs": r.diagram_refs,
            "tags": r.tags,
            "custom": r.custom,
        })
    return rows


def build_node_chunk_rows(
    elements: dict[str, ElementRecord],
    relations: dict[str, RelationRecord],
    diagrams: dict[str, DiagramRecord],
) -> list[dict]:
    rows = []
    tag_skip_keys = {"id", "text", "isEncapsulated"}
    allowed_types = {"Requirement", "Block", "Actor", "UseCase"}

    for e in elements.values():
        e_type = element_display_type(e)
        if e_type not in allowed_types:
            continue

        parts = [f"[{e_type}] {display_ref(e.name, e.id)}"]

        if e.source_id:
            parts.append(f"ID: {e.source_id}")
        if e.text:
            parts.append(f"Text: {e.text}")
        if e.notes:
            parts.append(f"Notes: {e.notes}")
        if e.package_path_names:
            parts.append(f"Package: {' / '.join(e.package_path_names)}")

        if e.diagram_refs:
            parts.append("Appears in diagrams:")
            for d_id in trimmed_list(e.diagram_refs, 10):
                d_name = diagrams[d_id].name if d_id in diagrams else None
                parts.append(f"- {display_ref(d_name, d_id)}")

        prop_lines = []
        for p in e.properties:
            if not p.name or not p.default_value:
                continue
            if p.unit_ref:
                prop_lines.append(f"- {p.name} = {p.default_value} {p.unit_ref}")
            else:
                prop_lines.append(f"- {p.name} = {p.default_value}")
        if prop_lines:
            parts.append("Properties:")
            parts.extend(prop_lines)

        tag_lines = []
        for k, v in e.tags.items():
            if k in tag_skip_keys:
                continue
            if v:
                tag_lines.append(f"- {k}: {v}")
        if tag_lines:
            parts.append("Attributes:")
            parts.extend(tag_lines)

        rel_lines = []
        for rel_id in trimmed_list(e.relation_ids, 12):
            rel = relations.get(rel_id)
            if rel is None:
                continue
            src_name = elements[rel.source_id].name if rel.source_id in elements else None
            tgt_name = elements[rel.target_id].name if rel.target_id in elements else None
            rel_lines.append(
                f"- {relation_display_type(rel)} [{short_id(rel.id)}]: "
                f"{display_ref(src_name, rel.source_id)} -> {display_ref(tgt_name, rel.target_id)}"
            )
        if rel_lines:
            parts.append("Connected relations:")
            parts.extend(rel_lines)

        rows.append({
            "chunk_id": f"node::{e.id}::0",
            "chunk_type": "node",
            "node_id": e.id,
            "name": e.name,
            "source_id": e.source_id,
            "type": e_type,
            "text": "\n".join(parts),
        })

    return rows


def build_relation_chunk_rows(
    relations: dict[str, RelationRecord],
    elements: dict[str, ElementRecord],
    packages: dict[str, PackageRecord],
    diagrams: dict[str, DiagramRecord],
) -> list[dict]:
    rows = []
    for r in relations.values():
        r_type = relation_display_type(r)
        source_name = elements[r.source_id].name if r.source_id in elements else None
        target_name = elements[r.target_id].name if r.target_id in elements else None
        source_type = element_display_type(elements[r.source_id]) if r.source_id in elements else None
        target_type = element_display_type(elements[r.target_id]) if r.target_id in elements else None
        pkg_names = package_path_names_by_id(r.package_id, packages)

        parts = [f"[{r_type}] {display_ref(source_name, r.source_id)} -> {display_ref(target_name, r.target_id)}"]

        if r.xmi_type:
            parts.append(f"Relation type: {r.xmi_type}")

        if source_name or r.source_id:
            if source_type:
                parts.append(f"Source element: {display_ref(source_name, r.source_id)} ({source_type})")
            else:
                parts.append(f"Source element: {display_ref(source_name, r.source_id)}")

        if target_name or r.target_id:
            if target_type:
                parts.append(f"Target element: {display_ref(target_name, r.target_id)} ({target_type})")
            else:
                parts.append(f"Target element: {display_ref(target_name, r.target_id)}")

        if pkg_names:
            parts.append(f"Package: {' / '.join(pkg_names)}")

        if r.diagram_refs:
            parts.append("Appears in diagrams:")
            for d_id in trimmed_list(r.diagram_refs, 10):
                d_name = diagrams[d_id].name if d_id in diagrams else None
                parts.append(f"- {display_ref(d_name, d_id)}")

        if r.direction:
            parts.append(f"Direction: {r.direction}")

        rows.append({
            "chunk_id": f"relation::{r.id}::0",
            "chunk_type": "relation",
            "relation_id": r.id,
            "source_id": r.source_id,
            "target_id": r.target_id,
            "source_name": source_name,
            "target_name": target_name,
            "type": r_type,
            "text": "\n".join(parts),
        })

    return rows


def build_diagram_chunk_rows(
    diagrams: dict[str, DiagramRecord],
    elements: dict[str, ElementRecord],
    relations: dict[str, RelationRecord],
    packages: dict[str, PackageRecord],
) -> list[dict]:
    rows = []
    for d in diagrams.values():
        pkg_names = package_path_names_by_id(d.package_id, packages)
        parts = [f"[Diagram] {display_ref(d.name, d.id)}"]

        if d.diagram_type:
            parts.append(f"Type: {d.diagram_type}")
        if pkg_names:
            parts.append(f"Package: {' / '.join(pkg_names)}")

        element_names = []
        for eid in d.element_ids:
            if eid in elements:
                element_names.append(display_ref(elements[eid].name, eid))
        if element_names:
            parts.append("Elements:")
            for name in trimmed_list(element_names, 15):
                parts.append(f"- {name}")
            if len(element_names) > 15:
                parts.append("Additional elements omitted.")

        relation_lines = []
        for rid in d.relation_ids:
            rel = relations.get(rid)
            if rel is None:
                continue
            src_name = elements[rel.source_id].name if rel.source_id in elements else None
            tgt_name = elements[rel.target_id].name if rel.target_id in elements else None
            relation_lines.append(
                f"- {relation_display_type(rel)} [{short_id(rid)}]: "
                f"{display_ref(src_name, rel.source_id)} -> {display_ref(tgt_name, rel.target_id)}"
            )
        if relation_lines:
            parts.append("Relations:")
            for line in trimmed_list(relation_lines, 15):
                parts.append(line)
            if len(relation_lines) > 15:
                parts.append("Additional relations omitted.")

        rows.append({
            "chunk_id": f"diagram::{d.id}::0",
            "chunk_type": "diagram",
            "diagram_id": d.id,
            "name": d.name,
            "type": d.diagram_type or "Diagram",
            "text": "\n".join(parts),
        })

    return rows


def build_package_chunk_rows(
    packages: dict[str, PackageRecord],
    elements: dict[str, ElementRecord],
    diagrams: dict[str, DiagramRecord],
) -> list[dict]:
    rows = []
    for p in packages.values():
        path_names = package_path_names_by_id(p.id, packages)
        parts = [f"[Package] {display_ref(p.name, p.id)}"]

        if path_names:
            parts.append(f"Path: {' / '.join(path_names)}")

        child_package_names = [display_ref(packages[pid].name, pid) for pid in p.child_package_ids if pid in packages]
        if child_package_names:
            parts.append("Child packages:")
            for name in trimmed_list(child_package_names, 12):
                parts.append(f"- {name}")
            if len(child_package_names) > 12:
                parts.append("Additional child packages omitted.")

        element_names = [display_ref(elements[eid].name, eid) for eid in p.element_ids if eid in elements]
        if element_names:
            parts.append("Elements:")
            for name in trimmed_list(element_names, 15):
                parts.append(f"- {name}")
            if len(element_names) > 15:
                parts.append("Additional elements omitted.")

        diagram_names = [display_ref(diagrams[did].name, did) for did in p.diagram_ids if did in diagrams]
        if diagram_names:
            parts.append("Diagrams:")
            for name in trimmed_list(diagram_names, 12):
                parts.append(f"- {name}")
            if len(diagram_names) > 12:
                parts.append("Additional diagrams omitted.")

        rows.append({
            "chunk_id": f"package::{p.id}::0",
            "chunk_type": "package",
            "package_id": p.id,
            "name": p.name,
            "type": "Package",
            "text": "\n".join(parts),
        })

    return rows


# =========================
# Main
# =========================

def main() -> None:
    input_path = Path("data/real/bike.xml")
    out_dir = Path("data/processed/bike")

    tree = ET.parse(input_path)
    root = tree.getroot()

    packages: dict[str, PackageRecord] = {}
    diagrams: dict[str, DiagramRecord] = {}
    elements: dict[str, ElementRecord] = {}
    relations: dict[str, RelationRecord] = {}

    parse_packages(root, packages)
    parse_basic_elements(root, elements, packages)
    parse_diagrams(root, diagrams, packages)
    parse_sysml_semantics(root, elements, relations)
    parse_extension_elements(root, elements)
    parse_relations(root, relations, packages)
    link_refs(packages, diagrams, elements, relations)

    package_rows = [p.__dict__ for p in packages.values()]
    diagram_rows = [
        {
            **{k: v for k, v in d.__dict__.items() if k not in {"element_views", "relation_views"}},
            "element_views": [ev.__dict__ for ev in d.element_views],
            "relation_views": [rv.__dict__ for rv in d.relation_views],
        }
        for d in diagrams.values()
    ]
    element_rows = build_node_rows(elements)
    relation_rows = build_edge_rows(relations)

    node_chunks = build_node_chunk_rows(elements, relations, diagrams)
    relation_chunks = build_relation_chunk_rows(relations, elements, packages, diagrams)
    diagram_chunks = build_diagram_chunk_rows(diagrams, elements, relations, packages)
    package_chunks = build_package_chunk_rows(packages, elements, diagrams)

    write_jsonl(out_dir / "packages.jsonl", package_rows)
    write_jsonl(out_dir / "diagrams.jsonl", diagram_rows)
    write_jsonl(out_dir / "elements.jsonl", element_rows)
    write_jsonl(out_dir / "relations.jsonl", relation_rows)

    write_jsonl(out_dir / "chunks_nodes.jsonl", node_chunks)
    write_jsonl(out_dir / "chunks_relations.jsonl", relation_chunks)
    write_jsonl(out_dir / "chunks_diagrams.jsonl", diagram_chunks)
    write_jsonl(out_dir / "chunks_packages.jsonl", package_chunks)

    all_chunks = node_chunks + relation_chunks + diagram_chunks + package_chunks
    write_jsonl(out_dir / "chunks.jsonl", all_chunks)

    print(f"packages={len(packages)}")
    print(f"diagrams={len(diagrams)}")
    print(f"elements={len(elements)}")
    print(f"relations={len(relations)}")
    print(f"node_chunks={len(node_chunks)}")
    print(f"relation_chunks={len(relation_chunks)}")
    print(f"diagram_chunks={len(diagram_chunks)}")
    print(f"package_chunks={len(package_chunks)}")


if __name__ == "__main__":
    main()
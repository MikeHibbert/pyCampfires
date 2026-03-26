import os
import json
from datetime import datetime
import importlib.util
from pathlib import Path

# Robust import of demo module without relying on package layout
_demo_path = Path(__file__).resolve().parent.parent / "demos" / "team_rag_auditor_demo.py"
spec = importlib.util.spec_from_file_location("team_rag_auditor_demo", str(_demo_path))
_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_demo)
maybe_export_langgraph_gui = _demo.maybe_export_langgraph_gui
maybe_export_langgraph_studio = _demo.maybe_export_langgraph_studio


class _Node:
    def __init__(self, id: str, label: str, properties: dict | None = None):
        self.id = id
        self.label = label
        self.properties = properties or {}


class _Edge:
    def __init__(self, src: str, dst: str, type: str, properties: dict | None = None):
        self.src = src
        self.dst = dst
        self.type = type
        self.properties = properties or {}


class _MiniGraphStore:
    def __init__(self):
        self._nodes = {}
        self._edges = []

    def add_node(self, node: _Node):
        self._nodes[node.id] = node

    def add_edge(self, edge: _Edge):
        self._edges.append(edge)


def _build_graph_with_selection_and_findings():
    gs = _MiniGraphStore()
    # Orderer + selection + camper
    gs.add_node(_Node("orderer:Orderer", "Orderer", {"name": "Orderer"}))
    gs.add_node(_Node("selection:abc", "Selection", {"criteria": {"skill": "python"}}))
    gs.add_node(_Node("camper:alice", "Camper", {"name": "alice", "role": "designer"}))
    gs.add_edge(_Edge("orderer:Orderer", "selection:abc", "MADE_SELECTION", {}))
    gs.add_edge(_Edge("orderer:Orderer", "camper:alice", "CREATED_MEMBER", {}))
    gs.add_edge(_Edge("selection:abc", "camper:alice", "ASSIGNED_ROLE", {"role": "designer"}))
    gs.add_edge(_Edge("selection:abc", "camper:alice", "SELECTED_MEMBER", {}))

    # Stage + procedure + finding
    gs.add_node(_Node("stage:design", "Stage", {"name": "design"}))
    gs.add_node(_Node("procedure:design", "Procedure", {"name": "Design Procedure"}))
    gs.add_node(_Node("procstep:t1:design:1", "ProcedureStep", {"name": "attempt-1", "actor": "Auditor"}))
    gs.add_node(_Node("finding:t1:design:12345", "Finding", {"summary": "ok", "importance": 0.8}))
    gs.add_edge(_Edge("stage:design", "procedure:design", "HAS_PROCEDURE", {}))
    gs.add_edge(_Edge("procedure:design", "procstep:t1:design:1", "HAS_STEP", {"order_index": 1}))
    gs.add_edge(_Edge("procstep:t1:design:1", "finding:t1:design:12345", "GENERATED_FINDING", {}))
    gs.add_edge(_Edge("stage:design", "finding:t1:design:12345", "HAS_FINDING", {}))
    return gs


def test_gui_export_label_filter(tmp_path, monkeypatch):
    # Enable GUI with label filter limited to non-selection/non-camper
    monkeypatch.setenv("CAMPFIRES_ENABLE_LANGGRAPH_GUI", "1")
    monkeypatch.setenv("CAMPFIRES_LANGGRAPH_EXPORT_FORMAT", "simple")
    monkeypatch.setenv("CAMPFIRES_EXPORT_LABELS", "Orderer,Stage,Procedure,ProcedureStep,Finding")

    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    task_description = "Filtered Export"
    solution_data = {"summary": "Summary"}
    graph_store = _build_graph_with_selection_and_findings()

    maybe_export_langgraph_gui(task_description, solution_data, str(out_dir), graph_store)

    out_path = out_dir / "team_rag_langgraph.json"
    assert out_path.exists(), "Filtered GUI export JSON file was not written"

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = {n["label"] for n in data["nodes"]}
    # Ensure Selection and Camper were filtered out
    assert "Selection" not in labels
    assert "Camper" not in labels
    # Retained labels exist
    assert {"Orderer", "Stage", "Procedure", "ProcedureStep", "Finding"}.issubset(labels)

    edge_labels = {e["label"] for e in data["edges"]}
    # Selection-related edges removed
    assert "MADE_SELECTION" not in edge_labels
    assert "CREATED_MEMBER" not in edge_labels
    assert "ASSIGNED_ROLE" not in edge_labels
    assert "SELECTED_MEMBER" not in edge_labels
    # Procedure/finding-related edges remain
    assert {"HAS_PROCEDURE", "HAS_STEP", "HAS_FINDING", "GENERATED_FINDING"}.issubset(edge_labels)


def test_minimal_mode_trims_properties(tmp_path, monkeypatch):
    # Enable minimal mode for both exporters
    monkeypatch.setenv("CAMPFIRES_ENABLE_LANGGRAPH_GUI", "1")
    monkeypatch.setenv("CAMPFIRES_ENABLE_LANGGRAPH_STUDIO", "1")
    monkeypatch.setenv("CAMPFIRES_EXPORT_MINIMAL", "1")
    monkeypatch.setenv("CAMPFIRES_LANGGRAPH_EXPORT_FORMAT", "simple")

    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    task_description = "Minimal Export"
    solution_data = {"summary": "Summary"}
    graph_store = _build_graph_with_selection_and_findings()

    maybe_export_langgraph_gui(task_description, solution_data, str(out_dir), graph_store)
    maybe_export_langgraph_studio(task_description, solution_data, str(out_dir), graph_store)

    # GUI
    gui_path = out_dir / "team_rag_langgraph.json"
    assert gui_path.exists()
    with open(gui_path, "r", encoding="utf-8") as f:
        gui = json.load(f)
    # Finding keeps summary/importance; others mostly name-only
    for n in gui["nodes"]:
        props = n.get("properties", {})
        if n["label"] == "Finding":
            assert set(props.keys()).issubset({"summary", "importance"})
        else:
            assert set(props.keys()).issubset({"name"})
    for e in gui["edges"]:
        assert e.get("properties", {}) == {}

    # Studio
    studio_path = out_dir / "team_rag_langgraph_studio.json"
    assert studio_path.exists()
    with open(studio_path, "r", encoding="utf-8") as f:
        studio = json.load(f)
    for n in studio["nodes"]:
        assert n.get("data", {}) == {}
    for e in studio["edges"]:
        assert e.get("data", {}) == {}


def test_non_serializable_properties_are_coerced(tmp_path, monkeypatch):
    # Enable GUI export
    monkeypatch.setenv("CAMPFIRES_ENABLE_LANGGRAPH_GUI", "1")
    monkeypatch.setenv("CAMPFIRES_LANGGRAPH_EXPORT_FORMAT", "simple")

    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    # Build graph with non-serializable values
    gs = _MiniGraphStore()
    gs.add_node(_Node("orderer:Orderer", "Orderer", {"name": "Orderer", "created_at": datetime.now()}))
    gs.add_node(_Node("finding:t1:design:ns", "Finding", {"summary": "ok", "importance": 0.9, "obj": object()}))
    gs.add_edge(_Edge("orderer:Orderer", "finding:t1:design:ns", "LINK", {"when": datetime.now(), "obj": object()}))

    maybe_export_langgraph_gui("NS Props", {"summary": "s"}, str(out_dir), gs)

    out_path = out_dir / "team_rag_langgraph.json"
    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify coercion
    orderer = next(n for n in data["nodes"] if n["id"] == "orderer:Orderer")
    assert isinstance(orderer["properties"].get("created_at", ""), str)
    finding = next(n for n in data["nodes"] if n["id"].startswith("finding:"))
    assert isinstance(finding["properties"].get("obj", ""), str)
    edge = next(e for e in data["edges"] if e["source"] == "orderer:Orderer")
    assert isinstance(edge["properties"].get("when", ""), str)
    assert isinstance(edge["properties"].get("obj", ""), str)
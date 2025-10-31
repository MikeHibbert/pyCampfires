import os
import json
import shutil
from datetime import datetime

import importlib.util
from pathlib import Path

# Robust import of demo module without relying on package layout
_demo_path = Path(__file__).resolve().parent.parent / "demos" / "team_rag_auditor_demo.py"
spec = importlib.util.spec_from_file_location("team_rag_auditor_demo", str(_demo_path))
_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_demo)
maybe_export_langgraph_gui = _demo.maybe_export_langgraph_gui


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


def _build_min_graph():
    gs = _MiniGraphStore()
    # Core actors
    gs.add_node(_Node("orderer:Orderer", "Orderer", {"name": "Orderer", "created_at": datetime.utcnow().isoformat()}))
    gs.add_node(_Node("selection:team", "Selection", {"team_size": 3}))
    gs.add_edge(_Edge("orderer:Orderer", "selection:team", "MADE_SELECTION", {}))

    # Team member
    gs.add_node(_Node("camper:alice", "Camper", {"role": "Designer"}))
    gs.add_edge(_Edge("orderer:Orderer", "camper:alice", "CREATED_MEMBER", {"created_at": datetime.utcnow().isoformat()}))
    gs.add_edge(_Edge("selection:team", "camper:alice", "ASSIGNED_ROLE", {"role": "Designer"}))

    # Stage + finding
    gs.add_node(_Node("stage:design", "Stage", {"name": "design"}))
    gs.add_node(_Node("finding:f1", "Finding", {"summary": "Design draft", "timestamp": datetime.utcnow().isoformat(), "importance": 1, "tags": ["pass"]}))
    gs.add_edge(_Edge("stage:design", "finding:f1", "HAS_FINDING", {}))

    # Procedure + step + generated finding link
    gs.add_node(_Node("procedure:design", "Procedure", {"name": "Design Procedure"}))
    gs.add_edge(_Edge("stage:design", "procedure:design", "HAS_PROCEDURE", {}))
    gs.add_node(_Node("step:design:1", "ProcedureStep", {"name": "Sketch", "actor": "alice", "status": "done", "importance": 1}))
    gs.add_edge(_Edge("procedure:design", "step:design:1", "HAS_STEP", {"order_index": 1}))
    gs.add_edge(_Edge("step:design:1", "finding:f1", "GENERATED_FINDING", {}))
    return gs


def test_gui_export_writes_expected_json(tmp_path, monkeypatch):
    # Enable GUI export for the function call
    monkeypatch.setenv("CAMPFIRES_ENABLE_LANGGRAPH_GUI", "1")
    monkeypatch.setenv("CAMPFIRES_LANGGRAPH_EXPORT_FORMAT", "simple")

    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    # Minimal inputs
    task_description = "Test Export"
    solution_data = {"summary": "Summary"}
    graph_store = _build_min_graph()

    # Call exporter
    maybe_export_langgraph_gui(task_description, solution_data, str(out_dir), graph_store)

    # Validate file
    out_path = out_dir / "team_rag_langgraph.json"
    assert out_path.exists(), "GUI export JSON file was not written"

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic structure
    assert "nodes" in data and isinstance(data["nodes"], list)
    assert "edges" in data and isinstance(data["edges"], list)
    assert "metadata" in data and isinstance(data["metadata"], dict)

    labels = {n["label"] for n in data["nodes"]}
    assert {"Orderer", "Selection", "Camper", "Stage", "Finding", "Procedure", "ProcedureStep"}.issubset(labels)

    edge_labels = {e["label"] for e in data["edges"]}
    required_edges = {"MADE_SELECTION", "CREATED_MEMBER", "ASSIGNED_ROLE", "HAS_FINDING", "HAS_PROCEDURE", "HAS_STEP", "GENERATED_FINDING"}
    assert required_edges.issubset(edge_labels)

    # Metadata contains process flows and procedures
    meta = data["metadata"]
    assert "process_flows" in meta and isinstance(meta["process_flows"], dict)
    assert "procedures" in meta and isinstance(meta["procedures"], dict)
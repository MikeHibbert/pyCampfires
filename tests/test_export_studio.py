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


def _build_min_graph():
    gs = _MiniGraphStore()
    # Orderer and a Stage
    gs.add_node(_Node("orderer:Orderer", "Orderer", {"name": "Orderer", "created_at": datetime.utcnow().isoformat()}))
    gs.add_node(_Node("stage:problem_solving", "Stage", {"name": "problem_solving"}))
    gs.add_edge(_Edge("orderer:Orderer", "stage:problem_solving", "STARTS", {}))

    # Procedure + step
    gs.add_node(_Node("procedure:ps", "Procedure", {"name": "PS Procedure"}))
    gs.add_edge(_Edge("stage:problem_solving", "procedure:ps", "HAS_PROCEDURE", {}))
    gs.add_node(_Node("step:ps:1", "ProcedureStep", {"name": "Analyze", "actor": "bob", "status": "done"}))
    gs.add_edge(_Edge("procedure:ps", "step:ps:1", "HAS_STEP", {"order_index": 1}))
    return gs


def test_studio_export_writes_expected_json(tmp_path, monkeypatch):
    # Enable Studio export for the function call
    monkeypatch.setenv("CAMPFIRES_ENABLE_LANGGRAPH_STUDIO", "1")

    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    # Minimal inputs
    task_description = "Test Export Studio"
    solution_data = {"summary": "Summary"}
    graph_store = _build_min_graph()

    # Call exporter
    maybe_export_langgraph_studio(task_description, solution_data, str(out_dir), graph_store)

    # Validate file
    out_path = out_dir / "team_rag_langgraph_studio.json"
    assert out_path.exists(), "Studio export JSON file was not written"

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic structure
    assert "nodes" in data and isinstance(data["nodes"], list)
    assert "edges" in data and isinstance(data["edges"], list)
    assert "start_node" in data
    assert "metadata" in data and isinstance(data["metadata"], dict)

    # Studio nodes use type/data
    types = {n["type"] for n in data["nodes"]}
    assert {"Orderer", "Stage", "Procedure", "ProcedureStep"}.issubset(types)
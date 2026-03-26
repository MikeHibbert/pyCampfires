import os
import json
import importlib.util
from datetime import datetime

import pytest

pytestmark = pytest.mark.asyncio


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


def _import_studio_exporter():
    # Robust import using file path to avoid package import issues
    file_path = os.path.join(os.path.dirname(__file__), "..", "demos", "team_rag_auditor_demo.py")
    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location("team_rag_auditor_demo", file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module.maybe_export_langgraph_studio


def test_studio_non_serializable_properties_are_coerced(tmp_path, monkeypatch):
    maybe_export_langgraph_studio = _import_studio_exporter()

    # Enable Studio export and ensure minimal mode is off
    monkeypatch.setenv("CAMPFIRES_ENABLE_LANGGRAPH_STUDIO", "1")
    monkeypatch.setenv("CAMPFIRES_EXPORT_MINIMAL", "0")

    out_dir = tmp_path / "reports"
    os.makedirs(out_dir, exist_ok=True)

    # Build graph with non-serializable values
    gs = _MiniGraphStore()
    gs.add_node(_Node("orderer:Orderer", "Orderer", {"name": "Orderer", "created_at": datetime.now()}))
    gs.add_node(_Node("finding:t1:design:ns", "Finding", {"summary": "ok", "importance": 0.9, "obj": object()}))
    gs.add_edge(_Edge("orderer:Orderer", "finding:t1:design:ns", "LINK", {"when": datetime.now(), "obj": object()}))

    # Call exporter
    maybe_export_langgraph_studio("NS Props Studio", {"summary": "s"}, str(out_dir), gs)

    out_path = out_dir / "team_rag_langgraph_studio.json"
    assert out_path.exists()
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify coercion for node data
    orderers = [n for n in data["nodes"] if n["type"] == "Orderer"]
    assert orderers, "Orderer node missing"
    orderer_data = orderers[0]["data"]
    assert isinstance(orderer_data.get("created_at", ""), str)

    findings = [n for n in data["nodes"] if n["type"] == "Finding"]
    assert findings, "Finding node missing"
    finding_data = findings[0]["data"]
    assert isinstance(finding_data.get("obj", ""), str)

    # Verify coercion for edge data
    edges = [e for e in data["edges"] if e["source"] == "orderer:Orderer" and e["target"].startswith("finding:")]
    assert edges, "Edge from orderer to finding missing"
    edge_data = edges[0]["data"]
    assert isinstance(edge_data.get("when", ""), str)
    assert isinstance(edge_data.get("obj", ""), str)
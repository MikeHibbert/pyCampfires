from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class Node:
    id: str
    label: str
    properties: Dict[str, Any]


@dataclass
class Edge:
    src: str
    dst: str
    type: str
    properties: Dict[str, Any]


class GraphStore:
    """
    Minimal graph store interface for knowledge graph and context management.
    """

    async def upsert_node(self, node: Node) -> None:
        raise NotImplementedError

    async def upsert_edge(self, edge: Edge) -> None:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Optional[Node]:
        raise NotImplementedError

    async def query_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Node]:
        raise NotImplementedError

    async def add_context(self, node_id: str, context: Dict[str, Any]) -> None:
        raise NotImplementedError

    async def get_context(self, node_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class InMemoryGraphStore(GraphStore):
    """
    Simple in-memory graph store used as a default fallback.
    """

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._context: Dict[str, Dict[str, Any]] = {}

    async def upsert_node(self, node: Node) -> None:
        self._nodes[node.id] = node

    async def upsert_edge(self, edge: Edge) -> None:
        self._edges.append(edge)

    async def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    async def query_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Node]:
        neighbors: List[Node] = []
        for e in self._edges:
            if e.src == node_id and (edge_type is None or e.type == edge_type):
                dst = self._nodes.get(e.dst)
                if dst:
                    neighbors.append(dst)
        return neighbors

    async def add_context(self, node_id: str, context: Dict[str, Any]) -> None:
        existing = self._context.get(node_id, {})
        existing.update(context)
        self._context[node_id] = existing

    async def get_context(self, node_id: str) -> Dict[str, Any]:
        return self._context.get(node_id, {})
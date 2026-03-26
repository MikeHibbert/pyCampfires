from typing import Dict, Any, List, Optional

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:
    GraphDatabase = None  # type: ignore

from .graph_store import GraphStore, Node, Edge


class Neo4jGraphStore(GraphStore):
    """
    Optional Neo4j-backed graph store. Requires `neo4j` driver.
    """

    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        if GraphDatabase is None:
            raise ImportError("neo4j driver not installed. Install with `pip install neo4j`. ")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def _session(self):
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    async def upsert_node(self, node: Node) -> None:
        cypher = (
            "MERGE (n:`%s` {id: $id}) "
            "SET n += $props" % node.label
        )
        with self._session() as session:
            session.run(cypher, id=node.id, props=node.properties)

    async def upsert_edge(self, edge: Edge) -> None:
        cypher = (
            "MATCH (a {id: $src}), (b {id: $dst}) "
            "MERGE (a)-[r:`%s`]->(b) "
            "SET r += $props" % edge.type
        )
        with self._session() as session:
            session.run(cypher, src=edge.src, dst=edge.dst, props=edge.properties)

    async def get_node(self, node_id: str) -> Optional[Node]:
        cypher = "MATCH (n {id: $id}) RETURN labels(n) AS labels, properties(n) AS props"
        with self._session() as session:
            result = session.run(cypher, id=node_id)
            record = result.single()
            if not record:
                return None
            labels = record["labels"]
            props = record["props"]
            return Node(id=node_id, label=labels[0] if labels else "Node", properties=props)

    async def query_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Node]:
        if edge_type:
            cypher = (
                "MATCH (a {id: $id})-[:`%s`]->(b) "
                "RETURN labels(b) AS labels, properties(b) AS props" % edge_type
            )
        else:
            cypher = (
                "MATCH (a {id: $id})-->(b) "
                "RETURN labels(b) AS labels, properties(b) AS props"
            )
        with self._session() as session:
            result = session.run(cypher, id=node_id)
            nodes: List[Node] = []
            for record in result:
                labels = record["labels"]
                props = record["props"]
                nodes.append(Node(id=str(props.get("id", "")), label=labels[0] if labels else "Node", properties=props))
            return nodes

    async def add_context(self, node_id: str, context: Dict[str, Any]) -> None:
        cypher = "MATCH (n {id: $id}) SET n.context = coalesce(n.context, {}) + $ctx"
        with self._session() as session:
            session.run(cypher, id=node_id, ctx=context)

    async def get_context(self, node_id: str) -> Dict[str, Any]:
        cypher = "MATCH (n {id: $id}) RETURN coalesce(n.context, {}) AS ctx"
        with self._session() as session:
            result = session.run(cypher, id=node_id)
            record = result.single()
            return dict(record["ctx"]) if record and record["ctx"] else {}
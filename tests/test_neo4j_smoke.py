import os
import uuid
import pytest

pytestmark = pytest.mark.asyncio


def _env(var: str) -> str | None:
    val = os.getenv(var)
    return val if val and val.strip() else None


async def _maybe_init_store():
    try:
        from campfires.core.neo4j_graph_store import Neo4jGraphStore  # type: ignore
        uri = _env("NEO4J_URI")
        user = _env("NEO4J_USER")
        password = _env("NEO4J_PASSWORD")
        database = _env("NEO4J_DATABASE")
        if not (uri and user and password):
            pytest.skip("Neo4j env not set; skipping smoke test")
        return Neo4jGraphStore(uri=uri, user=user, password=password, database=database)
    except ImportError:
        pytest.skip("neo4j driver not installed; skipping smoke test")


async def test_neo4j_smoke_labels_and_edges_preserved():
    store = await _maybe_init_store()
    if store is None:  # pragma: no cover - guarded by skip
        return

    from campfires.core.graph_store import Node, Edge

    # Unique IDs to avoid collisions across runs
    orderer_id = f"orderer:{uuid.uuid4()}"
    selection_id = f"selection:{uuid.uuid4()}"
    camper_id = f"camper:{uuid.uuid4()}"

    # Upsert nodes with labels and explicit 'id' property for retrieval
    await store.upsert_node(Node(id=orderer_id, label="Orderer", properties={"id": orderer_id, "name": "Orderer"}))
    await store.upsert_node(Node(id=selection_id, label="Selection", properties={"id": selection_id, "criteria": {"skill": "python"}}))
    await store.upsert_node(Node(id=camper_id, label="Camper", properties={"id": camper_id, "name": "alice", "role": "designer"}))

    # Upsert edges connecting the nodes
    await store.upsert_edge(Edge(src=orderer_id, dst=selection_id, type="MADE_SELECTION", properties={}))
    await store.upsert_edge(Edge(src=orderer_id, dst=camper_id, type="CREATED_MEMBER", properties={}))
    await store.upsert_edge(Edge(src=selection_id, dst=camper_id, type="ASSIGNED_ROLE", properties={"role": "designer"}))

    # Verify labels and properties via get_node
    got_orderer = await store.get_node(orderer_id)
    assert got_orderer is not None
    assert got_orderer.label == "Orderer"
    assert got_orderer.properties.get("id") == orderer_id

    got_selection = await store.get_node(selection_id)
    assert got_selection is not None
    assert got_selection.label == "Selection"
    assert got_selection.properties.get("criteria", {}).get("skill") == "python"

    # Neighbor query by edge type preserves destination labels
    via_selection = await store.query_neighbors(orderer_id, edge_type="MADE_SELECTION")
    assert any(n.label == "Selection" and n.properties.get("id") == selection_id for n in via_selection)

    via_created = await store.query_neighbors(orderer_id, edge_type="CREATED_MEMBER")
    assert any(n.label == "Camper" and n.properties.get("id") == camper_id for n in via_created)

    # Context storage round-trip
    await store.add_context(selection_id, {"status": "in_progress", "attempts": 1})
    ctx = await store.get_context(selection_id)
    assert ctx.get("status") == "in_progress"
    assert ctx.get("attempts") == 1
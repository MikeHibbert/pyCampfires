#!/usr/bin/env python3
"""
Neo4j Graph Store Demo (Dockerized)

Prerequisites:
- Docker Desktop installed and running
- Python package `neo4j` installed: `pip install neo4j`
- Start Neo4j with: `docker compose -f docker-compose.neo4j.yml up -d`

This script connects to the local Neo4j instance and exercises the
Neo4jGraphStore: node/edge upsert, neighbor query, and context storage.
"""

import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import campfires
sys.path.append(str(Path(__file__).parent.parent))

from campfires.core.neo4j_graph_store import Neo4jGraphStore
from campfires.core.graph_store import Node, Edge


async def main():
    store = Neo4jGraphStore(uri='bolt://localhost:7687', user='neo4j', password='password', database='neo4j')

    # Upsert nodes
    await store.upsert_node(Node(id='u1', label='User', properties={'id': 'u1', 'name': 'Ada'}))
    await store.upsert_node(Node(id='t42', label='Task', properties={'id': 't42', 'title': 'Greet the world'}))

    # Upsert edge
    await store.upsert_edge(Edge(src='u1', dst='t42', type='CREATED', properties={'ts': 123}))

    # Add context
    await store.add_context('t42', {'status': 'in_progress', 'attempts': 1})

    # Query neighbors
    neighbors = await store.query_neighbors('u1', 'CREATED')
    print('Neighbors from u1 via CREATED:')
    for n in neighbors:
        print(f"- {n.label} {n.properties}")

    # Read context
    ctx = await store.get_context('t42')
    print('Context for t42:', ctx)


if __name__ == '__main__':
    asyncio.run(main())
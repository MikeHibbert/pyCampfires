Graph Demos
===========

This folder contains demos that showcase graph-powered collaboration and memory.

Neo4j Graph Store Demo
- Spins up a local Neo4j and demonstrates node/edge upsert, neighbor queries, and context.
- Start Neo4j:
  - `docker compose -f docker-compose.neo4j.yml up -d`
- Run demo:
  - `python -m demos.neo4j_graph_demo`

Collaboration Demo (In-Memory or Neo4j)
- Two campers collaborate via the shared graph: one publishes findings, the other retrieves them.
- In-memory (no services):
  - `python -m demos.graph_collab_demo`
- Using Neo4j:
  - Ensure `docker-compose.neo4j.yml` is running.
  - Configure environment for the factory if integrating into your app:
    - `CAMPFIRES_GRAPH_BACKEND=neo4j`
    - `NEO4J_URI=bolt://localhost:7687`
    - `NEO4J_USER=neo4j`
    - `NEO4J_PASSWORD=<password>`
    - Optional: `NEO4J_DATABASE=<database>`
  - The in-repo `graph_collab_demo.py` uses in-memory by default; adapt it to use the factory and pass the Neo4j graph store if desired.

Notes
- Topic normalization: campers can map aliases via `config["topic_aliases"]`.
- Weighted retrieval: findings are ranked by `importance` and `context.confidence`.

Team RAG Auditor Demo (Graph-Enhanced)
- Orchestrates a small team with stage-specific RAG docs and publishes stage findings and team selection events to the shared graph.
- In-memory:
  - `python -m demos.team_rag_auditor_demo`
- Optional: enable Neo4j
  - Set env vars:
    - `CAMPFIRES_USE_NEO4J=1`
    - `NEO4J_URI=bolt://localhost:7687`
    - `NEO4J_USER=neo4j`
    - `NEO4J_PASSWORD=<password>`
    - Optional: `NEO4J_DATABASE=<database>`
  - Run: `python -m demos.team_rag_auditor_demo`

Notes on configuration
- `CAMPFIRES_GRAPH_BACKEND` controls backend selection when using `CampfireFactory` in your own apps.
- `CAMPFIRES_USE_NEO4J` is a demo-specific toggle inside `team_rag_auditor_demo.py` for convenience.
- Both flows respect `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and optional `NEO4J_DATABASE`.

Published graph artifacts
- Team selection: `Orderer` -> `Selection` -> `Camper` edges with selection rationale recorded on `Selection.properties`.
- Stage findings: `Auditor`, `Task`, `Stage` nodes linked to `Finding` with context (issues, recommendations, confidence, coverage).

Tips
- The in-memory variant prints a short graph summary at the end (node/edge counts and the first selection node). On Neo4j, explore via Bloom, Neodash, or Cypher.

LangGraph GUI Option
- Export a simple workflow graph JSON for GUI viewing.
- Enable via env: `CAMPFIRES_ENABLE_LANGGRAPH_GUI=1`
- Run: `python -m demos.team_rag_auditor_demo`
- Output: `reports/team_rag_langgraph.json`
- Notes:
  - The JSON format is lightweight and viewer-agnostic; you can adapt it for LangGraph Studio or custom front-ends.
  - If using LangGraph Studio, load or transform the JSON into a `StateGraph` as needed and visualize the workflow.

Filters and Minimal Mode
- `CAMPFIRES_EXPORT_LABELS`: Comma-separated labels to include in exports. Example: `Orderer,Stage,Procedure,ProcedureStep,Finding`.
- `CAMPFIRES_EXPORT_MINIMAL=1`: Reduce properties and edge data to essentials for compact visualization.
import asyncio
import logging
import os
import json
from typing import Dict, Any

from dataclasses import asdict
from datetime import datetime
from campfires import Campfire, Camper
from campfires.core.orderer import Orderer, OrdererConfig
from campfires.core.html_report_generator import HTMLReportGenerator, WorkflowReport, ReportStage
from campfires.core.team_auditor import TeamAuditor, TeamAuditorConfig
from campfires.core.graph_store import InMemoryGraphStore
from campfires.core.neo4j_graph_store import Neo4jGraphStore
from campfires.core.default_auditor import AuditContext, ValidationResult
from campfires.party_box import LocalDriver
from campfires.core.ollama import OllamaConfig, OllamaClient

# Configure logging
log_file = "./team_rag_auditor_demo.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[console_handler, file_handler])
logger = logging.getLogger(__name__)

class DemoCamper(Camper):
    def __init__(self, name: str, role: str, party_box: LocalDriver, **kwargs):
        super().__init__(party_box, {"name": name, "role": role, **kwargs})
        self.set_role(role)

    async def override_prompt(self, original_prompt: str) -> str:
        return original_prompt

    async def solve(self, *args, **kwargs) -> Dict[str, Any]:
        task_desc = None
        if args:
            task_desc = args[0] if isinstance(args[0], str) else None
        if not task_desc and 'task' in kwargs:
            task_desc = kwargs['task'].get('description')
        logger.info(f"{self.name} is solving: {task_desc or 'unknown task'}")
        solution = {"stage": "problem_solving", "output": f"Initial solution for {task_desc or 'task'}"}
        return solution

    async def design(self, *args, **kwargs) -> Dict[str, Any]:
        base = None
        if args:
            base = args[-1] if isinstance(args[-1], dict) else None
        base = base or kwargs.get('solution_data') or {}
        logger.info(f"{self.name} is designing based on: {base.get('output')}")
        design_output = {"stage": "design", "output": f"Design for {base.get('output')}"}
        return design_output

    async def implement(self, *args, **kwargs) -> Dict[str, Any]:
        base = None
        if args:
            base = args[-1] if isinstance(args[-1], dict) else None
        base = base or kwargs.get('design_data') or {}
        logger.info(f"{self.name} is implementing based on: {base.get('output')}")
        implementation_output = {"stage": "implementation", "output": f"Implementation of {base.get('output')}"}
        return implementation_output

    async def review_rag(self, rag_content: str) -> Dict[str, Any]:
        logger.info(f"{self.name} ({self.get_role()}) is reviewing RAG content (first 100 chars): {rag_content[:100]}...")
        return {"review_summary": "RAG content reviewed successfully."}

async def main():
    logger.info("Starting Team RAG Auditor Demo")

    # Define a simple task
    task = {
        "description": "Develop a simple Python script to calculate Fibonacci sequence up to N.",
        "requirements": [
            "Input N must be a positive integer.",
            "Output should be a list of Fibonacci numbers.",
            "Solution must be efficient for N up to 20."
        ]
    }

    # Initialize OllamaConfig and OllamaClient
    ollama_config = OllamaConfig(
        base_url="http://localhost:11434",
        model="llama2",  # Or your preferred Ollama model
        temperature=0.7,
        max_tokens=500
    )
    async with OllamaClient(ollama_config) as ollama_client:
        # Setup PartyBox
        party_box_path = "./demo_party_box_team_rag"
        if not os.path.exists(party_box_path):
            os.makedirs(party_box_path)
        party_box = LocalDriver(party_box_path, ollama_client=ollama_client)

        # Create dummy RAG documents for stages
        rag_dir = "./demo_rag_docs"
        os.makedirs(rag_dir, exist_ok=True)
        with open(os.path.join(rag_dir, "problem_solving_rag.txt"), "w") as f:
            f.write("Problem Solving RAG: Focus on understanding the problem and breaking it down. Consider edge cases.")
        with open(os.path.join(rag_dir, "design_rag.txt"), "w") as f:
            f.write("Design RAG: Focus on algorithm choice, data structures, and modularity. Pseudocode is helpful.")
        with open(os.path.join(rag_dir, "implementation_rag.txt"), "w") as f:
            f.write("Implementation RAG: Focus on clean code, error handling, and testing. Use Python best practices.")

        # Initialize TeamAuditorConfig with persistence and stage-specific RAG maps
        auditor_persistence_path = os.path.join(party_box_path, "auditor_state.json")
        team_auditor_config = TeamAuditorConfig(
            max_retries=2,
            retry_delay_seconds=1,
            persistence_path=auditor_persistence_path,
            stage_rag_map={
                "problem_solving": os.path.join(rag_dir, "problem_solving_rag.txt"),
                "design": os.path.join(rag_dir, "design_rag.txt"),
                "implementation": os.path.join(rag_dir, "implementation_rag.txt"),
            }
        )

        # Default Auditor Configuration (passed directly to TeamAuditor config)
        default_auditor_config_dict = {
            "validation_model": ollama_config.model,  # Use the model from OllamaConfig
            "temperature": 0.7,
            "max_tokens": 500
        }

        # Initialize OrdererConfig
        orderer_config = OrdererConfig()

        # Create Auditor and Orderer instances
        orderer = Orderer(party_box=party_box, auditor_class=TeamAuditor, config={"orderer": orderer_config.__dict__})
        # Choose graph store: InMemory by default; set CAMPFIRES_USE_NEO4J=1 to use Neo4j
        use_neo4j = os.getenv("CAMPFIRES_USE_NEO4J", "0").lower() in ("1", "true", "yes")
        if use_neo4j:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "neo4j")
            logger.info(f"Using Neo4jGraphStore at {uri} with user {user}")
            graph_store = Neo4jGraphStore(uri=uri, user=user, password=password)
        else:
            graph_store = InMemoryGraphStore()
        orderer.set_graph_store(graph_store)
        auditor = TeamAuditor(
            party_box=party_box,
            config={
                "team_auditor": asdict(team_auditor_config),
                "default_auditor": default_auditor_config_dict,
                "graph_enabled": True,
            },
        )
        # Also set graph store directly on auditor for completeness
        auditor.set_graph_store(graph_store)

        # Create campers
        camper1 = DemoCamper("Alice", "Problem Solver", party_box)
        camper2 = DemoCamper("Bob", "Designer", party_box)
        camper3 = DemoCamper("Charlie", "Implementer", party_box)

        campers = [camper1, camper2, camper3]

        # Create Campfire with "team_rag" mode
        campfire = Campfire(name="Team RAG Auditor Campfire", campers=campers, party_box=party_box, mode="team_rag", config={"orderer": orderer_config.__dict__})

        logger.info("Campers added to Campfire. Starting orchestration...")

        # Select team and initialize campers to trigger graph publishing
        roles = [getattr(c, 'role', None) or getattr(c, '_role', None) or c.config.get('role') or 'unknown' for c in campers]
        selected_team = await orderer.select_team(campers, {"roles": roles, "strategy": "all_available"})
        orderer.campers = selected_team
        await orderer.initialize_campers(orderer.campers, {"task_id": "task_123", "description": task.get("description")})

        # Run the orchestration
        task_id = "task_123"
        task_description = task.get("description")
        requirements = [
            "Input N must be a positive integer",
            "Output should be a list of Fibonacci numbers",
            "Efficient up to N=20"
        ]
        # Provide a meaningful summary and details to enrich the auditor's context
        solution_data = {
            "summary": "Python function to compute Fibonacci up to N with simple iteration",
            "approach": "Iterative loop accumulating sequence values; validates positive integer input; handles N<=2 quickly",
            "complexity": {
                "time": "O(N)",
                "space": "O(N)"
            },
            "example": {
                "input": 10,
                "output": [0,1,1,2,3,5,8,13,21,34]
            }
        }

        final_report = await orderer.orchestrate_problem_solving(
            task_id=task_id,
            task_description=task_description,
            requirements=requirements,
            solution_data=solution_data
        )

        logger.info(f"Final Solution: {final_report}")

        # Generate HTML report
        report_generator = HTMLReportGenerator("reports")
        report_file = report_generator.generate_html_report(filename="team_rag_auditor_report.html")
        logger.info(f"HTML report generated: {report_file}")

        # Save auditor state
        auditor._save_state()

        # Optionally export a LangGraph-GUI friendly workflow representation (dynamic from graph store)
        maybe_export_langgraph_gui(task_description, solution_data, "reports", graph_store)

        # Optionally export a standalone LangGraph Studio format graph
        maybe_export_langgraph_studio(task_description, solution_data, "reports", graph_store)

        logger.info("Demo finished.")

        # Print basic graph summary when using in-memory store
        try:
            if isinstance(graph_store, InMemoryGraphStore):
                nodes_count = len(graph_store._nodes)
                edges_count = len(graph_store._edges)
                logger.info(f"Graph summary (in-memory): nodes={nodes_count}, edges={edges_count}")
                selection_nodes = [n for n in graph_store._nodes.values() if n.label == "Selection"]
                for sn in selection_nodes[:3]:
                    logger.info(f"Selection node: id={sn.id}, props={sn.properties}")
        except Exception as e:
            logger.debug(f"Graph summary skipped: {e}")


def maybe_export_langgraph_gui(task_description: str, solution_data: Dict[str, Any], output_dir: str, graph_store: Any = None) -> None:
    """
    If CAMPFIRES_ENABLE_LANGGRAPH_GUI=1, export a simple workflow graph JSON that
    external viewers (including LangGraph-GUI) can use to visualize the Campfires flow.

    The export is a lightweight representation independent of LangGraph, so
    environments without LangGraph installed can still consume it.
    """
    enable_gui = os.getenv("CAMPFIRES_ENABLE_LANGGRAPH_GUI", "0").strip().lower() in {"1", "true", "yes"}
    if not enable_gui:
        return

    nodes = []
    edges = []
    process_flows = {}
    procedures = {}

    # Optional label filtering and minimal mode
    label_filter_env = os.getenv("CAMPFIRES_EXPORT_LABELS", "").strip()
    minimal_mode = os.getenv("CAMPFIRES_EXPORT_MINIMAL", "0").strip().lower() in {"1", "true", "yes"}

    # If a graph_store is available, export dynamic nodes/edges reflecting actual actions
    candidate_labels = {"Orderer", "Selection", "Camper", "Auditor", "Task", "Stage", "Finding", "Procedure", "ProcedureStep"}
    allowed_labels = {l.strip() for l in label_filter_env.split(",") if l.strip()} or candidate_labels
    try:
        if graph_store is not None and hasattr(graph_store, "_nodes") and hasattr(graph_store, "_edges"):
            # Collect nodes
            for node in getattr(graph_store, "_nodes", {}).values():
                label = getattr(node, "label", None)
                if label in allowed_labels:
                    # Ensure JSON-serializable properties with optional minimal mode
                    props = getattr(node, "properties", {}) or {}
                    serializable_props = {}
                    for k, v in props.items():
                        try:
                            json.dumps(v)
                            serializable_props[k] = v
                        except Exception:
                            serializable_props[k] = str(v)
                    if minimal_mode:
                        # Keep only a few helpful keys per label
                        if label == "Finding":
                            keep = {"summary", "importance"}
                        else:
                            keep = {"name"}
                        serializable_props = {k: serializable_props[k] for k in keep if k in serializable_props}
                    nodes.append({
                        "id": getattr(node, "id", "unknown"),
                        "label": label or "Unknown",
                        "properties": serializable_props,
                    })

            # Index nodes by id for edge validation and stage processing
            node_ids = {n["id"] for n in nodes}

            # Collect edges
            for edge in getattr(graph_store, "_edges", []) or []:
                src = getattr(edge, "src", None)
                dst = getattr(edge, "dst", None)
                etype = getattr(edge, "type", "LINK")
                eprops = getattr(edge, "properties", {}) or {}
                if src in node_ids and dst in node_ids:
                    try:
                        json.dumps(eprops)
                        serializable_eprops = eprops
                    except Exception:
                        serializable_eprops = {k: str(v) for k, v in eprops.items()}
                    if minimal_mode:
                        serializable_eprops = {}
                    edges.append({
                        "source": src,
                        "target": dst,
                        "label": etype,
                        "properties": serializable_eprops,
                    })

            # Build per-stage process flows from findings
            # Map id -> node dict for property access
            node_by_id = {n["id"]: n for n in nodes}
            # Identify stage nodes
            stage_nodes = [n for n in nodes if n["label"] == "Stage"]
            # Map findings for timestamp/sorting
            finding_nodes = {n["id"]: n for n in nodes if n["label"] == "Finding"}

            for stage in stage_nodes:
                stage_name = (stage.get("properties", {}) or {}).get("name", stage["id"]) or stage["id"]
                # Edges from stage to findings
                stage_edges = [e for e in edges if e["source"] == stage["id"] and e["label"] == "HAS_FINDING"]
                steps = []
                for se in stage_edges:
                    fnode = finding_nodes.get(se["target"]) if se["target"] in finding_nodes else None
                    if fnode:
                        props = fnode.get("properties", {}) or {}
                        tags = props.get("tags", []) or []
                        steps.append({
                            "id": fnode["id"],
                            "summary": props.get("summary", ""),
                            "timestamp": props.get("timestamp", ""),
                            "result": next((t for t in tags if t in {"pass", "fail", "partial", "needs_review", "error"}), None),
                            "importance": props.get("importance", 0),
                        })
                # Sort steps by timestamp if present
                try:
                    steps.sort(key=lambda s: s.get("timestamp") or "")
                except Exception:
                    pass
                process_flows[stage_name] = steps
            # Build CAMAS procedures per stage (if present in graph)
            for stage in stage_nodes:
                stage_id = stage["id"]
                stage_name = (stage.get("properties", {}) or {}).get("name", stage_id)
                # Find a procedure linked to the stage
                stage_proc_edges = [e for e in edges if e["source"] == stage_id and e["label"] == "HAS_PROCEDURE"]
                if not stage_proc_edges:
                    continue
                proc_id = stage_proc_edges[0]["target"]
                # Collect steps for this procedure
                step_edges = [e for e in edges if e["source"] == proc_id and e["label"] == "HAS_STEP"]
                # Sort by order_index if provided
                step_edges_sorted = sorted(step_edges, key=lambda e: (e.get("properties", {}) or {}).get("order_index", 0))
                node_by_id = {n["id"]: n for n in nodes}
                step_entries = []
                for se in step_edges_sorted:
                    step_id = se["target"]
                    step_node = node_by_id.get(step_id)
                    if not step_node:
                        continue
                    sprops = step_node.get("properties", {}) or {}
                    # Link finding generated by this step (if any)
                    gen_edges = [e for e in edges if e["source"] == step_id and e["label"] == "GENERATED_FINDING"]
                    finding_id = gen_edges[0]["target"] if gen_edges else None
                    step_entries.append({
                        "id": step_id,
                        "name": sprops.get("name"),
                        "actor": sprops.get("actor"),
                        "start_ts": sprops.get("start_ts"),
                        "status": sprops.get("status"),
                        "importance": sprops.get("importance"),
                        "finding_id": finding_id,
                    })
                procedures[stage_name] = {
                    "procedure_id": proc_id,
                    "steps": step_entries,
                }
        else:
            # Fallback: static skeleton when graph store is unavailable
            nodes = [
                {"id": "start", "label": "Start"},
                {"id": "selection", "label": "Team Selection"},
                {"id": "problem_solving", "label": "Problem Solving"},
                {"id": "design", "label": "Design"},
                {"id": "implementation", "label": "Implementation"},
                {"id": "requirement_analysis", "label": "Requirement Analysis"},
                {"id": "solution_quality", "label": "Solution Quality"},
                {"id": "end", "label": "End"},
            ]
            edges = [
                {"source": "start", "target": "selection", "label": "select_team"},
                {"source": "selection", "target": "problem_solving", "label": "solve"},
                {"source": "problem_solving", "target": "design", "label": "design"},
                {"source": "design", "target": "implementation", "label": "implement"},
                {"source": "implementation", "target": "requirement_analysis", "label": "analyze_requirements"},
                {"source": "requirement_analysis", "target": "solution_quality", "label": "evaluate_quality"},
                {"source": "solution_quality", "target": "end", "label": "complete"},
            ]
    except Exception as e:
        print(f"Dynamic export fallback due to error: {e}")

    metadata = {
        "subject": task_description,
        "solution_summary": solution_data.get("summary"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "process_flows": process_flows,
        "procedures": procedures,
    }

    # Optional: LangGraph Studio-friendly projection
    export_format = os.getenv("CAMPFIRES_LANGGRAPH_EXPORT_FORMAT", "simple").strip().lower()
    studio_graph = None
    if export_format == "studio":
        studio_nodes = [
            {"id": n["id"], "type": n["label"], "data": ({} if minimal_mode else (n.get("properties", {}) or {}))}
            for n in nodes
        ]
        studio_edges = [
            {"source": e["source"], "target": e["target"], "type": e["label"], "data": ({} if minimal_mode else (e.get("properties", {}) or {}))}
            for e in edges
        ]
        studio_graph = {
            "nodes": studio_nodes,
            "edges": studio_edges,
            "start_node": next((n["id"] for n in nodes if n["label"] == "Orderer"), None),
        }

    graph_export = {
        "name": "Campfires Team RAG Workflow",
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }
    if studio_graph is not None:
        graph_export["export_format"] = "studio"
        graph_export["studio"] = studio_graph

    # Write GUI export to disk
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "team_rag_langgraph.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph_export, f, indent=2)
        print(f"LangGraph GUI export written to: {out_path}")
        print("Tip: Load this JSON in custom viewers or adapt for LangGraph Studio.")
    except Exception as e:
        print(f"Failed to write LangGraph GUI export: {e}")


def maybe_export_langgraph_studio(task_description: str, solution_data: Dict[str, Any], output_dir: str, graph_store: Any = None) -> None:
    """
    If CAMPFIRES_ENABLE_LANGGRAPH_STUDIO=1, export a standalone JSON in
    LangGraph Studio-friendly format (nodes/edges with type/data), independent of
    the generic GUI export.
    """
    enable_studio = os.getenv("CAMPFIRES_ENABLE_LANGGRAPH_STUDIO", "0").strip().lower() in {"1", "true", "yes"}
    if not enable_studio:
        return

    nodes: list[dict] = []
    edges: list[dict] = []

    # Optional label filtering and minimal mode
    label_filter_env = os.getenv("CAMPFIRES_EXPORT_LABELS", "").strip()
    minimal_mode = os.getenv("CAMPFIRES_EXPORT_MINIMAL", "0").strip().lower() in {"1", "true", "yes"}

    candidate_labels = {"Orderer", "Selection", "Camper", "Auditor", "Task", "Stage", "Finding", "Procedure", "ProcedureStep"}
    allowed_labels = {l.strip() for l in label_filter_env.split(",") if l.strip()} or candidate_labels
    try:
        if graph_store is not None and hasattr(graph_store, "_nodes") and hasattr(graph_store, "_edges"):
            # Collect nodes
            for node in getattr(graph_store, "_nodes", {}).values():
                label = getattr(node, "label", None)
                if label in allowed_labels:
                    props = getattr(node, "properties", {}) or {}
                    serializable_props = {}
                    for k, v in props.items():
                        try:
                            json.dumps(v)
                            serializable_props[k] = v
                        except Exception:
                            serializable_props[k] = str(v)
                    if minimal_mode:
                        if label == "Finding":
                            keep = {"summary", "importance"}
                        else:
                            keep = {"name"}
                        serializable_props = {k: serializable_props[k] for k in keep if k in serializable_props}
                    nodes.append({
                        "id": getattr(node, "id", "unknown"),
                        "label": label or "Unknown",
                        "properties": serializable_props,
                    })

            node_ids = {n["id"] for n in nodes}

            # Collect edges
            for edge in getattr(graph_store, "_edges", []) or []:
                src = getattr(edge, "src", None)
                dst = getattr(edge, "dst", None)
                etype = getattr(edge, "type", "LINK")
                eprops = getattr(edge, "properties", {}) or {}
                if src in node_ids and dst in node_ids:
                    try:
                        json.dumps(eprops)
                        serializable_eprops = eprops
                    except Exception:
                        serializable_eprops = {k: str(v) for k, v in eprops.items()}
                    if minimal_mode:
                        serializable_eprops = {}
                    edges.append({
                        "source": src,
                        "target": dst,
                        "label": etype,
                        "properties": serializable_eprops,
                    })
        else:
            # Minimal fallback if no graph_store
            nodes = [
                {"id": "orderer:start", "label": "Orderer", "properties": {"name": "Orderer"}},
                {"id": "stage:problem_solving", "label": "Stage", "properties": {"name": "problem_solving"}},
            ]
            edges = [
                {"source": "orderer:start", "target": "stage:problem_solving", "label": "STARTS", "properties": {}},
            ]
    except Exception as e:
        print(f"Studio export fallback due to error: {e}")

    # Project to Studio format
    studio_nodes = [
        {"id": n["id"], "type": n["label"], "data": ({} if minimal_mode else (n.get("properties", {}) or {}))}
        for n in nodes
    ]
    studio_edges = [
        {"source": e["source"], "target": e["target"], "type": e["label"], "data": ({} if minimal_mode else (e.get("properties", {}) or {}))}
        for e in edges
    ]
    start_node = next((n["id"] for n in nodes if n["label"] == "Orderer"), None)

    studio_export = {
        "nodes": studio_nodes,
        "edges": studio_edges,
        "start_node": start_node,
        "metadata": {
            "subject": task_description,
            "solution_summary": solution_data.get("summary"),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "team_rag_langgraph_studio.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(studio_export, f, indent=2)
        print(f"LangGraph Studio export written to: {out_path}")
        print("Tip: Load this JSON directly in LangGraph Studio or compatible viewers.")
    except Exception as e:
        print(f"Failed to write LangGraph Studio export: {e}")
    # Guard GUI export: only trigger via the GUI exporter when explicitly enabled
    enable_gui = os.getenv("CAMPFIRES_ENABLE_LANGGRAPH_GUI", "0").strip().lower() in {"1", "true", "yes"}
    if enable_gui:
        # Delegate to the GUI exporter to avoid scope issues and ensure consistent format
        maybe_export_langgraph_gui(task_description, solution_data, output_dir, graph_store)


if __name__ == "__main__":
    asyncio.run(main())
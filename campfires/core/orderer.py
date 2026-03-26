import asyncio
import logging
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field

from .camper import Camper
from .torch import Torch
from campfires.core.default_auditor import AuditContext, ValidationReport, TaskRequirement, ValidationResult
from .team_auditor import TeamAuditor, TeamAuditorConfig
from ..party_box.box_driver import BoxDriver
from ..mcp.protocol import MCPProtocol
from .graph_store import GraphStore, Node, Edge

logger = logging.getLogger(__name__)


@dataclass
class OrdererConfig:
    team_selection_model: str = "meta-llama/llama-3.2-3b-instruct:free"
    rag_generation_model: str = "meta-llama/llama-3.2-3b-instruct:free"
    default_auditor_config: Dict[str, Any] = field(default_factory=dict)
    team_auditor_config: Dict[str, Any] = field(default_factory=dict)


class Orderer:
    """
    The Orderer orchestrates the team problem-solving RAG process.
    It handles task reception, team selection, camper initialization,
    RAG generation, and role-based RAG optimization.
    """

    def __init__(
        self,
        party_box: BoxDriver,
        mcp_protocol: Optional[MCPProtocol] = None,
        config: Optional[Dict[str, Any]] = None,
        auditor_class: Type[TeamAuditor] = TeamAuditor
    ):
        self.party_box = party_box
        self.mcp_protocol = mcp_protocol
        self.config = config or {}
        self.orderer_config = OrdererConfig(**self.config.get("orderer", {}))
        self.auditor_class = auditor_class
        self.auditor = self.auditor_class(
            party_box=self.party_box,
            config={**self.orderer_config.default_auditor_config, **self.orderer_config.team_auditor_config}
        )
        self.campers: List[Camper] = []
        self.current_task: Optional[str] = None
        # Optional graph store for publishing selection events
        self._graph_store: Optional[GraphStore] = None
        self._graph_enabled: bool = True

    async def receive_task(self, task_description: str) -> None:
        """
        Receives a new task and sets it as the current task for the Orderer.
        
        Args:
            task_description: A detailed description of the task.
        """
        self.current_task = task_description
        print(f"Orderer received task: {task_description}")

    async def select_team(self, available_campers: List[Camper], criteria: Dict[str, Any]) -> List[Camper]:
        """
        Selects a team of campers based on specified criteria.
        
        Args:
            available_campers: A list of all available Camper instances.
            criteria: A dictionary specifying selection criteria (e.g., roles, skills).
            
        Returns:
            A list of selected Camper instances.
        """
        selected_team = []
        # Placeholder for actual team selection logic
        # For now, it just selects all available campers
        selected_team = available_campers
        print(f"Orderer selected a team of {len(selected_team)} campers.")
        # Publish team selection to graph if enabled
        try:
            await self._publish_team_selection(selected_team, criteria)
        except Exception as ge:
            logger.debug(f"Graph publish for team selection skipped/failed: {ge}")
        return selected_team

    async def initialize_campers(self, campers: List[Camper], task_context: Dict[str, Any]) -> None:
        """
        Initializes selected campers with task-specific context and configurations.
        
        Args:
            campers: A list of Camper instances to initialize.
            task_context: Contextual information relevant to the task.
        """
        for camper in campers:
            # Example: setting a unique address for each camper
            camper.set_address(f"camper_{camper.name}_{uuid.uuid4().hex[:8]}")
            # Example: adding task context to in-memory RAG
            camper.add_in_memory_rag("task_context", json.dumps(task_context))
            print(f"Initialized camper {camper.name} with address {camper.get_address()}")

    async def generate_rag(self, source_documents: List[str], query: str, role: Optional[str] = None) -> str:
        """
        Generates RAG (Retrieval Augmented Generation) content based on source documents and a query.
        Optionally filters by role for role-based RAG optimization.
        
        Args:
            source_documents: A list of document paths or content strings.
            query: The query to generate RAG for.
            role: Optional role to filter RAG generation.
            
        Returns:
            The generated RAG content as a string.
        """
        print(f"Orderer generating RAG for query: {query} (Role: {role or 'Any'})")
        # Placeholder for actual RAG generation logic
        # This would typically involve an LLM call with the source documents
        generated_content = f"RAG content for '{query}' (Role: {role or 'Any'}): "
        for doc in source_documents:
            generated_content += f"[From {doc}] "
        return generated_content

    async def optimize_rag(self, rag_content: str, optimization_criteria: Dict[str, Any]) -> str:
        """
        Optimizes generated RAG content based on specified criteria.
        
        Args:
            rag_content: The RAG content to optimize.
            optimization_criteria: Criteria for optimization (e.g., conciseness, relevance).
            
        Returns:
            The optimized RAG content as a string.
        """
        print(f"Orderer optimizing RAG content based on criteria: {optimization_criteria}")
        # Placeholder for actual RAG optimization logic
        # This would typically involve an LLM call to refine the RAG content
        optimized_content = f"Optimized RAG: {rag_content} (optimized for {', '.join(optimization_criteria.keys())})"
        return optimized_content

    async def orchestrate_problem_solving(
        self,
        task_id: str,
        task_description: str,
        requirements: List[Any],
        solution_data: Dict[str, Any]
    ) -> ValidationReport:
        """
        Orchestrates the problem-solving, design, and implementation stages.
        """
        logger.info(f"Orderer orchestrating problem-solving for task: {task_id}")

        # Convert string requirements to TaskRequirement objects
        task_requirements = []
        for i, req_str in enumerate(requirements):
            task_requirements.append(TaskRequirement(id=f"req_{i+1}", description=req_str, priority="medium", validation_criteria=[]))

        audit_context = AuditContext(
            task_id=task_id,
            task_description=task_description,
            requirements=task_requirements,
            solution_data=solution_data
        )

        # Example stages - these would be more dynamic in a real implementation
        stages = ["problem_solving", "design", "implementation"]
        final_report = None
        current_solution_data = solution_data.copy()

        for stage in stages:
            logger.info(f"Orderer entering stage: {stage} for task: {task_id}")
            
            stage_output = {}
            for camper in self.campers:
                if stage == "problem_solving":
                    camper_output = await camper.solve(task_description, requirements, current_solution_data)
                elif stage == "design":
                    camper_output = await camper.design(task_description, requirements, current_solution_data)
                elif stage == "implementation":
                    camper_output = await camper.implement(task_description, requirements, current_solution_data)
                else:
                    camper_output = {}
                stage_output[camper.name] = camper_output
            
            # Aggregate camper outputs into current_solution_data for the next stage and audit
            current_solution_data[stage] = stage_output

            audit_context.solution_data = current_solution_data # Update audit context with latest solution data
            report = await self.auditor.audit_with_challenge(audit_context, stage)
            if report.overall_result != ValidationResult.PASS:
                logger.warning(f"Stage {stage} failed for task {task_id}. Report: {report}")
                # Here, you might trigger re-planning or re-execution by campers
            else:
                logger.info(f"Stage {stage} passed for task {task_id}.")
            final_report = report

        return final_report

    def _save_state(self) -> None:
        """
        Saves the Orderer's state (e.g., current task, selected team) for persistence.
        """
        # Placeholder for persistence logic
        logger.info("Orderer state saved.")

    def _load_state(self) -> None:
        """
        Loads the Orderer's state from a persistent store.
        """
        # Placeholder for loading logic
        logger.info("Orderer state loaded.")

    def set_graph_store(self, graph_store: GraphStore) -> None:
        """
        Inject a GraphStore and pass it to the auditor.
        """
        self._graph_store = graph_store
        self._graph_enabled = True
        try:
            if hasattr(self.auditor, 'set_graph_store'):
                self.auditor.set_graph_store(graph_store)
        except Exception as e:
            logger.debug(f"Failed to set graph store on auditor: {e}")

    async def _publish_team_selection(self, selected_team: List[Camper], criteria: Dict[str, Any]) -> None:
        """
        Publish an Orderer team selection event to the graph store.
        """
        if not self._graph_enabled or self._graph_store is None:
            return
        # Build nodes
        orderer_id = f"orderer:{self.__class__.__name__}"
        selection_id = f"selection:{uuid.uuid4().hex}"
        # Basic rationale based on roles and criteria
        role_counts = {}
        for camper in selected_team:
            role = getattr(camper, 'role', None) or getattr(camper, '_role', None) or (getattr(camper, 'config', {}) or {}).get('role') or 'unknown'
            role_counts[role] = role_counts.get(role, 0) + 1
        team_size = len(selected_team)
        rationale_lines = [
            f"Selected {team_size} campers",
            "Roles: " + ", ".join([f"{r} x{c}" for r, c in role_counts.items()])
        ]
        if criteria:
            rationale_lines.append("Criteria: " + ", ".join([f"{k}={v}" for k, v in criteria.items()]))
        rationale_text = "; ".join(rationale_lines)
        await self._graph_store.upsert_node(Node(id=orderer_id, label="Orderer", properties={"name": self.__class__.__name__}))
        await self._graph_store.upsert_node(Node(id=selection_id, label="Selection", properties={
            "criteria": criteria,
            "rationale": rationale_text,
            "team_size": team_size,
            "roles": role_counts,
        }))
        # Link orderer to selection
        await self._graph_store.upsert_edge(Edge(src=orderer_id, dst=selection_id, type="MADE_SELECTION", properties={}))
        # Add camper nodes and edges
        for camper in selected_team:
            camper_id = camper.get_address() or f"camper:{camper.name}"
            role = getattr(camper, 'role', None) or getattr(camper, '_role', None) or (getattr(camper, 'config', {}) or {}).get('role') or 'unknown'
            await self._graph_store.upsert_node(Node(id=camper_id, label="Camper", properties={"name": camper.name, "role": role}))
            # Link orderer directly to camper creation for clarity
            await self._graph_store.upsert_edge(Edge(
                src=orderer_id,
                dst=camper_id,
                type="CREATED_MEMBER",
                properties={"created_at": datetime.utcnow().isoformat() + "Z"}
            ))
            # Link selection to camper with explicit role assignment
            await self._graph_store.upsert_edge(Edge(
                src=selection_id,
                dst=camper_id,
                type="ASSIGNED_ROLE",
                properties={"role": role, "rationale": rationale_text}
            ))
            # Preserve original selection edge for backward compatibility
            await self._graph_store.upsert_edge(Edge(src=selection_id, dst=camper_id, type="SELECTED_MEMBER", properties={}))
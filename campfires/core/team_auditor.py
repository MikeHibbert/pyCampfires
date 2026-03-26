import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from .default_auditor import DefaultAuditor, ValidationResult, ValidationSeverity, ValidationIssue, ValidationReport, TaskRequirement, AuditContext
from ..party_box.box_driver import BoxDriver

logger = logging.getLogger(__name__)


@dataclass
class TeamAuditorConfig:
    max_retries: int = 3
    retry_delay_seconds: int = 5
    persistence_path: str = "./auditor_state.json"
    stage_rag_map: Dict[str, str] = field(default_factory=dict)


class TeamAuditor(DefaultAuditor):
    """
    TeamAuditor extends DefaultAuditor with challenge mode, stage-specific RAG,
    and self-persistence for team problem-solving scenarios.
    """

    def __init__(
        self,
        party_box: BoxDriver,
        zeitgeist_engine: Any = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(party_box, zeitgeist_engine, config)
        self.team_auditor_config = TeamAuditorConfig(**self.config.get("team_auditor", {}))
        logger.debug(f"TeamAuditorConfig stage_rag_map: {self.team_auditor_config.stage_rag_map}")
        self._load_state()

    async def audit_with_challenge(
        self,
        audit_context: AuditContext,
        stage: str,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None
    ) -> ValidationReport:
        logger.info(f"Auditing stage: {stage} for task: {audit_context.task_id}")
        # logger.info(f"Audit context received in audit_with_challenge: {audit_context}")

        # 1. Analyze Requirements and Solution
        analysis_data = self._analyze_requirements_and_solution(audit_context)

        """
        Performs an audit with a challenge mode, allowing for retries.
        Loads stage-specific RAG documents before auditing.
        """
        max_retries = max_retries or self.team_auditor_config.max_retries
        retry_delay = retry_delay or self.team_auditor_config.retry_delay_seconds

        self._load_stage_rag_document(stage)

        # Retrieve prior findings for this stage to inform context (if graph enabled)
        try:
            if getattr(self, "_graph_store", None):
                stage_id = f"stage:{(stage or 'audit').strip().lower()}"
                findings = await self._graph_store.query_neighbors(stage_id, edge_type="HAS_FINDING")
                ranked: List[Dict[str, Any]] = []
                for f in findings:
                    ctx = await self._graph_store.get_context(f.id)
                    importance = float(f.properties.get('importance') or 0)
                    confidence = float((ctx or {}).get('confidence') or 0)
                    score = 0.6 * importance + 0.4 * confidence
                    ranked.append({"summary": f.properties.get('summary'), "score": score})
                ranked.sort(key=lambda x: x['score'], reverse=True)
                # Append top 3 summaries to historical data for audit context
                for item in ranked[:3]:
                    audit_context.historical_data.append({"summary": item['summary']})
        except Exception as ge:
            logger.debug(f"Graph retrieval skipped/failed for stage {stage}: {ge}")

        current_retries = 0
        while current_retries < max_retries:
            # Annotate execution context so DefaultAuditor can publish
            audit_context.execution_context['stage'] = stage
            audit_context.execution_context['attempt'] = current_retries + 1
            report = await self.audit(audit_context)
            if report.overall_result == ValidationResult.PASS:
                logger.info(f"Audit for stage {stage} passed on attempt {current_retries + 1}.")
                return report
            else:
                logger.warning(
                    f"Audit for stage {stage} failed on attempt {current_retries + 1}. "
                    f"Report: {asdict(report)}"
                )
                if current_retries < max_retries - 1:
                    logger.info(f"Retrying stage {stage} after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    current_retries += 1
                else:
                    logger.error(f"Max retries ({max_retries}) exceeded for stage {stage}.")
                    return report

    def _load_stage_rag_document(self, stage: str) -> None:
        """
        Loads RAG document content specific to the current stage.
        """
        rag_path = self.team_auditor_config.stage_rag_map.get(stage)
        if rag_path:
            logger.info(f"Loading stage-specific RAG for stage '{stage}' from {rag_path}")
            with open(rag_path, 'r') as f:
                rag_content = f.read()
                logger.debug(f"RAG Document for stage '{stage}':\n{rag_content}")
            self.set_rag_document_path(rag_path)
        else:
            logger.info(f"No specific RAG document configured for stage '{stage}'. Using default.")
            # Optionally, clear or reset RAG if no stage-specific one is found
            self.set_rag_document_path(None) # Or set to a default general RAG

    def _load_state(self) -> None:
        """
        Loads the auditor's state from the persistence path.
        """
        persistence_file = Path(self.team_auditor_config.persistence_path)
        if persistence_file.exists():
            try:
                with open(persistence_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    # Restore TeamAuditorConfig attributes
                    self.team_auditor_config.max_retries = state.get("max_retries", self.team_auditor_config.max_retries)
                    self.team_auditor_config.retry_delay_seconds = state.get("retry_delay_seconds", self.team_auditor_config.retry_delay_seconds)
                    self.team_auditor_config.stage_rag_map = state.get("stage_rag_map", self.team_auditor_config.stage_rag_map)
                logger.info(f"Auditor state loaded from {persistence_file}")
            except Exception as e:
                logger.error(f"Failed to load auditor state from {persistence_file}: {e}")
        else:
            logger.info(f"No existing auditor state found at {persistence_file}")

    def _save_state(self) -> None:
        """
        Saves the auditor's state to the persistence path.
        """
        persistence_file = Path(self.team_auditor_config.persistence_path)
        try:
            with open(persistence_file, 'w', encoding='utf-8') as f:
                state = {
                    "max_retries": self.team_auditor_config.max_retries,
                    "retry_delay_seconds": self.team_auditor_config.retry_delay_seconds,
                    "stage_rag_map": self.team_auditor_config.stage_rag_map,
                }
                json.dump(state, f, indent=4)
            logger.info(f"Auditor state saved to {persistence_file}")
        except Exception as e:
            logger.error(f"Failed to save auditor state to {persistence_file}: {e}")
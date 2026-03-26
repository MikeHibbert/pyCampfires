import pytest
from datetime import datetime

from campfires.core.graph_store import InMemoryGraphStore
from campfires.core.orderer import Orderer
from campfires.core.default_auditor import DefaultAuditor, AuditContext, ValidationReport, ValidationResult, TaskRequirement


class _FakeCamper:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def get_address(self):
        return f"camper:{self.name}"


@pytest.mark.asyncio
async def test_orderer_publish_team_selection_edges():
    gs = InMemoryGraphStore()
    orderer = Orderer(party_box=None, config={})
    orderer.set_graph_store(gs)

    team = [_FakeCamper("alice", "designer")]
    criteria = {"skill": "python"}

    await orderer._publish_team_selection(team, criteria)

    edge_types = {e.type for e in gs._edges}
    assert {"MADE_SELECTION", "CREATED_MEMBER", "ASSIGNED_ROLE", "SELECTED_MEMBER"}.issubset(edge_types)

    node_labels = {n.label for n in gs._nodes.values()}
    assert {"Orderer", "Selection", "Camper"}.issubset(node_labels)


@pytest.mark.asyncio
async def test_auditor_share_stage_report_edges():
    gs = InMemoryGraphStore()
    auditor = DefaultAuditor(party_box=None, config={"graph_enabled": True})
    auditor.set_graph_store(gs)

    # Minimal audit context and report
    ctx = AuditContext(
        task_id="t1",
        task_description="Design stage",
        requirements=[TaskRequirement(id="r1", description="d", priority="medium", validation_criteria=[])],
        solution_data={"summary": "s"},
    )
    report = ValidationReport(
        task_id=ctx.task_id,
        task_description=ctx.task_description,
        solution_summary=ctx.solution_data.get("summary", ""),
        overall_result=ValidationResult.PASS,
        confidence_score=0.9,
        validation_timestamp=datetime.now(),
    )

    await auditor._share_stage_report(audit_context=ctx, stage="design", report=report, attempt=1)

    edge_types = {e.type for e in gs._edges}
    assert {"CREATED", "HAS_FINDING", "HAS_PROCEDURE", "HAS_STEP", "GENERATED_FINDING"}.issubset(edge_types)

    node_labels = {n.label for n in gs._nodes.values()}
    assert {"Auditor", "Task", "Stage", "Finding", "Procedure", "ProcedureStep"}.issubset(node_labels)
"""
HTML Report Generator for Sequential Orchestration Demo

This module provides functionality to generate comprehensive HTML reports
that capture the various stages of demo execution, workflow progress,
and detailed insights from the sequential orchestration process.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AIInsight:
    """Represents AI thoughts and perspectives during execution."""
    role_name: str
    camper_id: str
    stage: str
    initial_reaction: str
    perspective_change: str
    key_thoughts: List[str] = field(default_factory=list)
    challenges_faced: List[str] = field(default_factory=list)
    solutions_discovered: List[str] = field(default_factory=list)
    rag_impact: str = ""
    confidence_level: float = 0.0


@dataclass
class ReportStage:
    """Represents a single stage in the demo execution."""
    stage_id: str
    stage_name: str
    description: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    details: Dict[str, Any] = field(default_factory=dict)
    sub_stages: List['ReportStage'] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    ai_insights: List[AIInsight] = field(default_factory=list)
    role_thoughts: Dict[str, str] = field(default_factory=dict)  # role_name -> thoughts
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate stage duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def complete(self, status: str = "completed", **metrics):
        """Mark stage as completed with optional metrics."""
        self.end_time = datetime.now()
        self.status = status
        self.metrics.update(metrics)
    
    def add_error(self, error: str):
        """Add an error to this stage."""
        self.errors.append(error)
        if self.status == "running":
            self.status = "failed"
    
    def add_ai_insight(self, insight: AIInsight):
        """Add AI insight to this stage."""
        self.ai_insights.append(insight)
    
    def add_role_thoughts(self, role_name: str, thoughts: str):
        """Add thoughts from a specific role."""
        self.role_thoughts[role_name] = thoughts


@dataclass
class WorkflowReport:
    """Represents a complete workflow execution report."""
    workflow_id: str
    workflow_name: str
    original_task: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    stages: List[ReportStage] = field(default_factory=list)
    campers_created: List[Dict[str, Any]] = field(default_factory=list)
    rag_states: List[Dict[str, Any]] = field(default_factory=list)
    audit_results: Dict[str, Any] = field(default_factory=dict)
    final_results: Dict[str, Any] = field(default_factory=dict)
    
    # Enhanced AI perspective tracking
    task_understanding: str = ""
    initial_ai_reactions: Dict[str, str] = field(default_factory=dict)  # role -> reaction
    perspective_evolution: Dict[str, List[str]] = field(default_factory=dict)  # role -> changes
    rag_customization_impact: Dict[str, str] = field(default_factory=dict)  # role -> impact
    role_collaboration_insights: List[str] = field(default_factory=list)
    solution_summary: str = ""
    key_discoveries: List[str] = field(default_factory=list)
    challenges_overcome: List[str] = field(default_factory=list)
    ai_confidence_progression: Dict[str, List[float]] = field(default_factory=dict)  # role -> confidence over time
    
    @property
    def total_duration(self) -> Optional[float]:
        """Calculate total workflow duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def add_ai_reaction(self, role_name: str, reaction: str):
        """Add initial AI reaction to the task."""
        self.initial_ai_reactions[role_name] = reaction
    
    def add_perspective_change(self, role_name: str, change: str):
        """Add a perspective change for a role."""
        if role_name not in self.perspective_evolution:
            self.perspective_evolution[role_name] = []
        self.perspective_evolution[role_name].append(change)
    
    def add_rag_impact(self, role_name: str, impact: str):
        """Add RAG customization impact for a role."""
        self.rag_customization_impact[role_name] = impact
    
    def add_confidence_point(self, role_name: str, confidence: float):
        """Add a confidence measurement point for a role."""
        if role_name not in self.ai_confidence_progression:
            self.ai_confidence_progression[role_name] = []
        self.ai_confidence_progression[role_name].append(confidence)


class HTMLReportGenerator:
    """
    Generates comprehensive HTML reports for sequential orchestration demos.
    
    This class captures execution stages, workflow progress, and detailed
    insights, then formats them into a visually appealing HTML report.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize the HTML report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.demo_start_time = datetime.now()
        self.demo_stages: List[ReportStage] = []
        self.workflows: List[WorkflowReport] = []
        self.current_stage: Optional[ReportStage] = None
        self.current_workflow: Optional[WorkflowReport] = None
        
        # Demo-level metrics
        self.demo_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "total_campers_created": 0,
            "total_rag_states": 0,
            "total_tasks_executed": 0
        }
    
    def start_demo_stage(self, stage_name: str, description: str, **details) -> str:
        """
        Start a new demo stage.
        
        Args:
            stage_name: Name of the stage
            description: Description of what this stage does
            **details: Additional stage details
            
        Returns:
            Stage ID for reference
        """
        # Complete previous stage if exists
        if self.current_stage and self.current_stage.status == "running":
            self.current_stage.complete()
        
        stage_id = f"stage_{len(self.demo_stages) + 1}_{uuid.uuid4().hex[:8]}"
        stage = ReportStage(
            stage_id=stage_id,
            stage_name=stage_name,
            description=description,
            start_time=datetime.now(),
            details=details
        )
        
        self.demo_stages.append(stage)
        self.current_stage = stage
        return stage_id
    
    def complete_demo_stage(self, status: str = "completed", **metrics):
        """Complete the current demo stage."""
        if self.current_stage:
            self.current_stage.complete(status, **metrics)
    
    def add_stage_detail(self, key: str, value: Any):
        """Add a detail to the current stage."""
        if self.current_stage:
            self.current_stage.details[key] = value
    
    def add_stage_error(self, error: str):
        """Add an error to the current stage."""
        if self.current_stage:
            self.current_stage.add_error(error)
    
    def start_workflow(self, workflow_name: str, original_task: str) -> str:
        """
        Start tracking a new workflow.
        
        Args:
            workflow_name: Name of the workflow
            original_task: Original task description
            
        Returns:
            Workflow ID for reference
        """
        workflow_id = f"workflow_{len(self.workflows) + 1}_{uuid.uuid4().hex[:8]}"
        workflow = WorkflowReport(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            original_task=original_task,
            start_time=datetime.now()
        )
        
        self.workflows.append(workflow)
        self.current_workflow = workflow
        self.demo_metrics["total_workflows"] += 1
        return workflow_id
    
    def complete_workflow(self, status: str = "completed", **results):
        """Complete the current workflow."""
        if self.current_workflow:
            self.current_workflow.end_time = datetime.now()
            self.current_workflow.status = status
            self.current_workflow.final_results = results
            
            if status == "completed":
                self.demo_metrics["successful_workflows"] += 1
    
    def add_workflow_stage(self, stage_name: str, description: str, **details) -> str:
        """Add a stage to the current workflow."""
        if not self.current_workflow:
            return ""
        
        stage_id = f"wf_stage_{len(self.current_workflow.stages) + 1}_{uuid.uuid4().hex[:8]}"
        stage = ReportStage(
            stage_id=stage_id,
            stage_name=stage_name,
            description=description,
            start_time=datetime.now(),
            details=details
        )
        
        self.current_workflow.stages.append(stage)
        return stage_id
    
    def complete_workflow_stage(self, stage_id: str, status: str = "completed", **metrics):
        """Complete a workflow stage by ID."""
        if not self.current_workflow:
            return
        
        for stage in self.current_workflow.stages:
            if stage.stage_id == stage_id:
                stage.complete(status, **metrics)
                break
    
    def add_camper_created(self, camper_info: Dict[str, Any]):
        """Record a camper creation."""
        if self.current_workflow:
            self.current_workflow.campers_created.append({
                **camper_info,
                "created_at": datetime.now().isoformat()
            })
        self.demo_metrics["total_campers_created"] += 1
    
    def add_rag_state(self, rag_info: Dict[str, Any]):
        """Record a RAG state operation."""
        if self.current_workflow:
            self.current_workflow.rag_states.append({
                **rag_info,
                "timestamp": datetime.now().isoformat()
            })
        self.demo_metrics["total_rag_states"] += 1
    
    def add_audit_results(self, audit_results: Dict[str, Any]):
        """Add audit results to the current workflow."""
        if self.current_workflow:
            self.current_workflow.audit_results.update(audit_results)
    
    # AI Perspective Tracking Methods
    def set_task_understanding(self, understanding: str):
        """Set the AI's understanding of the current task."""
        if self.current_workflow:
            self.current_workflow.task_understanding = understanding
    
    def add_ai_reaction(self, role_name: str, reaction: str):
        """Add initial AI reaction to the task."""
        if self.current_workflow:
            self.current_workflow.add_ai_reaction(role_name, reaction)
    
    def add_perspective_change(self, role_name: str, change: str):
        """Add a perspective change for a role."""
        if self.current_workflow:
            self.current_workflow.add_perspective_change(role_name, change)
    
    def add_rag_impact(self, role_name: str, impact: str):
        """Add RAG customization impact for a role."""
        if self.current_workflow:
            self.current_workflow.add_rag_impact(role_name, impact)
    
    def add_role_collaboration_insight(self, insight: str):
        """Add insight about role collaboration."""
        if self.current_workflow:
            self.current_workflow.role_collaboration_insights.append(insight)
    
    def set_solution_summary(self, summary: str):
        """Set the solution summary for the current workflow."""
        if self.current_workflow:
            self.current_workflow.solution_summary = summary
    
    def add_key_discovery(self, discovery: str):
        """Add a key discovery from the workflow."""
        if self.current_workflow:
            self.current_workflow.key_discoveries.append(discovery)
    
    def add_challenge_overcome(self, challenge: str):
        """Add a challenge that was overcome."""
        if self.current_workflow:
            self.current_workflow.challenges_overcome.append(challenge)
    
    def add_confidence_point(self, role_name: str, confidence: float):
        """Add a confidence measurement point for a role."""
        if self.current_workflow:
            self.current_workflow.add_confidence_point(role_name, confidence)
    
    def add_stage_ai_insight(self, role_name: str, camper_id: str, stage: str, 
                           initial_reaction: str, perspective_change: str,
                           key_thoughts: List[str] = None, challenges_faced: List[str] = None,
                           solutions_discovered: List[str] = None, rag_impact: str = "",
                           confidence_level: float = 0.0):
        """Add detailed AI insight for a specific stage."""
        if self.current_stage:
            insight = AIInsight(
                role_name=role_name,
                camper_id=camper_id,
                stage=stage,
                initial_reaction=initial_reaction,
                perspective_change=perspective_change,
                key_thoughts=key_thoughts or [],
                challenges_faced=challenges_faced or [],
                solutions_discovered=solutions_discovered or [],
                rag_impact=rag_impact,
                confidence_level=confidence_level
            )
            self.current_stage.add_ai_insight(insight)
    
    def add_stage_role_thoughts(self, role_name: str, thoughts: str):
        """Add thoughts from a specific role to the current stage."""
        if self.current_stage:
            self.current_stage.add_role_thoughts(role_name, thoughts)
    
    # Enhanced reporting methods for meeting insights and thought processes
    def add_meeting_insights(self, insights: Dict[str, Any]):
        """Add detailed meeting insights to the current stage."""
        if self.current_stage:
            if not hasattr(self.current_stage, 'meeting_insights'):
                self.current_stage.meeting_insights = {}
            self.current_stage.meeting_insights.update(insights)
    
    def add_thought_process(self, role_name: str, thought_process: Dict[str, Any]):
        """Add detailed thought process for a role to the current stage."""
        if self.current_stage:
            if not hasattr(self.current_stage, 'thought_processes'):
                self.current_stage.thought_processes = {}
            self.current_stage.thought_processes[role_name] = thought_process
    
    def add_planned_outcomes(self, role_name: str, planned_outcomes: Dict[str, Any]):
        """Add planned outcomes for a role to the current stage."""
        if self.current_stage:
            if not hasattr(self.current_stage, 'planned_outcomes'):
                self.current_stage.planned_outcomes = {}
            self.current_stage.planned_outcomes[role_name] = planned_outcomes
    
    def add_decision_making_process(self, role_name: str, decisions: Dict[str, str]):
        """Add decision-making process details for a role."""
        if self.current_stage:
            if not hasattr(self.current_stage, 'decision_processes'):
                self.current_stage.decision_processes = {}
            self.current_stage.decision_processes[role_name] = decisions
    
    def add_collaboration_details(self, collaboration_insights: List[str]):
        """Add detailed collaboration insights to the current stage."""
        if self.current_stage:
            if not hasattr(self.current_stage, 'collaboration_details'):
                self.current_stage.collaboration_details = []
            self.current_stage.collaboration_details.extend(collaboration_insights)
    
    def generate_html_report(self, filename: Optional[str] = None) -> str:
        """
        Generate the complete HTML report.
        
        Args:
            filename: Optional custom filename for the report
            
        Returns:
            Path to the generated HTML file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sequential_orchestration_report_{timestamp}.html"
        
        # Complete any running stages
        if self.current_stage and self.current_stage.status == "running":
            self.current_stage.complete()
        if self.current_workflow and self.current_workflow.status == "running":
            self.complete_workflow()
        
        # Calculate final metrics
        demo_end_time = datetime.now()
        total_duration = (demo_end_time - self.demo_start_time).total_seconds()
        
        # Generate HTML content
        html_content = self._generate_html_content(total_duration)
        
        # Save to file
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_html_content(self, total_duration: float) -> str:
        """Generate the complete HTML content for the report."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sequential Orchestration Demo Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header(total_duration)}
        {self._generate_executive_summary()}
        {self._generate_demo_stages_section()}
        {self._generate_workflows_section()}
        {self._generate_metrics_section()}
        {self._generate_footer()}
    </div>
    
    <script>
        {self._get_javascript()}
    </script>
</body>
</html>
"""
    
    def _get_css_styles(self) -> str:
        """Get CSS styles for the HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            color: #7f8c8d;
            font-size: 1.2em;
        }
        
        .section {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        .stage-card, .workflow-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stage-card:hover, .workflow-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        
        .status-completed {
            background: #d4edda;
            color: #155724;
        }
        
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-running {
            background: #fff3cd;
            color: #856404;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .timeline {
            position: relative;
            padding-left: 30px;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 15px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #3498db;
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 20px;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -23px;
            top: 5px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #3498db;
        }
        
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        
        .collapsible::after {
            content: ' ▼';
            font-size: 0.8em;
            color: #666;
        }
        
        .collapsible.collapsed::after {
            content: ' ▶';
        }
        
        .collapsible-content {
            margin-top: 15px;
            padding-left: 20px;
            border-left: 3px solid #ecf0f1;
        }
        
        .collapsible-content.hidden {
            display: none;
        }
        
        .footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }
        
        .error-list {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }
        
        .error-item {
            color: #721c24;
            margin-bottom: 5px;
        }
        
        .duration {
            color: #666;
            font-size: 0.9em;
            font-style: italic;
        }
        
        .camper-list, .rag-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        
        .camper-item, .rag-item {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 10px;
            font-size: 0.9em;
        }
        
        /* AI Insights Styles */
        .ai-insight-section {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .ai-insight-section h5 {
            color: #007bff;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .insight-content {
            background: white;
            padding: 12px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
            line-height: 1.5;
        }
        
        .role-reaction, .rag-impact, .perspective-evolution {
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 10px;
            margin: 8px 0;
        }
        
        .reactions-list, .impact-list, .evolution-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        /* Enhanced Stage Styles */
        .enhanced-stage {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .stage-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        
        .stage-description {
            color: #6c757d;
            font-style: italic;
            margin-bottom: 10px;
        }
        
        .stage-thoughts {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        
        .stage-thoughts h6 {
            color: #1976d2;
            margin-bottom: 8px;
        }
        
        .role-thought {
            background: white;
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            border: 1px solid #bbdefb;
        }
        
        .stage-ai-insights {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        
        .stage-ai-insights h6 {
            color: #f57c00;
            margin-bottom: 8px;
        }
        
        .ai-insight-detail {
            background: white;
            border: 1px solid #ffcc02;
            border-radius: 8px;
            padding: 12px;
            margin: 10px 0;
        }
        
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        
        .insight-item {
            background: #fafafa;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            font-size: 0.9em;
        }
        
        .insight-item strong {
            color: #333;
            display: block;
        }
        
        /* Enhanced Meeting Insights Styles */
        .meeting-insights-section {
            background: #e8f5e8;
            border-left: 4px solid #4caf50;
            padding: 12px;
            margin: 12px 0;
            border-radius: 6px;
        }
        
        .meeting-insights-section h6 {
            color: #2e7d32;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 12px;
        }
        
        .meeting-insight-item {
            background: white;
            border: 1px solid #c8e6c9;
            border-radius: 6px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Thought Processes Styles */
        .thought-processes-section {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
            padding: 12px;
            margin: 12px 0;
            border-radius: 6px;
        }
        
        .thought-processes-section h6 {
            color: #7b1fa2;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .processes-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .thought-process-item {
            background: white;
            border: 1px solid #e1bee7;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        
        .thought-process-item h7 {
            color: #8e24aa;
            font-weight: bold;
            font-size: 1.1em;
            display: block;
            margin-bottom: 10px;
        }
        
        .process-details {
            background: #fafafa;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            line-height: 1.6;
        }
        
        /* Planned Outcomes Styles */
        .planned-outcomes-section {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 12px;
            margin: 12px 0;
            border-radius: 6px;
        }
        
        .planned-outcomes-section h6 {
            color: #1976d2;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .outcomes-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .planned-outcome-item {
            background: white;
            border: 1px solid #bbdefb;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        
        .planned-outcome-item h7 {
            color: #1565c0;
            font-weight: bold;
            font-size: 1.1em;
            display: block;
            margin-bottom: 10px;
        }
        
        .outcome-details {
            background: #fafafa;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            line-height: 1.6;
        }
        
        /* Decision Processes Styles */
        .decision-processes-section {
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 12px;
            margin: 12px 0;
            border-radius: 6px;
        }
        
        .decision-processes-section h6 {
            color: #f57c00;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .decisions-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .decision-process-item {
            background: white;
            border: 1px solid #ffcc02;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        
        .decision-process-item h7 {
            color: #ef6c00;
            font-weight: bold;
            font-size: 1.1em;
            display: block;
            margin-bottom: 10px;
        }
        
        .decision-details {
            background: #fafafa;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            line-height: 1.6;
        }
        
        /* Collaboration Details Styles */
        .collaboration-details-section {
            background: #fce4ec;
            border-left: 4px solid #e91e63;
            padding: 12px;
            margin: 12px 0;
            border-radius: 6px;
        }
        
        .collaboration-details-section h6 {
            color: #c2185b;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .collaboration-content {
            background: white;
            border: 1px solid #f8bbd9;
            border-radius: 6px;
            padding: 12px;
            line-height: 1.6;
        }
            margin-bottom: 4px;
        }
        
        /* Enhanced Workflow Styles */
        .enhanced-workflow {
            border: 2px solid #e9ecef;
            background: linear-gradient(to right, #f8f9fa, #ffffff);
        }
        
        .enhanced-timeline {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
        }
        
        .solution-summary {
            background: #e8f5e8;
            border-left: 4px solid #28a745;
        }
        
        .solution-summary h5 {
            color: #155724;
        }
        
        .discoveries-challenges {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .discoveries, .challenges {
            background: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
        
        .discoveries {
            border-left: 4px solid #28a745;
        }
        
        .challenges {
            border-left: 4px solid #dc3545;
        }
        
        .technical-details {
            background: #f1f3f4;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .technical-details h5 {
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 5px;
            margin-bottom: 10px;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .insight-grid {
                grid-template-columns: 1fr;
            }
            
            .discoveries-challenges {
                grid-template-columns: 1fr;
            }
            
            .stage-header {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        """
    
    def _generate_header(self, total_duration: float) -> str:
        """Generate the report header."""
        return f"""
        <div class="header">
            <h1>🔥 Sequential Orchestration Demo Report</h1>
            <div class="subtitle">
                Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
            </div>
            <div class="subtitle">
                Total Execution Time: {total_duration:.2f} seconds
            </div>
        </div>
        """
    
    def _generate_executive_summary(self) -> str:
        """Generate the executive summary section."""
        success_rate = (self.demo_metrics["successful_workflows"] / max(1, self.demo_metrics["total_workflows"])) * 100
        
        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{self.demo_metrics['total_workflows']}</div>
                    <div class="metric-label">Total Workflows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{success_rate:.1f}%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.demo_metrics['total_campers_created']}</div>
                    <div class="metric-label">Campers Created</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{self.demo_metrics['total_rag_states']}</div>
                    <div class="metric-label">RAG States</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_demo_stages_section(self) -> str:
        """Generate the demo stages section."""
        stages_html = ""
        for i, stage in enumerate(self.demo_stages, 1):
            duration_text = f" ({stage.duration:.2f}s)" if stage.duration else ""
            
            errors_html = ""
            if stage.errors:
                errors_html = f"""
                <div class="error-list">
                    <strong>Errors:</strong>
                    {''.join(f'<div class="error-item">• {error}</div>' for error in stage.errors)}
                </div>
                """
            
            details_html = ""
            if stage.details:
                details_items = []
                for key, value in stage.details.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value, indent=2)
                    details_items.append(f"<strong>{key}:</strong> {value}")
                details_html = f"""
                <div class="collapsible-content">
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        {'<br>'.join(details_items)}
                    </div>
                </div>
                """
            
            stages_html += f"""
            <div class="timeline-item">
                <div class="stage-card">
                    <h4 class="collapsible" onclick="toggleCollapsible(this)">
                        {i}. {stage.stage_name}
                        <span class="status-badge status-{stage.status}">{stage.status}</span>
                        <span class="duration">{duration_text}</span>
                    </h4>
                    <p>{stage.description}</p>
                    {errors_html}
                    {details_html}
                </div>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>🎯 Demo Execution Stages</h2>
            <div class="timeline">
                {stages_html}
            </div>
        </div>
        """
    
    def _generate_workflows_section(self) -> str:
        """Generate the workflows section with comprehensive AI perspectives."""
        workflows_html = ""
        
        for i, workflow in enumerate(self.workflows, 1):
            duration_text = f" ({workflow.total_duration:.2f}s)" if workflow.total_duration else ""
            
            # Generate task understanding section
            task_understanding_html = ""
            if workflow.task_understanding:
                task_understanding_html = f"""
                <div class="ai-insight-section">
                    <h5>🎯 Task Understanding</h5>
                    <div class="insight-content">
                        {workflow.task_understanding}
                    </div>
                </div>
                """
            
            # Generate initial AI reactions
            reactions_html = ""
            if workflow.initial_ai_reactions:
                reaction_items = []
                for role, reaction in workflow.initial_ai_reactions.items():
                    reaction_items.append(f"""
                    <div class="role-reaction">
                        <strong>🤖 {role}:</strong> {reaction}
                    </div>
                    """)
                reactions_html = f"""
                <div class="ai-insight-section">
                    <h5>💭 Initial AI Reactions</h5>
                    <div class="reactions-list">
                        {''.join(reaction_items)}
                    </div>
                </div>
                """
            
            # Generate RAG customization impact
            rag_impact_html = ""
            if workflow.rag_customization_impact:
                impact_items = []
                for role, impact in workflow.rag_customization_impact.items():
                    impact_items.append(f"""
                    <div class="rag-impact">
                        <strong>🧠 {role}:</strong> {impact}
                    </div>
                    """)
                rag_impact_html = f"""
                <div class="ai-insight-section">
                    <h5>🔧 RAG Customization Impact</h5>
                    <div class="impact-list">
                        {''.join(impact_items)}
                    </div>
                </div>
                """
            
            # Generate workflow stages with AI insights
            workflow_stages_html = ""
            for stage in workflow.stages:
                stage_duration = f" ({stage.duration:.2f}s)" if stage.duration else ""
                
                # Generate role thoughts for this stage
                role_thoughts_html = ""
                if stage.role_thoughts:
                    thoughts_items = []
                    for role, thoughts in stage.role_thoughts.items():
                        thoughts_items.append(f"""
                        <div class="role-thought">
                            <strong>💭 {role}:</strong> {thoughts}
                        </div>
                        """)
                    role_thoughts_html = f"""
                    <div class="stage-thoughts">
                        <h6>Role Thoughts:</h6>
                        {''.join(thoughts_items)}
                    </div>
                    """
                
                # Generate AI insights for this stage
                ai_insights_html = ""
                if stage.ai_insights:
                    insights_items = []
                    for insight in stage.ai_insights:
                        key_thoughts = '<br>'.join([f"• {thought}" for thought in insight.key_thoughts])
                        challenges = '<br>'.join([f"• {challenge}" for challenge in insight.challenges_faced])
                        solutions = '<br>'.join([f"• {solution}" for solution in insight.solutions_discovered])
                        
                        insights_items.append(f"""
                        <div class="ai-insight-detail">
                            <strong>🤖 {insight.role_name} ({insight.camper_id})</strong>
                            <div class="insight-grid">
                                <div class="insight-item">
                                    <strong>Initial Reaction:</strong> {insight.initial_reaction}
                                </div>
                                <div class="insight-item">
                                    <strong>Perspective Change:</strong> {insight.perspective_change}
                                </div>
                                {f'<div class="insight-item"><strong>Key Thoughts:</strong><br>{key_thoughts}</div>' if key_thoughts else ''}
                                {f'<div class="insight-item"><strong>Challenges:</strong><br>{challenges}</div>' if challenges else ''}
                                {f'<div class="insight-item"><strong>Solutions:</strong><br>{solutions}</div>' if solutions else ''}
                                {f'<div class="insight-item"><strong>RAG Impact:</strong> {insight.rag_impact}</div>' if insight.rag_impact else ''}
                                <div class="insight-item">
                                    <strong>Confidence:</strong> {insight.confidence_level:.1f}/10
                                </div>
                            </div>
                        </div>
                        """)
                    ai_insights_html = f"""
                    <div class="stage-ai-insights">
                        <h6>AI Insights:</h6>
                        {''.join(insights_items)}
                    </div>
                    """
                
                # Generate meeting insights for this stage
                meeting_insights_html = ""
                if hasattr(stage, 'meeting_insights') and stage.meeting_insights:
                    insights_items = []
                    for key, value in stage.meeting_insights.items():
                        if isinstance(value, list):
                            value_str = '<br>'.join([f"• {item}" for item in value])
                        else:
                            value_str = str(value)
                        insights_items.append(f"""
                        <div class="meeting-insight-item">
                            <strong>🎯 {key.replace('_', ' ').title()}:</strong><br>{value_str}
                        </div>
                        """)
                    meeting_insights_html = f"""
                    <div class="meeting-insights-section">
                        <h6>📋 Meeting Insights:</h6>
                        <div class="insights-grid">
                            {''.join(insights_items)}
                        </div>
                    </div>
                    """
                
                # Generate thought processes for this stage
                thought_processes_html = ""
                if hasattr(stage, 'thought_processes') and stage.thought_processes:
                    process_items = []
                    for role, process in stage.thought_processes.items():
                        process_details = []
                        for key, value in process.items():
                            if isinstance(value, list):
                                value_str = '<br>'.join([f"• {item}" for item in value])
                            else:
                                value_str = str(value)
                            process_details.append(f"<strong>{key.replace('_', ' ').title()}:</strong> {value_str}")
                        
                        process_items.append(f"""
                        <div class="thought-process-item">
                            <h7>🧠 {role} Thought Process:</h7>
                            <div class="process-details">
                                {'<br><br>'.join(process_details)}
                            </div>
                        </div>
                        """)
                    thought_processes_html = f"""
                    <div class="thought-processes-section">
                        <h6>💭 Detailed Thought Processes:</h6>
                        <div class="processes-container">
                            {''.join(process_items)}
                        </div>
                    </div>
                    """
                
                # Generate planned outcomes for this stage
                planned_outcomes_html = ""
                if hasattr(stage, 'planned_outcomes') and stage.planned_outcomes:
                    outcome_items = []
                    for role, outcomes in stage.planned_outcomes.items():
                        outcome_details = []
                        for key, value in outcomes.items():
                            if isinstance(value, list):
                                value_str = '<br>'.join([f"• {item}" for item in value])
                            else:
                                value_str = str(value)
                            outcome_details.append(f"<strong>{key.replace('_', ' ').title()}:</strong> {value_str}")
                        
                        outcome_items.append(f"""
                        <div class="planned-outcome-item">
                            <h7>🎯 {role} Planned Outcomes:</h7>
                            <div class="outcome-details">
                                {'<br><br>'.join(outcome_details)}
                            </div>
                        </div>
                        """)
                    planned_outcomes_html = f"""
                    <div class="planned-outcomes-section">
                        <h6>🚀 Planned Outcomes:</h6>
                        <div class="outcomes-container">
                            {''.join(outcome_items)}
                        </div>
                    </div>
                    """
                
                # Generate decision-making processes for this stage
                decision_processes_html = ""
                if hasattr(stage, 'decision_processes') and stage.decision_processes:
                    decision_items = []
                    for role, decisions in stage.decision_processes.items():
                        decision_details = []
                        for key, value in decisions.items():
                            decision_details.append(f"<strong>{key.replace('_', ' ').title()}:</strong> {value}")
                        
                        decision_items.append(f"""
                        <div class="decision-process-item">
                            <h7>⚖️ {role} Decision Process:</h7>
                            <div class="decision-details">
                                {'<br><br>'.join(decision_details)}
                            </div>
                        </div>
                        """)
                    decision_processes_html = f"""
                    <div class="decision-processes-section">
                        <h6>⚖️ Decision-Making Processes:</h6>
                        <div class="decisions-container">
                            {''.join(decision_items)}
                        </div>
                    </div>
                    """
                
                # Generate collaboration details for this stage
                collaboration_details_html = ""
                if hasattr(stage, 'collaboration_details') and stage.collaboration_details:
                    collaboration_list = '<br>'.join([f"• {detail}" for detail in stage.collaboration_details])
                    collaboration_details_html = f"""
                    <div class="collaboration-details-section">
                        <h6>🤝 Collaboration Details:</h6>
                        <div class="collaboration-content">
                            {collaboration_list}
                        </div>
                    </div>
                    """
                
                workflow_stages_html += f"""
                <div class="timeline-item enhanced-stage">
                    <div class="stage-header">
                        <strong>{stage.stage_name}</strong>
                        <span class="status-badge status-{stage.status}">{stage.status}</span>
                        <span class="duration">{stage_duration}</span>
                    </div>
                    <div class="stage-description">{stage.description}</div>
                    {role_thoughts_html}
                    {ai_insights_html}
                    {meeting_insights_html}
                    {thought_processes_html}
                    {planned_outcomes_html}
                    {decision_processes_html}
                    {collaboration_details_html}
                </div>
                """
            
            # Generate perspective evolution
            perspective_evolution_html = ""
            if workflow.perspective_evolution:
                evolution_items = []
                for role, changes in workflow.perspective_evolution.items():
                    changes_list = '<br>'.join([f"• {change}" for change in changes])
                    evolution_items.append(f"""
                    <div class="perspective-evolution">
                        <strong>🔄 {role}:</strong><br>{changes_list}
                    </div>
                    """)
                perspective_evolution_html = f"""
                <div class="ai-insight-section">
                    <h5>🔄 Perspective Evolution</h5>
                    <div class="evolution-list">
                        {''.join(evolution_items)}
                    </div>
                </div>
                """
            
            # Generate collaboration insights
            collaboration_html = ""
            if workflow.role_collaboration_insights:
                insights_list = '<br>'.join([f"• {insight}" for insight in workflow.role_collaboration_insights])
                collaboration_html = f"""
                <div class="ai-insight-section">
                    <h5>🤝 Role Collaboration Insights</h5>
                    <div class="insight-content">
                        {insights_list}
                    </div>
                </div>
                """
            
            # Generate key discoveries and challenges
            discoveries_challenges_html = ""
            if workflow.key_discoveries or workflow.challenges_overcome:
                discoveries_list = '<br>'.join([f"✅ {discovery}" for discovery in workflow.key_discoveries])
                challenges_list = '<br>'.join([f"💪 {challenge}" for challenge in workflow.challenges_overcome])
                
                discoveries_challenges_html = f"""
                <div class="ai-insight-section">
                    <h5>🔍 Key Discoveries & Challenges Overcome</h5>
                    <div class="discoveries-challenges">
                        {f'<div class="discoveries"><strong>Discoveries:</strong><br>{discoveries_list}</div>' if discoveries_list else ''}
                        {f'<div class="challenges"><strong>Challenges Overcome:</strong><br>{challenges_list}</div>' if challenges_list else ''}
                    </div>
                </div>
                """
            
            # Generate solution summary
            solution_summary_html = ""
            if workflow.solution_summary:
                solution_summary_html = f"""
                <div class="ai-insight-section solution-summary">
                    <h5>🎯 Solution Summary</h5>
                    <div class="insight-content">
                        {workflow.solution_summary}
                    </div>
                </div>
                """
            
            # Generate campers list
            campers_html = ""
            if workflow.campers_created:
                campers_items = []
                for camper in workflow.campers_created:
                    campers_items.append(f"""
                    <div class="camper-item">
                        <strong>{camper.get('name', 'Unknown')}</strong><br>
                        Role: {camper.get('role', 'N/A')}<br>
                        <small>Created: {camper.get('created_at', 'N/A')}</small>
                    </div>
                    """)
                campers_html = f"""
                <h5>👥 Campers Created:</h5>
                <div class="camper-list">
                    {''.join(campers_items)}
                </div>
                """
            
            # Generate RAG states list
            rag_states_html = ""
            if workflow.rag_states:
                rag_items = []
                for rag in workflow.rag_states:
                    rag_items.append(f"""
                    <div class="rag-item">
                        <strong>{rag.get('operation', 'Unknown')}</strong><br>
                        Camper: {rag.get('camper_id', 'N/A')}<br>
                        <small>{rag.get('timestamp', 'N/A')}</small>
                    </div>
                    """)
                rag_states_html = f"""
                <h5>🧠 RAG Operations:</h5>
                <div class="rag-list">
                    {''.join(rag_items)}
                </div>
                """
            
            workflows_html += f"""
            <div class="workflow-card enhanced-workflow">
                <h3 class="collapsible" onclick="toggleCollapsible(this)">
                    Workflow {i}: {workflow.workflow_name}
                    <span class="status-badge status-{workflow.status}">{workflow.status}</span>
                    <span class="duration">{duration_text}</span>
                </h3>
                <p><strong>Original Task:</strong> {workflow.original_task}</p>
                
                <div class="collapsible-content">
                    {task_understanding_html}
                    {reactions_html}
                    {rag_impact_html}
                    
                    <h4>📋 Execution Stages with AI Insights:</h4>
                    <div class="timeline enhanced-timeline">
                        {workflow_stages_html}
                    </div>
                    
                    {perspective_evolution_html}
                    {collaboration_html}
                    {discoveries_challenges_html}
                    {solution_summary_html}
                    
                    <div class="technical-details">
                        {campers_html}
                        {rag_states_html}
                        {self._format_audit_results(workflow.audit_results)}
                    </div>
                </div>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>🔄 Workflow Executions with AI Perspectives</h2>
            {workflows_html}
        </div>
        """
    
    def _format_audit_results(self, audit_results: Dict[str, Any]) -> str:
        """Format audit results for display."""
        if not audit_results:
            return ""
        
        return f"""
        <h5>🔍 Audit Results:</h5>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
            <pre>{json.dumps(audit_results, indent=2)}</pre>
        </div>
        """
    
    def _generate_metrics_section(self) -> str:
        """Generate the detailed metrics section."""
        return f"""
        <div class="section">
            <h2>📈 Detailed Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(self.demo_stages)}</div>
                    <div class="metric-label">Demo Stages</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{sum(len(w.stages) for w in self.workflows)}</div>
                    <div class="metric-label">Workflow Stages</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{sum(len(w.campers_created) for w in self.workflows)}</div>
                    <div class="metric-label">Total Campers</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{sum(len(w.rag_states) for w in self.workflows)}</div>
                    <div class="metric-label">RAG Operations</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate the report footer."""
        return f"""
        <div class="footer">
            <p>Generated by Campfires Sequential Orchestration Demo</p>
            <p>Report ID: {uuid.uuid4().hex[:8]} | Generated at {datetime.now().isoformat()}</p>
        </div>
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features."""
        return """
        function toggleCollapsible(element) {
            var content = element.nextElementSibling;
            while (content && !content.classList.contains('collapsible-content')) {
                content = content.nextElementSibling;
            }
            
            if (content) {
                content.classList.toggle('hidden');
                element.classList.toggle('collapsed');
            }
        }
        
        // Initialize collapsed state for some sections
        document.addEventListener('DOMContentLoaded', function() {
            var collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(function(collapsible, index) {
                // Collapse workflow details by default
                if (collapsible.textContent.includes('Workflow')) {
                    toggleCollapsible(collapsible);
                }
            });
        });
        """
"""
Role-aware orchestration system for intelligent task decomposition and dynamic role assignment.

This module provides the core infrastructure for analyzing complex tasks, breaking them down
into manageable subtasks, and dynamically generating appropriate roles and campers to handle
each component.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .torch import Torch
from .camper import Camper
from ..mcp.protocol import MCPProtocol
from ..core.openrouter import LLMCamperMixin, OpenRouterConfig


logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Enumeration of task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class SubTask:
    """Represents a decomposed subtask with role requirements."""
    id: str
    description: str
    required_role: str
    dependencies: List[str]
    priority: int
    estimated_complexity: TaskComplexity
    context_requirements: List[str]
    success_criteria: str


@dataclass
class RoleRequirement:
    """Defines requirements for a specific role."""
    role_name: str
    expertise_areas: List[str]
    required_capabilities: List[str]
    personality_traits: List[str]
    context_sources: List[str]


class TaskDecomposer(LLMCamperMixin):
    """
    Analyzes complex tasks and breaks them down into manageable subtasks.
    
    Uses LLM capabilities to understand task requirements and identify
    optimal decomposition strategies.
    """
    
    def __init__(self, config: Dict[str, Any], mcp_protocol: Optional[MCPProtocol] = None):
        """
        Initialize the task decomposer.
        
        Args:
            config: Configuration dictionary
            mcp_protocol: MCP protocol for communication
        """
        self.config = config
        self.mcp_protocol = mcp_protocol
        
        # Setup LLM capabilities
        openrouter_config = OpenRouterConfig(
            api_key=config.get('openrouter_api_key', '')
        )
        self.setup_llm(openrouter_config, mcp_protocol)
        
        # Task analysis templates
        self.decomposition_prompt_template = """
        You are an expert task analyst. Analyze the following task and break it down into manageable subtasks.
        
        TASK: {task_description}
        CONTEXT: {context}
        CONSTRAINTS: {constraints}
        
        For each subtask, provide:
        1. Clear description
        2. Required role/expertise
        3. Dependencies on other subtasks
        4. Priority level (1-10)
        5. Complexity estimate
        6. Success criteria
        
        Format your response as a structured analysis that can guide role assignment and execution planning.
        """
    
    async def analyze_task(self, torch: Torch) -> Dict[str, Any]:
        """
        Analyze a task torch and determine its complexity and requirements.
        
        Args:
            torch: The task torch to analyze
            
        Returns:
            Dictionary containing task analysis results
        """
        try:
            analysis_prompt = f"""
            Analyze this task for complexity and requirements:
            
            Task: {torch.claim}
            Metadata: {torch.metadata}
            
            Determine:
            1. Overall complexity level (simple/moderate/complex/highly_complex)
            2. Required expertise areas
            3. Potential challenges
            4. Resource requirements
            5. Estimated time/effort
            
            Provide a structured analysis.
            """
            
            analysis_result = await self.llm_completion_with_mcp(analysis_prompt)
            
            return {
                "analysis": analysis_result,
                "complexity": self._extract_complexity(analysis_result),
                "expertise_areas": self._extract_expertise_areas(analysis_result),
                "challenges": self._extract_challenges(analysis_result)
            }
            
        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
            return {
                "analysis": "Analysis failed",
                "complexity": TaskComplexity.MODERATE,
                "expertise_areas": ["general"],
                "challenges": ["analysis_error"]
            }
    
    async def decompose_task(self, torch: Torch, max_subtasks: int = 10) -> List[SubTask]:
        """
        Decompose a complex task into manageable subtasks.
        
        Args:
            torch: The task torch to decompose
            max_subtasks: Maximum number of subtasks to create
            
        Returns:
            List of SubTask objects
        """
        try:
            # First analyze the task
            analysis = await self.analyze_task(torch)
            
            # Generate decomposition prompt
            decomposition_prompt = self.decomposition_prompt_template.format(
                task_description=torch.claim,
                context=torch.metadata.get('context', 'No additional context'),
                constraints=torch.metadata.get('constraints', 'No specific constraints')
            )
            
            decomposition_result = await self.llm_completion_with_mcp(decomposition_prompt)
            
            # Parse the decomposition result into SubTask objects
            subtasks = self._parse_decomposition_result(decomposition_result, max_subtasks)
            
            logger.info(f"Decomposed task into {len(subtasks)} subtasks")
            return subtasks
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            # Return a single subtask as fallback
            return [SubTask(
                id="fallback_task",
                description=torch.claim,
                required_role="general",
                dependencies=[],
                priority=5,
                estimated_complexity=TaskComplexity.MODERATE,
                context_requirements=[],
                success_criteria="Complete the original task"
            )]
    
    def _extract_complexity(self, analysis: str) -> TaskComplexity:
        """Extract complexity level from analysis text."""
        analysis_lower = analysis.lower()
        if "highly_complex" in analysis_lower or "highly complex" in analysis_lower:
            return TaskComplexity.HIGHLY_COMPLEX
        elif "complex" in analysis_lower:
            return TaskComplexity.COMPLEX
        elif "moderate" in analysis_lower:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _extract_expertise_areas(self, analysis: str) -> List[str]:
        """Extract required expertise areas from analysis text."""
        # Simple keyword extraction - could be enhanced with NLP
        expertise_keywords = [
            "technical", "creative", "analytical", "communication", "leadership",
            "research", "design", "development", "testing", "documentation"
        ]
        
        found_areas = []
        analysis_lower = analysis.lower()
        for keyword in expertise_keywords:
            if keyword in analysis_lower:
                found_areas.append(keyword)
        
        return found_areas if found_areas else ["general"]
    
    def _extract_challenges(self, analysis: str) -> List[str]:
        """Extract potential challenges from analysis text."""
        # Simple extraction - could be enhanced
        return ["complexity_management", "coordination", "quality_assurance"]
    
    def _parse_decomposition_result(self, result: str, max_subtasks: int) -> List[SubTask]:
        """Parse LLM decomposition result into SubTask objects."""
        # This is a simplified parser - in production, you might want more sophisticated parsing
        subtasks = []
        
        # For now, create a basic structure based on the result
        # This would be enhanced with proper parsing logic
        lines = result.split('\n')
        current_subtask = None
        subtask_count = 0
        
        for line in lines:
            line = line.strip()
            if not line or subtask_count >= max_subtasks:
                continue
                
            # Look for subtask indicators
            if any(indicator in line.lower() for indicator in ['subtask', 'task', 'step']):
                if current_subtask:
                    subtasks.append(current_subtask)
                
                subtask_count += 1
                current_subtask = SubTask(
                    id=f"subtask_{subtask_count}",
                    description=line,
                    required_role="general",
                    dependencies=[],
                    priority=5,
                    estimated_complexity=TaskComplexity.MODERATE,
                    context_requirements=[],
                    success_criteria="Complete subtask successfully"
                )
        
        # Add the last subtask
        if current_subtask:
            subtasks.append(current_subtask)
        
        # If no subtasks were parsed, create a default one
        if not subtasks:
            subtasks.append(SubTask(
                id="default_subtask",
                description="Process the main task",
                required_role="general",
                dependencies=[],
                priority=5,
                estimated_complexity=TaskComplexity.MODERATE,
                context_requirements=[],
                success_criteria="Complete the task successfully"
            ))
        
        return subtasks


class DynamicRoleGenerator(LLMCamperMixin):
    """
    Generates role requirements and specifications based on task analysis.
    
    Creates detailed role definitions that can be used to instantiate
    appropriate campers for specific subtasks.
    """
    
    def __init__(self, config: Dict[str, Any], mcp_protocol: Optional[MCPProtocol] = None):
        """
        Initialize the dynamic role generator.
        
        Args:
            config: Configuration dictionary
            mcp_protocol: MCP protocol for communication
        """
        self.config = config
        self.mcp_protocol = mcp_protocol
        
        # Setup LLM capabilities
        openrouter_config = OpenRouterConfig(
            api_key=config.get('openrouter_api_key', '')
        )
        self.setup_llm(openrouter_config, mcp_protocol)
        
        # Role generation templates
        self.role_generation_prompt = """
        Based on the following subtask, generate a detailed role specification:
        
        SUBTASK: {subtask_description}
        REQUIRED_ROLE: {required_role}
        CONTEXT: {context}
        
        Generate a role specification including:
        1. Role name and title
        2. Key expertise areas
        3. Required capabilities
        4. Personality traits that would be effective
        5. Context sources that would be helpful
        6. Success metrics for this role
        
        Make the role specific enough to be actionable but flexible enough to handle variations.
        """
    
    async def generate_role_requirement(self, subtask: SubTask) -> RoleRequirement:
        """
        Generate a detailed role requirement for a specific subtask.
        
        Args:
            subtask: The subtask requiring a role
            
        Returns:
            RoleRequirement object with detailed specifications
        """
        try:
            role_prompt = self.role_generation_prompt.format(
                subtask_description=subtask.description,
                required_role=subtask.required_role,
                context=subtask.context_requirements
            )
            
            role_spec = await self.llm_completion_with_mcp(role_prompt)
            
            # Parse the role specification
            role_requirement = self._parse_role_specification(role_spec, subtask)
            
            logger.info(f"Generated role requirement: {role_requirement.role_name}")
            return role_requirement
            
        except Exception as e:
            logger.error(f"Role generation failed: {e}")
            # Return a basic role requirement as fallback
            return RoleRequirement(
                role_name=subtask.required_role,
                expertise_areas=["general"],
                required_capabilities=["task_processing"],
                personality_traits=["reliable", "thorough"],
                context_sources=[]
            )
    
    async def generate_multiple_roles(self, subtasks: List[SubTask]) -> Dict[str, RoleRequirement]:
        """
        Generate role requirements for multiple subtasks.
        
        Args:
            subtasks: List of subtasks requiring roles
            
        Returns:
            Dictionary mapping subtask IDs to role requirements
        """
        role_requirements = {}
        
        for subtask in subtasks:
            role_req = await self.generate_role_requirement(subtask)
            role_requirements[subtask.id] = role_req
        
        return role_requirements
    
    def _parse_role_specification(self, spec: str, subtask: SubTask) -> RoleRequirement:
        """Parse LLM-generated role specification into RoleRequirement object."""
        # Simplified parsing - would be enhanced in production
        return RoleRequirement(
            role_name=subtask.required_role,
            expertise_areas=self._extract_expertise_from_spec(spec),
            required_capabilities=self._extract_capabilities_from_spec(spec),
            personality_traits=self._extract_traits_from_spec(spec),
            context_sources=subtask.context_requirements
        )
    
    def _extract_expertise_from_spec(self, spec: str) -> List[str]:
        """Extract expertise areas from role specification."""
        # Simple keyword extraction
        expertise_keywords = [
            "analysis", "research", "design", "development", "testing",
            "communication", "leadership", "creativity", "problem-solving"
        ]
        
        found_expertise = []
        spec_lower = spec.lower()
        for keyword in expertise_keywords:
            if keyword in spec_lower:
                found_expertise.append(keyword)
        
        return found_expertise if found_expertise else ["general"]
    
    def _extract_capabilities_from_spec(self, spec: str) -> List[str]:
        """Extract required capabilities from role specification."""
        return ["task_processing", "communication", "analysis"]
    
    def _extract_traits_from_spec(self, spec: str) -> List[str]:
        """Extract personality traits from role specification."""
        return ["reliable", "thorough", "collaborative"]


class RoleAwareOrchestrator:
    """
    Main orchestrator that combines task decomposition and role generation
    to create intelligent, role-aware execution plans.
    """
    
    def __init__(self, config: Dict[str, Any], mcp_protocol: Optional[MCPProtocol] = None):
        """
        Initialize the role-aware orchestrator.
        
        Args:
            config: Configuration dictionary
            mcp_protocol: MCP protocol for communication
        """
        self.config = config
        self.mcp_protocol = mcp_protocol
        
        # Initialize components
        self.task_decomposer = TaskDecomposer(config, mcp_protocol)
        self.role_generator = DynamicRoleGenerator(config, mcp_protocol)
        
        # Orchestration state
        self.active_orchestrations: Dict[str, Dict[str, Any]] = {}
    
    async def orchestrate_task(self, torch: Torch) -> Dict[str, Any]:
        """
        Orchestrate a complex task through role-aware decomposition.
        
        Args:
            torch: The task torch to orchestrate
            
        Returns:
            Dictionary containing orchestration plan and metadata
        """
        orchestration_id = f"orch_{torch.id}"
        
        try:
            logger.info(f"Starting role-aware orchestration for task: {torch.claim}")
            
            # Step 1: Decompose the task
            subtasks = await self.task_decomposer.decompose_task(torch)
            
            # Step 2: Generate role requirements
            role_requirements = await self.role_generator.generate_multiple_roles(subtasks)
            
            # Step 3: Create orchestration plan
            orchestration_plan = {
                "id": orchestration_id,
                "original_task": torch.claim,
                "subtasks": subtasks,
                "role_requirements": role_requirements,
                "execution_order": self._determine_execution_order(subtasks),
                "resource_requirements": self._calculate_resource_requirements(subtasks),
                "estimated_duration": self._estimate_duration(subtasks)
            }
            
            # Store the orchestration
            self.active_orchestrations[orchestration_id] = orchestration_plan
            
            logger.info(f"Orchestration plan created with {len(subtasks)} subtasks")
            return orchestration_plan
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            raise
    
    def _determine_execution_order(self, subtasks: List[SubTask]) -> List[List[str]]:
        """Determine optimal execution order considering dependencies."""
        # Simple topological sort based on dependencies
        # Returns list of lists, where each inner list contains tasks that can run in parallel
        
        # For now, return a simple sequential order
        # This would be enhanced with proper dependency analysis
        return [[subtask.id] for subtask in sorted(subtasks, key=lambda x: x.priority, reverse=True)]
    
    def _calculate_resource_requirements(self, subtasks: List[SubTask]) -> Dict[str, Any]:
        """Calculate resource requirements for the orchestration."""
        return {
            "estimated_campers": len(subtasks),
            "complexity_distribution": {
                complexity.value: sum(1 for st in subtasks if st.estimated_complexity == complexity)
                for complexity in TaskComplexity
            },
            "parallel_capacity": len([st for st in subtasks if not st.dependencies])
        }
    
    def _estimate_duration(self, subtasks: List[SubTask]) -> Dict[str, int]:
        """Estimate duration for the orchestration."""
        complexity_weights = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 2,
            TaskComplexity.COMPLEX: 4,
            TaskComplexity.HIGHLY_COMPLEX: 8
        }
        
        total_weight = sum(complexity_weights[st.estimated_complexity] for st in subtasks)
        
        return {
            "estimated_minutes": total_weight * 5,  # 5 minutes per complexity unit
            "min_duration": len(subtasks) * 2,      # Minimum 2 minutes per subtask
            "max_duration": total_weight * 10       # Maximum estimate
        }
    
    def get_orchestration_status(self, orchestration_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an active orchestration."""
        return self.active_orchestrations.get(orchestration_id)
    
    def list_active_orchestrations(self) -> List[str]:
        """List all active orchestration IDs."""
        return list(self.active_orchestrations.keys())
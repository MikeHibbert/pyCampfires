"""
PartyOrchestrator for coordinating multiple campfires and managing execution flow.

This module provides the orchestration layer that coordinates multiple campfire instances,
manages task distribution, handles execution topologies, and ensures efficient
communication through the MCP protocol.
"""

import asyncio
import logging
import uuid
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .campfire import Campfire
from .torch import Torch
from .factory import CampfireFactory
from .orchestration import (
    TaskDecomposer, RoleAwareOrchestrator, SubTask, 
    RoleRequirement, TaskComplexity
)
from ..party_box.box_driver import BoxDriver
from ..mcp.protocol import MCPProtocol


logger = logging.getLogger(__name__)


class ExecutionTopology(Enum):
    """Supported execution topologies for task processing."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"


class TaskStatus(Enum):
    """Status of tasks in the orchestration system."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionPlan:
    """Plan for executing a complex task across multiple campfires."""
    id: str
    original_task: str
    subtasks: List[SubTask]
    topology: ExecutionTopology
    dependencies: Dict[str, List[str]]  # subtask_id -> [dependency_ids]
    estimated_duration: int  # minutes
    priority: int
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Tracks the execution of a task within the orchestration system."""
    id: str
    subtask_id: str
    campfire_instance_id: str
    status: TaskStatus
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_torch: Optional[Torch] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PartyOrchestrator:
    """
    Orchestrates multiple campfires to execute complex tasks efficiently.
    
    The PartyOrchestrator manages the lifecycle of complex tasks by:
    1. Decomposing tasks into subtasks
    2. Creating appropriate campfire instances
    3. Distributing subtasks based on topology
    4. Monitoring execution progress
    5. Aggregating results
    6. Handling failures and retries
    """
    
    def __init__(self, 
                 party_box: BoxDriver,
                 campfire_factory: CampfireFactory,
                 mcp_protocol: Optional[MCPProtocol] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the PartyOrchestrator.
        
        Args:
            party_box: Shared Party Box for all campfires
            campfire_factory: Factory for creating campfire instances
            mcp_protocol: MCP protocol for communication
            config: Orchestrator configuration
        """
        self.party_box = party_box
        self.campfire_factory = campfire_factory
        self.mcp_protocol = mcp_protocol
        self.config = config or {}
        
        # Initialize orchestration components
        self.task_decomposer = TaskDecomposer(
            config=self.config.get('decomposer', {}),
            mcp_protocol=mcp_protocol
        )
        self.role_orchestrator = RoleAwareOrchestrator(
            config=self.config.get('role_orchestrator', {}),
            mcp_protocol=mcp_protocol
        )
        
        # Execution state
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.task_executions: Dict[str, TaskExecution] = {}
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        
        # Configuration
        self.max_concurrent_executions = self.config.get('max_concurrent_executions', 20)
        self.default_retry_limit = self.config.get('default_retry_limit', 3)
        self.execution_timeout_minutes = self.config.get('execution_timeout_minutes', 60)
        
        # Background tasks
        self._execution_workers: List[asyncio.Task] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_running = False
    
    async def start(self):
        """Start the orchestrator and its background workers."""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start execution workers
        worker_count = self.config.get('worker_count', 5)
        for i in range(worker_count):
            worker = asyncio.create_task(self._execution_worker(f"worker_{i}"))
            self._execution_workers.append(worker)
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_executions())
        
        logger.info(f"PartyOrchestrator started with {worker_count} workers")
    
    async def stop(self):
        """Stop the orchestrator and clean up resources."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Cancel all workers
        for worker in self._execution_workers:
            worker.cancel()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._execution_workers, return_exceptions=True)
        if self._monitoring_task:
            await asyncio.gather(self._monitoring_task, return_exceptions=True)
        
        self._execution_workers.clear()
        self._monitoring_task = None
        
        logger.info("PartyOrchestrator stopped")
    
    async def execute_complex_task(self, 
                                 task_description: str,
                                 topology: ExecutionTopology = ExecutionTopology.ADAPTIVE,
                                 priority: int = 5,
                                 context: Dict[str, Any] = None) -> str:
        """
        Execute a complex task using multiple campfires.
        
        Args:
            task_description: Description of the task to execute
            topology: Execution topology to use
            priority: Task priority (1-10, higher is more important)
            context: Additional context for task execution
            
        Returns:
            Execution plan ID for tracking progress
        """
        if not self._is_running:
            await self.start()
        
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        context = context or {}
        
        try:
            # Analyze task complexity
            complexity = await self.task_decomposer.analyze_task_complexity(task_description)
            
            # Decompose task into subtasks
            subtasks = await self.task_decomposer.decompose_task(
                task_description, 
                complexity,
                context
            )
            
            # Generate role requirements for each subtask
            role_requirements = []
            for subtask in subtasks:
                role_req = await self.role_orchestrator.generate_role_for_subtask(subtask)
                role_requirements.append(role_req)
            
            # Create execution plan
            execution_plan = ExecutionPlan(
                id=plan_id,
                original_task=task_description,
                subtasks=subtasks,
                topology=topology,
                dependencies=self._analyze_dependencies(subtasks),
                estimated_duration=self._estimate_duration(subtasks, topology),
                priority=priority,
                metadata={
                    'context': context,
                    'complexity': complexity.value,
                    'role_requirements': [req.__dict__ for req in role_requirements]
                }
            )
            
            # Store execution plan
            self.active_plans[plan_id] = execution_plan
            
            # Create task executions for each subtask
            for i, subtask in enumerate(subtasks):
                execution_id = f"exec_{plan_id}_{i}"
                execution = TaskExecution(
                    id=execution_id,
                    subtask_id=subtask.id,
                    campfire_instance_id="",  # Will be assigned later
                    status=TaskStatus.PENDING,
                    metadata={
                        'role_requirement': role_requirements[i].__dict__,
                        'plan_id': plan_id
                    }
                )
                self.task_executions[execution_id] = execution
            
            # Queue executions based on topology
            await self._queue_executions_by_topology(execution_plan)
            
            logger.info(f"Created execution plan {plan_id} with {len(subtasks)} subtasks")
            return plan_id
            
        except Exception as e:
            logger.error(f"Failed to create execution plan: {e}")
            raise
    
    def _analyze_dependencies(self, subtasks: List[SubTask]) -> Dict[str, List[str]]:
        """Analyze dependencies between subtasks."""
        dependencies = {}
        
        for subtask in subtasks:
            deps = []
            # Simple dependency analysis based on subtask metadata
            if 'depends_on' in subtask.metadata:
                deps = subtask.metadata['depends_on']
            elif subtask.priority > 5:  # High priority tasks might depend on others
                # Find lower priority tasks that might be dependencies
                for other in subtasks:
                    if (other.id != subtask.id and 
                        other.priority <= subtask.priority and
                        any(keyword in other.description.lower() 
                            for keyword in ['setup', 'prepare', 'initialize'])):
                        deps.append(other.id)
            
            dependencies[subtask.id] = deps
        
        return dependencies
    
    def _estimate_duration(self, subtasks: List[SubTask], topology: ExecutionTopology) -> int:
        """Estimate execution duration in minutes."""
        base_duration = sum(subtask.estimated_duration for subtask in subtasks)
        
        # Adjust based on topology
        if topology == ExecutionTopology.PARALLEL:
            # Parallel execution - duration is roughly the longest subtask
            return max(subtask.estimated_duration for subtask in subtasks) + 5
        elif topology == ExecutionTopology.SEQUENTIAL:
            # Sequential execution - sum of all durations
            return base_duration + len(subtasks) * 2  # Add overhead
        elif topology == ExecutionTopology.HIERARCHICAL:
            # Hierarchical - somewhere between parallel and sequential
            return int(base_duration * 0.7)
        else:
            # Default estimation
            return base_duration
    
    async def _queue_executions_by_topology(self, plan: ExecutionPlan):
        """Queue task executions based on the specified topology."""
        if plan.topology == ExecutionTopology.SEQUENTIAL:
            await self._queue_sequential_executions(plan)
        elif plan.topology == ExecutionTopology.PARALLEL:
            await self._queue_parallel_executions(plan)
        elif plan.topology == ExecutionTopology.HIERARCHICAL:
            await self._queue_hierarchical_executions(plan)
        elif plan.topology == ExecutionTopology.ADAPTIVE:
            await self._queue_adaptive_executions(plan)
        else:
            # Default to parallel
            await self._queue_parallel_executions(plan)
    
    async def _queue_sequential_executions(self, plan: ExecutionPlan):
        """Queue executions for sequential processing."""
        # Sort subtasks by priority and dependencies
        sorted_subtasks = sorted(plan.subtasks, key=lambda x: (x.priority, x.id))
        
        for subtask in sorted_subtasks:
            execution_id = self._find_execution_by_subtask(subtask.id)
            if execution_id:
                await self.execution_queue.put(execution_id)
    
    async def _queue_parallel_executions(self, plan: ExecutionPlan):
        """Queue executions for parallel processing."""
        # Queue all executions at once for parallel processing
        for subtask in plan.subtasks:
            execution_id = self._find_execution_by_subtask(subtask.id)
            if execution_id:
                await self.execution_queue.put(execution_id)
    
    async def _queue_hierarchical_executions(self, plan: ExecutionPlan):
        """Queue executions for hierarchical processing."""
        # Group subtasks by priority and queue in waves
        priority_groups = {}
        for subtask in plan.subtasks:
            priority = subtask.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(subtask)
        
        # Queue highest priority tasks first
        for priority in sorted(priority_groups.keys(), reverse=True):
            for subtask in priority_groups[priority]:
                execution_id = self._find_execution_by_subtask(subtask.id)
                if execution_id:
                    await self.execution_queue.put(execution_id)
    
    async def _queue_adaptive_executions(self, plan: ExecutionPlan):
        """Queue executions using adaptive topology based on task characteristics."""
        # Analyze task characteristics to choose optimal topology
        total_subtasks = len(plan.subtasks)
        avg_duration = sum(st.estimated_duration for st in plan.subtasks) / total_subtasks
        
        if total_subtasks <= 3 or avg_duration > 30:
            # Few tasks or long-running tasks - use sequential
            await self._queue_sequential_executions(plan)
        elif total_subtasks > 10 and avg_duration < 10:
            # Many short tasks - use parallel
            await self._queue_parallel_executions(plan)
        else:
            # Medium complexity - use hierarchical
            await self._queue_hierarchical_executions(plan)
    
    def _find_execution_by_subtask(self, subtask_id: str) -> Optional[str]:
        """Find execution ID by subtask ID."""
        for exec_id, execution in self.task_executions.items():
            if execution.subtask_id == subtask_id:
                return exec_id
        return None
    
    async def _execution_worker(self, worker_name: str):
        """Worker that processes task executions from the queue."""
        logger.info(f"Execution worker {worker_name} started")
        
        while self._is_running:
            try:
                # Get next execution from queue
                execution_id = await asyncio.wait_for(
                    self.execution_queue.get(), 
                    timeout=1.0
                )
                
                if execution_id not in self.task_executions:
                    continue
                
                execution = self.task_executions[execution_id]
                
                # Check if dependencies are satisfied
                if not await self._check_dependencies(execution):
                    # Re-queue for later
                    await asyncio.sleep(2)
                    await self.execution_queue.put(execution_id)
                    continue
                
                # Execute the task
                await self._execute_task(execution, worker_name)
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
        
        logger.info(f"Execution worker {worker_name} stopped")
    
    async def _check_dependencies(self, execution: TaskExecution) -> bool:
        """Check if all dependencies for an execution are satisfied."""
        plan_id = execution.metadata.get('plan_id')
        if not plan_id or plan_id not in self.active_plans:
            return True
        
        plan = self.active_plans[plan_id]
        dependencies = plan.dependencies.get(execution.subtask_id, [])
        
        if not dependencies:
            return True
        
        # Check if all dependency executions are completed
        for dep_subtask_id in dependencies:
            dep_execution_id = self._find_execution_by_subtask(dep_subtask_id)
            if not dep_execution_id:
                continue
            
            dep_execution = self.task_executions[dep_execution_id]
            if dep_execution.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _execute_task(self, execution: TaskExecution, worker_name: str):
        """Execute a single task."""
        try:
            # Update execution status
            execution.status = TaskStatus.ASSIGNED
            execution.assigned_at = datetime.now()
            
            # Get subtask and role requirement
            plan_id = execution.metadata.get('plan_id')
            plan = self.active_plans[plan_id]
            subtask = next(st for st in plan.subtasks if st.id == execution.subtask_id)
            role_req_data = execution.metadata.get('role_requirement', {})
            
            # Reconstruct role requirement
            role_requirement = RoleRequirement(
                role_name=role_req_data.get('role_name', 'general'),
                expertise_areas=role_req_data.get('expertise_areas', []),
                required_capabilities=role_req_data.get('required_capabilities', []),
                personality_traits=role_req_data.get('personality_traits', []),
                context_sources=role_req_data.get('context_sources', [])
            )
            
            # Create campfire for the subtask
            campfire_id = await self.campfire_factory.create_campfire_for_subtask(
                subtask, role_requirement
            )
            execution.campfire_instance_id = campfire_id
            
            # Update status
            execution.status = TaskStatus.IN_PROGRESS
            execution.started_at = datetime.now()
            
            # Create torch for processing
            torch = Torch(
                claim=subtask.description,
                confidence=0.8,
                metadata={
                    'subtask_id': subtask.id,
                    'plan_id': plan_id,
                    'worker': worker_name,
                    'context': subtask.context
                }
            )
            
            # Process torch in campfire
            result_torch = await self.campfire_factory.process_torch_in_campfire(
                campfire_id, torch
            )
            
            if result_torch:
                # Task completed successfully
                execution.status = TaskStatus.COMPLETED
                execution.completed_at = datetime.now()
                execution.result_torch = result_torch
                
                logger.info(f"Task {execution.subtask_id} completed by {worker_name}")
            else:
                # Task failed
                execution.status = TaskStatus.FAILED
                execution.error_message = "Processing returned no result"
                
                # Retry if under limit
                if execution.retry_count < self.default_retry_limit:
                    execution.retry_count += 1
                    execution.status = TaskStatus.PENDING
                    await self.execution_queue.put(execution.id)
                    logger.warning(f"Task {execution.subtask_id} failed, retrying ({execution.retry_count}/{self.default_retry_limit})")
                else:
                    logger.error(f"Task {execution.subtask_id} failed permanently")
            
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            logger.error(f"Task execution failed: {e}")
            
            # Retry if under limit
            if execution.retry_count < self.default_retry_limit:
                execution.retry_count += 1
                execution.status = TaskStatus.PENDING
                await self.execution_queue.put(execution.id)
    
    async def _monitor_executions(self):
        """Monitor execution progress and handle timeouts."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.now()
                
                # Check for timed out executions
                for execution in self.task_executions.values():
                    if (execution.status == TaskStatus.IN_PROGRESS and 
                        execution.started_at and
                        (current_time - execution.started_at).total_seconds() > self.execution_timeout_minutes * 60):
                        
                        execution.status = TaskStatus.FAILED
                        execution.error_message = "Execution timeout"
                        logger.warning(f"Task {execution.subtask_id} timed out")
                
                # Check for completed plans
                await self._check_completed_plans()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _check_completed_plans(self):
        """Check for completed execution plans and clean up."""
        completed_plans = []
        
        for plan_id, plan in self.active_plans.items():
            plan_executions = [
                exec for exec in self.task_executions.values()
                if exec.metadata.get('plan_id') == plan_id
            ]
            
            if all(exec.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                   for exec in plan_executions):
                completed_plans.append(plan_id)
        
        # Clean up completed plans
        for plan_id in completed_plans:
            logger.info(f"Execution plan {plan_id} completed")
            # Could trigger completion callbacks here
    
    def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an execution plan."""
        if plan_id not in self.active_plans:
            return None
        
        plan = self.active_plans[plan_id]
        plan_executions = [
            exec for exec in self.task_executions.values()
            if exec.metadata.get('plan_id') == plan_id
        ]
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(1 for exec in plan_executions if exec.status == status)
        
        return {
            'plan_id': plan_id,
            'original_task': plan.original_task,
            'topology': plan.topology.value,
            'total_subtasks': len(plan.subtasks),
            'status_counts': status_counts,
            'created_at': plan.created_at.isoformat(),
            'estimated_duration': plan.estimated_duration,
            'priority': plan.priority
        }
    
    def list_active_plans(self) -> List[Dict[str, Any]]:
        """List all active execution plans."""
        return [self.get_execution_status(plan_id) for plan_id in self.active_plans.keys()]
    
    async def cancel_execution_plan(self, plan_id: str) -> bool:
        """Cancel an execution plan and all its tasks."""
        if plan_id not in self.active_plans:
            return False
        
        # Cancel all executions for this plan
        for execution in self.task_executions.values():
            if execution.metadata.get('plan_id') == plan_id:
                if execution.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                    execution.status = TaskStatus.CANCELLED
                
                # Terminate associated campfire if exists
                if execution.campfire_instance_id:
                    await self.campfire_factory.terminate_campfire(execution.campfire_instance_id)
        
        logger.info(f"Cancelled execution plan {plan_id}")
        return True
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        total_executions = len(self.task_executions)
        completed_executions = sum(1 for exec in self.task_executions.values() 
                                 if exec.status == TaskStatus.COMPLETED)
        failed_executions = sum(1 for exec in self.task_executions.values() 
                              if exec.status == TaskStatus.FAILED)
        
        return {
            'active_plans': len(self.active_plans),
            'total_executions': total_executions,
            'completed_executions': completed_executions,
            'failed_executions': failed_executions,
            'success_rate': (completed_executions / total_executions * 100) if total_executions > 0 else 0,
            'queue_size': self.execution_queue.qsize(),
            'worker_count': len(self._execution_workers),
            'is_running': self._is_running
        }
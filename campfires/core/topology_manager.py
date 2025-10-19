"""
TopologyManager for sophisticated execution pattern management.

This module provides advanced topology management for orchestrating multiple
campfires and tasks with different execution patterns including sequential,
parallel, hierarchical, and adaptive topologies.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .party_orchestrator import ExecutionTopology, TaskStatus, TaskExecution


logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Extended topology types."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    PIPELINE = "pipeline"
    SCATTER_GATHER = "scatter_gather"
    CONDITIONAL = "conditional"
    HYBRID = "hybrid"


class NodeType(Enum):
    """Execution node types."""
    TASK = "task"
    CAMPFIRE = "campfire"
    GATEWAY = "gateway"
    DECISION = "decision"
    MERGE = "merge"
    SPLIT = "split"


class ExecutionStrategy(Enum):
    """Execution strategies for different scenarios."""
    FAIL_FAST = "fail_fast"
    CONTINUE_ON_ERROR = "continue_on_error"
    RETRY_FAILED = "retry_failed"
    BEST_EFFORT = "best_effort"
    ALL_OR_NOTHING = "all_or_nothing"


@dataclass
class ExecutionNode:
    """Represents a node in the execution topology."""
    id: str
    name: str
    node_type: NodeType
    dependencies: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 5  # 1-10, higher is more important
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyDefinition:
    """Defines a complete execution topology."""
    id: str
    name: str
    description: str
    topology_type: TopologyType
    nodes: List[ExecutionNode]
    execution_strategy: ExecutionStrategy
    global_timeout_minutes: int = 60
    max_concurrent_nodes: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context for topology execution."""
    topology_id: str
    execution_id: str
    start_time: datetime
    variables: Dict[str, Any] = field(default_factory=dict)
    node_results: Dict[str, Any] = field(default_factory=dict)
    node_status: Dict[str, TaskStatus] = field(default_factory=dict)
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyExecutionResult:
    """Result of topology execution."""
    topology_id: str
    execution_id: str
    overall_status: TaskStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    node_results: Dict[str, Any]
    node_status: Dict[str, TaskStatus]
    success_count: int
    failure_count: int
    error_summary: List[str]
    metrics: Dict[str, Any]


class TopologyManager:
    """
    Advanced topology manager for complex execution patterns.
    
    Supports:
    - Sequential execution with dependencies
    - Parallel execution with resource management
    - Hierarchical execution with sub-topologies
    - Adaptive execution based on runtime conditions
    - Pipeline execution with data flow
    - Scatter-gather patterns
    - Conditional execution with decision nodes
    - Hybrid topologies combining multiple patterns
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the topology manager.
        
        Args:
            config: Manager configuration
        """
        self.config = config or {}
        
        # Execution configuration
        self.max_concurrent_executions = self.config.get('max_concurrent_executions', 5)
        self.default_timeout_minutes = self.config.get('default_timeout_minutes', 30)
        self.retry_delay_seconds = self.config.get('retry_delay_seconds', 5)
        
        # Active executions
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        
        # Topology registry
        self._topology_registry: Dict[str, TopologyDefinition] = {}
        
        # Execution strategies
        self._strategy_handlers = {
            ExecutionStrategy.FAIL_FAST: self._execute_fail_fast,
            ExecutionStrategy.CONTINUE_ON_ERROR: self._execute_continue_on_error,
            ExecutionStrategy.RETRY_FAILED: self._execute_retry_failed,
            ExecutionStrategy.BEST_EFFORT: self._execute_best_effort,
            ExecutionStrategy.ALL_OR_NOTHING: self._execute_all_or_nothing
        }
        
        # Node type handlers
        self._node_handlers = {
            NodeType.TASK: self._execute_task_node,
            NodeType.CAMPFIRE: self._execute_campfire_node,
            NodeType.GATEWAY: self._execute_gateway_node,
            NodeType.DECISION: self._execute_decision_node,
            NodeType.MERGE: self._execute_merge_node,
            NodeType.SPLIT: self._execute_split_node
        }
    
    def register_topology(self, topology: TopologyDefinition):
        """
        Register a topology definition.
        
        Args:
            topology: Topology definition to register
        """
        # Validate topology
        self._validate_topology(topology)
        
        # Register topology
        self._topology_registry[topology.id] = topology
        logger.info(f"Registered topology: {topology.name} ({topology.id})")
    
    def get_topology(self, topology_id: str) -> Optional[TopologyDefinition]:
        """
        Get a registered topology.
        
        Args:
            topology_id: ID of the topology
            
        Returns:
            Topology definition or None if not found
        """
        return self._topology_registry.get(topology_id)
    
    def list_topologies(self) -> List[TopologyDefinition]:
        """
        List all registered topologies.
        
        Returns:
            List of topology definitions
        """
        return list(self._topology_registry.values())
    
    async def execute_topology(self, 
                             topology_id: str,
                             execution_variables: Dict[str, Any] = None,
                             node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """
        Execute a topology.
        
        Args:
            topology_id: ID of the topology to execute
            execution_variables: Variables for execution context
            node_executors: Custom node executor functions
            
        Returns:
            Topology execution result
        """
        topology = self.get_topology(topology_id)
        if not topology:
            raise ValueError(f"Topology not found: {topology_id}")
        
        # Create execution context
        execution_id = f"{topology_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        context = ExecutionContext(
            topology_id=topology_id,
            execution_id=execution_id,
            start_time=datetime.now(),
            variables=execution_variables or {},
            node_status={node.id: TaskStatus.PENDING for node in topology.nodes}
        )
        
        # Store active execution
        self._active_executions[execution_id] = context
        
        try:
            logger.info(f"Starting topology execution: {execution_id}")
            
            # Execute based on topology type
            if topology.topology_type == TopologyType.SEQUENTIAL:
                result = await self._execute_sequential_topology(topology, context, node_executors)
            elif topology.topology_type == TopologyType.PARALLEL:
                result = await self._execute_parallel_topology(topology, context, node_executors)
            elif topology.topology_type == TopologyType.HIERARCHICAL:
                result = await self._execute_hierarchical_topology(topology, context, node_executors)
            elif topology.topology_type == TopologyType.ADAPTIVE:
                result = await self._execute_adaptive_topology(topology, context, node_executors)
            elif topology.topology_type == TopologyType.PIPELINE:
                result = await self._execute_pipeline_topology(topology, context, node_executors)
            elif topology.topology_type == TopologyType.SCATTER_GATHER:
                result = await self._execute_scatter_gather_topology(topology, context, node_executors)
            elif topology.topology_type == TopologyType.CONDITIONAL:
                result = await self._execute_conditional_topology(topology, context, node_executors)
            elif topology.topology_type == TopologyType.HYBRID:
                result = await self._execute_hybrid_topology(topology, context, node_executors)
            else:
                raise ValueError(f"Unsupported topology type: {topology.topology_type}")
            
            logger.info(f"Topology execution completed: {execution_id} - {result.overall_status.value}")
            return result
            
        except Exception as e:
            logger.error(f"Topology execution failed: {execution_id} - {e}")
            return self._create_error_result(topology, context, str(e))
        finally:
            # Clean up
            if execution_id in self._active_executions:
                del self._active_executions[execution_id]
            if execution_id in self._execution_tasks:
                task = self._execution_tasks[execution_id]
                if not task.done():
                    task.cancel()
                del self._execution_tasks[execution_id]
    
    async def _execute_sequential_topology(self, 
                                         topology: TopologyDefinition,
                                         context: ExecutionContext,
                                         node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute topology sequentially."""
        # Sort nodes by dependencies
        execution_order = self._calculate_execution_order(topology.nodes)
        
        for node in execution_order:
            try:
                # Check dependencies
                if not self._check_dependencies(node, context):
                    context.node_status[node.id] = TaskStatus.FAILED
                    context.error_log.append({
                        'node_id': node.id,
                        'error': 'Dependencies not satisfied',
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    if topology.execution_strategy == ExecutionStrategy.FAIL_FAST:
                        break
                    continue
                
                # Execute node
                context.node_status[node.id] = TaskStatus.RUNNING
                result = await self._execute_node(node, context, node_executors)
                
                if result['success']:
                    context.node_status[node.id] = TaskStatus.COMPLETED
                    context.node_results[node.id] = result['data']
                else:
                    context.node_status[node.id] = TaskStatus.FAILED
                    context.error_log.append({
                        'node_id': node.id,
                        'error': result.get('error', 'Unknown error'),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    if topology.execution_strategy == ExecutionStrategy.FAIL_FAST:
                        break
                
            except Exception as e:
                context.node_status[node.id] = TaskStatus.FAILED
                context.error_log.append({
                    'node_id': node.id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                if topology.execution_strategy == ExecutionStrategy.FAIL_FAST:
                    break
        
        return self._create_execution_result(topology, context)
    
    async def _execute_parallel_topology(self, 
                                        topology: TopologyDefinition,
                                        context: ExecutionContext,
                                        node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute topology in parallel."""
        # Group nodes by dependency levels
        dependency_levels = self._calculate_dependency_levels(topology.nodes)
        
        for level_nodes in dependency_levels:
            # Execute all nodes in this level in parallel
            tasks = []
            for node in level_nodes:
                if self._check_dependencies(node, context):
                    context.node_status[node.id] = TaskStatus.RUNNING
                    task = asyncio.create_task(
                        self._execute_node_with_error_handling(node, context, node_executors)
                    )
                    tasks.append((node.id, task))
                else:
                    context.node_status[node.id] = TaskStatus.FAILED
                    context.error_log.append({
                        'node_id': node.id,
                        'error': 'Dependencies not satisfied',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Wait for all tasks in this level to complete
            if tasks:
                results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                for (node_id, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        context.node_status[node_id] = TaskStatus.FAILED
                        context.error_log.append({
                            'node_id': node_id,
                            'error': str(result),
                            'timestamp': datetime.now().isoformat()
                        })
                    elif result['success']:
                        context.node_status[node_id] = TaskStatus.COMPLETED
                        context.node_results[node_id] = result['data']
                    else:
                        context.node_status[node_id] = TaskStatus.FAILED
                        context.error_log.append({
                            'node_id': node_id,
                            'error': result.get('error', 'Unknown error'),
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Check if we should continue based on execution strategy
            if topology.execution_strategy == ExecutionStrategy.FAIL_FAST:
                failed_nodes = [node_id for node_id, status in context.node_status.items() 
                              if status == TaskStatus.FAILED]
                if failed_nodes:
                    break
        
        return self._create_execution_result(topology, context)
    
    async def _execute_hierarchical_topology(self, 
                                           topology: TopologyDefinition,
                                           context: ExecutionContext,
                                           node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute hierarchical topology with sub-topologies."""
        # Identify parent-child relationships
        hierarchy = self._build_hierarchy(topology.nodes)
        
        # Execute from root nodes down
        await self._execute_hierarchy_level(hierarchy['root'], hierarchy, context, node_executors)
        
        return self._create_execution_result(topology, context)
    
    async def _execute_adaptive_topology(self, 
                                       topology: TopologyDefinition,
                                       context: ExecutionContext,
                                       node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute adaptive topology that changes based on runtime conditions."""
        # Start with initial execution plan
        execution_plan = self._create_adaptive_plan(topology, context)
        
        while execution_plan:
            # Execute next batch of nodes
            current_batch = execution_plan.pop(0)
            
            # Execute batch
            for node in current_batch:
                if self._check_dependencies(node, context):
                    context.node_status[node.id] = TaskStatus.RUNNING
                    result = await self._execute_node(node, context, node_executors)
                    
                    if result['success']:
                        context.node_status[node.id] = TaskStatus.COMPLETED
                        context.node_results[node.id] = result['data']
                    else:
                        context.node_status[node.id] = TaskStatus.FAILED
                        context.error_log.append({
                            'node_id': node.id,
                            'error': result.get('error', 'Unknown error'),
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Adapt execution plan based on current results
            execution_plan = self._adapt_execution_plan(execution_plan, context, topology)
        
        return self._create_execution_result(topology, context)
    
    async def _execute_pipeline_topology(self, 
                                        topology: TopologyDefinition,
                                        context: ExecutionContext,
                                        node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute pipeline topology with data flow."""
        # Build pipeline stages
        pipeline_stages = self._build_pipeline_stages(topology.nodes)
        
        # Execute pipeline stages
        pipeline_data = context.variables.copy()
        
        for stage in pipeline_stages:
            stage_results = {}
            
            # Execute all nodes in the stage
            for node in stage:
                if self._check_dependencies(node, context):
                    context.node_status[node.id] = TaskStatus.RUNNING
                    
                    # Pass pipeline data to node
                    node_context = context.variables.copy()
                    node_context.update(pipeline_data)
                    
                    result = await self._execute_node(node, context, node_executors, node_context)
                    
                    if result['success']:
                        context.node_status[node.id] = TaskStatus.COMPLETED
                        context.node_results[node.id] = result['data']
                        stage_results[node.id] = result['data']
                    else:
                        context.node_status[node.id] = TaskStatus.FAILED
                        context.error_log.append({
                            'node_id': node.id,
                            'error': result.get('error', 'Unknown error'),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        if topology.execution_strategy == ExecutionStrategy.FAIL_FAST:
                            return self._create_execution_result(topology, context)
            
            # Update pipeline data with stage results
            pipeline_data.update(stage_results)
        
        return self._create_execution_result(topology, context)
    
    async def _execute_scatter_gather_topology(self, 
                                             topology: TopologyDefinition,
                                             context: ExecutionContext,
                                             node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute scatter-gather topology."""
        # Identify scatter and gather nodes
        scatter_nodes = [node for node in topology.nodes if node.node_type == NodeType.SPLIT]
        gather_nodes = [node for node in topology.nodes if node.node_type == NodeType.MERGE]
        worker_nodes = [node for node in topology.nodes 
                       if node.node_type not in [NodeType.SPLIT, NodeType.MERGE]]
        
        # Execute scatter phase
        for scatter_node in scatter_nodes:
            context.node_status[scatter_node.id] = TaskStatus.RUNNING
            result = await self._execute_node(scatter_node, context, node_executors)
            
            if result['success']:
                context.node_status[scatter_node.id] = TaskStatus.COMPLETED
                context.node_results[scatter_node.id] = result['data']
            else:
                context.node_status[scatter_node.id] = TaskStatus.FAILED
                return self._create_execution_result(topology, context)
        
        # Execute worker nodes in parallel
        worker_tasks = []
        for worker_node in worker_nodes:
            if self._check_dependencies(worker_node, context):
                context.node_status[worker_node.id] = TaskStatus.RUNNING
                task = asyncio.create_task(
                    self._execute_node_with_error_handling(worker_node, context, node_executors)
                )
                worker_tasks.append((worker_node.id, task))
        
        # Wait for all worker tasks
        if worker_tasks:
            results = await asyncio.gather(*[task for _, task in worker_tasks], return_exceptions=True)
            
            for (node_id, _), result in zip(worker_tasks, results):
                if isinstance(result, Exception):
                    context.node_status[node_id] = TaskStatus.FAILED
                    context.error_log.append({
                        'node_id': node_id,
                        'error': str(result),
                        'timestamp': datetime.now().isoformat()
                    })
                elif result['success']:
                    context.node_status[node_id] = TaskStatus.COMPLETED
                    context.node_results[node_id] = result['data']
                else:
                    context.node_status[node_id] = TaskStatus.FAILED
        
        # Execute gather phase
        for gather_node in gather_nodes:
            if self._check_dependencies(gather_node, context):
                context.node_status[gather_node.id] = TaskStatus.RUNNING
                result = await self._execute_node(gather_node, context, node_executors)
                
                if result['success']:
                    context.node_status[gather_node.id] = TaskStatus.COMPLETED
                    context.node_results[gather_node.id] = result['data']
                else:
                    context.node_status[gather_node.id] = TaskStatus.FAILED
        
        return self._create_execution_result(topology, context)
    
    async def _execute_conditional_topology(self, 
                                          topology: TopologyDefinition,
                                          context: ExecutionContext,
                                          node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute conditional topology with decision nodes."""
        # Identify decision nodes and execution paths
        decision_nodes = [node for node in topology.nodes if node.node_type == NodeType.DECISION]
        
        # Execute nodes based on conditions
        executed_nodes = set()
        
        for node in topology.nodes:
            if node.id in executed_nodes:
                continue
            
            # Check if node should be executed based on conditions
            if self._should_execute_node(node, context):
                if self._check_dependencies(node, context):
                    context.node_status[node.id] = TaskStatus.RUNNING
                    result = await self._execute_node(node, context, node_executors)
                    
                    if result['success']:
                        context.node_status[node.id] = TaskStatus.COMPLETED
                        context.node_results[node.id] = result['data']
                    else:
                        context.node_status[node.id] = TaskStatus.FAILED
                        context.error_log.append({
                            'node_id': node.id,
                            'error': result.get('error', 'Unknown error'),
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    executed_nodes.add(node.id)
                else:
                    context.node_status[node.id] = TaskStatus.SKIPPED

    # Execution Strategy Methods
    async def _execute_fail_fast(self, topology: TopologyDefinition, context: ExecutionContext, node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute with fail-fast strategy - stop on first error."""
        try:
            return await self._topology_handlers[topology.topology_type](topology, context, node_executors)
        except Exception as e:
            logger.error(f"Fail-fast execution failed: {e}")
            return self._create_error_result(topology, context, str(e))

    async def _execute_continue_on_error(self, topology: TopologyDefinition, context: ExecutionContext, node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute with continue-on-error strategy - continue despite errors."""
        try:
            return await self._topology_handlers[topology.topology_type](topology, context, node_executors)
        except Exception as e:
            logger.warning(f"Continue-on-error execution encountered error: {e}")
            return self._create_execution_result(topology, context)

    async def _execute_retry_failed(self, topology: TopologyDefinition, context: ExecutionContext, node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute with retry strategy - retry failed nodes."""
        try:
            return await self._topology_handlers[topology.topology_type](topology, context, node_executors)
        except Exception as e:
            logger.info(f"Retry-failed execution will retry: {e}")
            return await self._topology_handlers[topology.topology_type](topology, context, node_executors)

    async def _execute_best_effort(self, topology: TopologyDefinition, context: ExecutionContext, node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute with best-effort strategy - complete as much as possible."""
        try:
            return await self._topology_handlers[topology.topology_type](topology, context, node_executors)
        except Exception as e:
            logger.info(f"Best-effort execution completed with errors: {e}")
            return self._create_execution_result(topology, context)

    async def _execute_all_or_nothing(self, topology: TopologyDefinition, context: ExecutionContext, node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute with all-or-nothing strategy - rollback on any failure."""
        try:
            result = await self._topology_handlers[topology.topology_type](topology, context, node_executors)
            if result.failure_count > 0:
                logger.warning("All-or-nothing execution failed, rolling back")
                return self._create_error_result(topology, context, "All-or-nothing strategy failed")
            return result
        except Exception as e:
            logger.error(f"All-or-nothing execution failed: {e}")
            return self._create_error_result(topology, context, str(e))
    
    async def _execute_hybrid_topology(self, 
                                      topology: TopologyDefinition,
                                      context: ExecutionContext,
                                      node_executors: Dict[str, Callable] = None) -> TopologyExecutionResult:
        """Execute hybrid topology combining multiple patterns."""
        # Analyze topology structure and determine execution strategy
        execution_groups = self._analyze_hybrid_structure(topology.nodes)
        
        for group_type, nodes in execution_groups.items():
            if group_type == 'sequential':
                await self._execute_sequential_group(nodes, context, node_executors)
            elif group_type == 'parallel':
                await self._execute_parallel_group(nodes, context, node_executors)
            elif group_type == 'conditional':
                await self._execute_conditional_group(nodes, context, node_executors)
        
        return self._create_execution_result(topology, context)
    
    # Helper methods for topology execution
    
    def _validate_topology(self, topology: TopologyDefinition):
        """Validate topology definition."""
        if not topology.nodes:
            raise ValueError("Topology must have at least one node")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(topology.nodes):
            raise ValueError("Topology has circular dependencies")
        
        # Validate node references
        node_ids = {node.id for node in topology.nodes}
        for node in topology.nodes:
            for dep in node.dependencies:
                if dep not in node_ids:
                    raise ValueError(f"Node {node.id} references unknown dependency: {dep}")
    
    def _has_circular_dependencies(self, nodes: List[ExecutionNode]) -> bool:
        """Check for circular dependencies in nodes."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str, node_map: Dict[str, ExecutionNode]) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = node_map.get(node_id)
            if node:
                for dep in node.dependencies:
                    if dep not in visited:
                        if has_cycle(dep, node_map):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        node_map = {node.id: node for node in nodes}
        
        for node in nodes:
            if node.id not in visited:
                if has_cycle(node.id, node_map):
                    return True
        
        return False
    
    def _calculate_execution_order(self, nodes: List[ExecutionNode]) -> List[ExecutionNode]:
        """Calculate execution order based on dependencies."""
        # Topological sort
        in_degree = {node.id: 0 for node in nodes}
        node_map = {node.id: node for node in nodes}
        
        # Calculate in-degrees
        for node in nodes:
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node.id] += 1
        
        # Find nodes with no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority
            queue.sort(key=lambda x: node_map[x].priority, reverse=True)
            current = queue.pop(0)
            result.append(node_map[current])
            
            # Update in-degrees
            for node in nodes:
                if current in node.dependencies:
                    in_degree[node.id] -= 1
                    if in_degree[node.id] == 0:
                        queue.append(node.id)
        
        return result
    
    def _calculate_dependency_levels(self, nodes: List[ExecutionNode]) -> List[List[ExecutionNode]]:
        """Calculate dependency levels for parallel execution."""
        levels = []
        remaining_nodes = nodes.copy()
        node_map = {node.id: node for node in nodes}
        
        while remaining_nodes:
            # Find nodes with no unresolved dependencies
            current_level = []
            completed_nodes = {node.id for level in levels for node in level}
            
            for node in remaining_nodes:
                unresolved_deps = [dep for dep in node.dependencies if dep not in completed_nodes]
                if not unresolved_deps:
                    current_level.append(node)
            
            if not current_level:
                # This shouldn't happen with valid topologies
                break
            
            # Sort by priority
            current_level.sort(key=lambda x: x.priority, reverse=True)
            levels.append(current_level)
            
            # Remove processed nodes
            for node in current_level:
                remaining_nodes.remove(node)
        
        return levels
    
    def _check_dependencies(self, node: ExecutionNode, context: ExecutionContext) -> bool:
        """Check if node dependencies are satisfied."""
        for dep in node.dependencies:
            if dep not in context.node_status:
                return False
            if context.node_status[dep] != TaskStatus.COMPLETED:
                return False
        return True
    
    def _should_execute_node(self, node: ExecutionNode, context: ExecutionContext) -> bool:
        """Check if node should be executed based on conditions."""
        if not node.conditions:
            return True
        
        # Evaluate conditions
        for condition_key, condition_value in node.conditions.items():
            if condition_key == 'if_variable':
                var_name, expected_value = condition_value.split('=', 1)
                if context.variables.get(var_name) != expected_value:
                    return False
            elif condition_key == 'if_node_result':
                node_id, expected_result = condition_value.split('=', 1)
                if context.node_results.get(node_id) != expected_result:
                    return False
            elif condition_key == 'if_node_status':
                node_id, expected_status = condition_value.split('=', 1)
                if context.node_status.get(node_id) != TaskStatus(expected_status):
                    return False
        
        return True
    
    async def _execute_node(self, 
                          node: ExecutionNode, 
                          context: ExecutionContext,
                          node_executors: Dict[str, Callable] = None,
                          node_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single node."""
        try:
            # Get node executor
            if node_executors and node.id in node_executors:
                executor = node_executors[node.id]
            else:
                executor = self._node_handlers.get(node.node_type, self._default_node_executor)
            
            # Execute with timeout
            if node.timeout_seconds:
                result = await asyncio.wait_for(
                    executor(node, context, node_context or {}),
                    timeout=node.timeout_seconds
                )
            else:
                result = await executor(node, context, node_context or {})
            
            return result
            
        except asyncio.TimeoutError:
            return {'success': False, 'error': f'Node execution timed out after {node.timeout_seconds} seconds'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_node_with_error_handling(self, 
                                               node: ExecutionNode, 
                                               context: ExecutionContext,
                                               node_executors: Dict[str, Callable] = None) -> Dict[str, Any]:
        """Execute node with error handling and retries."""
        last_error = None
        
        for attempt in range(node.max_retries + 1):
            try:
                result = await self._execute_node(node, context, node_executors)
                if result['success']:
                    return result
                else:
                    last_error = result.get('error', 'Unknown error')
                    if attempt < node.max_retries:
                        await asyncio.sleep(self.retry_delay_seconds)
            except Exception as e:
                last_error = str(e)
                if attempt < node.max_retries:
                    await asyncio.sleep(self.retry_delay_seconds)
        
        return {'success': False, 'error': f'Failed after {node.max_retries + 1} attempts: {last_error}'}
    
    # Default node executors
    
    async def _execute_task_node(self, node: ExecutionNode, context: ExecutionContext, node_context: Dict[str, Any]) -> Dict[str, Any]:
        """Default task node executor."""
        # Placeholder implementation
        await asyncio.sleep(0.1)  # Simulate work
        return {'success': True, 'data': f'Task {node.id} completed'}
    
    async def _execute_campfire_node(self, node: ExecutionNode, context: ExecutionContext, node_context: Dict[str, Any]) -> Dict[str, Any]:
        """Default campfire node executor."""
        # Placeholder implementation
        await asyncio.sleep(0.2)  # Simulate work
        return {'success': True, 'data': f'Campfire {node.id} completed'}
    
    async def _execute_gateway_node(self, node: ExecutionNode, context: ExecutionContext, node_context: Dict[str, Any]) -> Dict[str, Any]:
        """Default gateway node executor."""
        return {'success': True, 'data': f'Gateway {node.id} passed'}
    
    async def _execute_decision_node(self, node: ExecutionNode, context: ExecutionContext, node_context: Dict[str, Any]) -> Dict[str, Any]:
        """Default decision node executor."""
        # Simple decision logic
        decision = node.conditions.get('decision_logic', 'true')
        return {'success': True, 'data': {'decision': decision}}
    
    async def _execute_merge_node(self, node: ExecutionNode, context: ExecutionContext, node_context: Dict[str, Any]) -> Dict[str, Any]:
        """Default merge node executor."""
        # Collect results from dependencies
        merged_data = {}
        for dep in node.dependencies:
            if dep in context.node_results:
                merged_data[dep] = context.node_results[dep]
        
        return {'success': True, 'data': merged_data}
    
    async def _execute_split_node(self, node: ExecutionNode, context: ExecutionContext, node_context: Dict[str, Any]) -> Dict[str, Any]:
        """Default split node executor."""
        # Split input data
        split_count = node.metadata.get('split_count', 2)
        input_data = node_context.get('input_data', [])
        
        if isinstance(input_data, list):
            chunk_size = len(input_data) // split_count
            chunks = [input_data[i:i + chunk_size] for i in range(0, len(input_data), chunk_size)]
        else:
            chunks = [input_data] * split_count
        
        return {'success': True, 'data': {'chunks': chunks}}
    
    async def _default_node_executor(self, node: ExecutionNode, context: ExecutionContext, node_context: Dict[str, Any]) -> Dict[str, Any]:
        """Default node executor for unknown node types."""
        return {'success': True, 'data': f'Node {node.id} executed with default executor'}
    
    def _create_execution_result(self, topology: TopologyDefinition, context: ExecutionContext) -> TopologyExecutionResult:
        """Create execution result from context."""
        end_time = datetime.now()
        duration = (end_time - context.start_time).total_seconds()
        
        success_count = sum(1 for status in context.node_status.values() if status == TaskStatus.COMPLETED)
        failure_count = sum(1 for status in context.node_status.values() if status == TaskStatus.FAILED)
        
        # Determine overall status
        if failure_count == 0:
            overall_status = TaskStatus.COMPLETED
        elif success_count == 0:
            overall_status = TaskStatus.FAILED
        else:
            overall_status = TaskStatus.PARTIAL
        
        return TopologyExecutionResult(
            topology_id=topology.id,
            execution_id=context.execution_id,
            overall_status=overall_status,
            start_time=context.start_time,
            end_time=end_time,
            duration_seconds=duration,
            node_results=context.node_results,
            node_status=context.node_status,
            success_count=success_count,
            failure_count=failure_count,
            error_summary=[error['error'] for error in context.error_log],
            metrics=context.metrics
        )
    
    def _create_error_result(self, topology: TopologyDefinition, context: ExecutionContext, error: str) -> TopologyExecutionResult:
        """Create error execution result."""
        end_time = datetime.now()
        duration = (end_time - context.start_time).total_seconds()
        
        return TopologyExecutionResult(
            topology_id=topology.id,
            execution_id=context.execution_id,
            overall_status=TaskStatus.FAILED,
            start_time=context.start_time,
            end_time=end_time,
            duration_seconds=duration,
            node_results=context.node_results,
            node_status=context.node_status,
            success_count=0,
            failure_count=len(topology.nodes),
            error_summary=[error],
            metrics={}
        )
    
    # Additional helper methods for complex topologies
    
    def _build_hierarchy(self, nodes: List[ExecutionNode]) -> Dict[str, Any]:
        """Build hierarchy structure from nodes."""
        # Simplified hierarchy building
        hierarchy = {'root': [], 'children': {}}
        
        for node in nodes:
            if not node.dependencies:
                hierarchy['root'].append(node)
            else:
                for dep in node.dependencies:
                    if dep not in hierarchy['children']:
                        hierarchy['children'][dep] = []
                    hierarchy['children'][dep].append(node)
        
        return hierarchy
    
    async def _execute_hierarchy_level(self, nodes: List[ExecutionNode], hierarchy: Dict[str, Any], context: ExecutionContext, node_executors: Dict[str, Callable] = None):
        """Execute a level in the hierarchy."""
        for node in nodes:
            if self._check_dependencies(node, context):
                context.node_status[node.id] = TaskStatus.RUNNING
                result = await self._execute_node(node, context, node_executors)
                
                if result['success']:
                    context.node_status[node.id] = TaskStatus.COMPLETED
                    context.node_results[node.id] = result['data']
                    
                    # Execute children
                    children = hierarchy['children'].get(node.id, [])
                    if children:
                        await self._execute_hierarchy_level(children, hierarchy, context, node_executors)
                else:
                    context.node_status[node.id] = TaskStatus.FAILED
    
    def _create_adaptive_plan(self, topology: TopologyDefinition, context: ExecutionContext) -> List[List[ExecutionNode]]:
        """Create initial adaptive execution plan."""
        # Start with dependency-based grouping
        return self._calculate_dependency_levels(topology.nodes)
    
    def _adapt_execution_plan(self, current_plan: List[List[ExecutionNode]], context: ExecutionContext, topology: TopologyDefinition) -> List[List[ExecutionNode]]:
        """Adapt execution plan based on current results."""
        # Simple adaptation: remove failed dependencies
        adapted_plan = []
        
        for batch in current_plan:
            adapted_batch = []
            for node in batch:
                if self._check_dependencies(node, context):
                    adapted_batch.append(node)
            
            if adapted_batch:
                adapted_plan.append(adapted_batch)
        
        return adapted_plan
    
    def _build_pipeline_stages(self, nodes: List[ExecutionNode]) -> List[List[ExecutionNode]]:
        """Build pipeline stages from nodes."""
        # Use dependency levels as pipeline stages
        return self._calculate_dependency_levels(nodes)
    
    def _analyze_hybrid_structure(self, nodes: List[ExecutionNode]) -> Dict[str, List[ExecutionNode]]:
        """Analyze hybrid topology structure."""
        # Simple grouping by node metadata
        groups = {'sequential': [], 'parallel': [], 'conditional': []}
        
        for node in nodes:
            group_type = node.metadata.get('execution_group', 'sequential')
            if group_type in groups:
                groups[group_type].append(node)
            else:
                groups['sequential'].append(node)
        
        return groups
    
    async def _execute_sequential_group(self, nodes: List[ExecutionNode], context: ExecutionContext, node_executors: Dict[str, Callable] = None):
        """Execute a group of nodes sequentially."""
        for node in nodes:
            if self._check_dependencies(node, context):
                context.node_status[node.id] = TaskStatus.RUNNING
                result = await self._execute_node(node, context, node_executors)
                
                if result['success']:
                    context.node_status[node.id] = TaskStatus.COMPLETED
                    context.node_results[node.id] = result['data']
                else:
                    context.node_status[node.id] = TaskStatus.FAILED
    
    async def _execute_parallel_group(self, nodes: List[ExecutionNode], context: ExecutionContext, node_executors: Dict[str, Callable] = None):
        """Execute a group of nodes in parallel."""
        tasks = []
        for node in nodes:
            if self._check_dependencies(node, context):
                context.node_status[node.id] = TaskStatus.RUNNING
                task = asyncio.create_task(
                    self._execute_node_with_error_handling(node, context, node_executors)
                )
                tasks.append((node.id, task))
        
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (node_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    context.node_status[node_id] = TaskStatus.FAILED
                elif result['success']:
                    context.node_status[node_id] = TaskStatus.COMPLETED
                    context.node_results[node_id] = result['data']
                else:
                    context.node_status[node_id] = TaskStatus.FAILED
    
    async def _execute_conditional_group(self, nodes: List[ExecutionNode], context: ExecutionContext, node_executors: Dict[str, Callable] = None):
        """Execute a group of nodes conditionally."""
        for node in nodes:
            if self._should_execute_node(node, context) and self._check_dependencies(node, context):
                context.node_status[node.id] = TaskStatus.RUNNING
                result = await self._execute_node(node, context, node_executors)
                
                if result['success']:
                    context.node_status[node.id] = TaskStatus.COMPLETED
                    context.node_results[node.id] = result['data']
                else:
                    context.node_status[node.id] = TaskStatus.FAILED
            else:
                context.node_status[node.id] = TaskStatus.SKIPPED
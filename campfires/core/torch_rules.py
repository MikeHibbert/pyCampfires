"""
Torch Rules Engine for Conditional Processing and Routing.

This module provides a sophisticated rules engine for the Campfires framework,
enabling conditional processing, dynamic routing, and rule-based decision making
for task orchestration and execution flow control.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from abc import ABC, abstractmethod
import operator
from collections import defaultdict

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of rules in the engine."""
    CONDITION = "condition"
    ROUTING = "routing"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    TRIGGER = "trigger"
    FILTER = "filter"
    AGGREGATION = "aggregation"
    TEMPORAL = "temporal"


class OperatorType(Enum):
    """Supported operators for rule conditions."""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_EQUAL = "le"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    AND = "and"
    OR = "or"
    NOT = "not"


class ActionType(Enum):
    """Types of actions that can be triggered by rules."""
    ROUTE_TO = "route_to"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    REJECT = "reject"
    DELAY = "delay"
    RETRY = "retry"
    LOG = "log"
    ALERT = "alert"
    EXECUTE = "execute"
    BRANCH = "branch"
    MERGE = "merge"
    SPLIT = "split"


class RulePriority(Enum):
    """Rule execution priorities."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class RuleCondition:
    """Individual rule condition."""
    field: str
    operator: OperatorType
    value: Any
    case_sensitive: bool = True
    negate: bool = False


@dataclass
class RuleAction:
    """Action to be executed when rule matches."""
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    target: Optional[str] = None
    delay_seconds: float = 0.0
    retry_count: int = 0
    timeout_seconds: Optional[float] = None


@dataclass
class RuleMetadata:
    """Metadata for rules."""
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    version: str = "1.0"
    tags: Set[str] = field(default_factory=set)
    enabled: bool = True
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


@dataclass
class TorchRule:
    """Complete rule definition."""
    metadata: RuleMetadata
    rule_type: RuleType
    priority: RulePriority
    conditions: List[RuleCondition]
    actions: List[RuleAction]
    condition_logic: str = "AND"  # AND, OR, CUSTOM
    custom_logic: Optional[str] = None  # For complex condition logic
    context_requirements: Set[str] = field(default_factory=set)
    execution_window: Optional[Tuple[datetime, datetime]] = None
    max_executions: Optional[int] = None
    cooldown_seconds: float = 0.0
    dependencies: List[str] = field(default_factory=list)  # Rule IDs this rule depends on


@dataclass
class RuleExecutionContext:
    """Context for rule execution."""
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    trace_id: str = ""
    user_context: Dict[str, Any] = field(default_factory=dict)
    session_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleExecutionResult:
    """Result of rule execution."""
    rule_id: str
    matched: bool
    executed: bool
    actions_performed: List[RuleAction]
    execution_time_ms: float
    error: Optional[str] = None
    output_data: Optional[Dict[str, Any]] = None
    next_rules: List[str] = field(default_factory=list)
    routing_decision: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleConditionEvaluator:
    """Evaluates rule conditions against context data."""
    
    def __init__(self):
        """Initialize the condition evaluator."""
        self._operators = {
            OperatorType.EQUALS: operator.eq,
            OperatorType.NOT_EQUALS: operator.ne,
            OperatorType.GREATER_THAN: operator.gt,
            OperatorType.GREATER_EQUAL: operator.ge,
            OperatorType.LESS_THAN: operator.lt,
            OperatorType.LESS_EQUAL: operator.le,
        }
    
    def evaluate_condition(self, condition: RuleCondition, context: RuleExecutionContext) -> bool:
        """
        Evaluate a single condition against context data.
        
        Args:
            condition: Rule condition to evaluate
            context: Execution context with data
            
        Returns:
            True if condition matches
        """
        try:
            # Get field value from context
            field_value = self._get_field_value(condition.field, context.data)
            
            # Handle null checks first
            if condition.operator == OperatorType.IS_NULL:
                result = field_value is None
            elif condition.operator == OperatorType.IS_NOT_NULL:
                result = field_value is not None
            elif field_value is None:
                # If field is None and we're not checking for null, condition fails
                result = False
            else:
                # Evaluate based on operator
                result = self._evaluate_operator(condition, field_value)
            
            # Apply negation if specified
            if condition.negate:
                result = not result
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating condition {condition.field} {condition.operator.value}: {e}")
            return False
    
    def evaluate_conditions(self, 
                          conditions: List[RuleCondition], 
                          context: RuleExecutionContext,
                          logic: str = "AND",
                          custom_logic: Optional[str] = None) -> bool:
        """
        Evaluate multiple conditions with specified logic.
        
        Args:
            conditions: List of conditions to evaluate
            context: Execution context
            logic: Logic operator (AND, OR, CUSTOM)
            custom_logic: Custom logic expression for complex conditions
            
        Returns:
            True if conditions match according to logic
        """
        if not conditions:
            return True
        
        # Evaluate each condition
        condition_results = []
        for i, condition in enumerate(conditions):
            result = self.evaluate_condition(condition, context)
            condition_results.append(result)
            logger.debug(f"Condition {i}: {condition.field} {condition.operator.value} {condition.value} = {result}")
        
        # Apply logic
        if logic == "AND":
            return all(condition_results)
        elif logic == "OR":
            return any(condition_results)
        elif logic == "CUSTOM" and custom_logic:
            return self._evaluate_custom_logic(condition_results, custom_logic)
        else:
            logger.warning(f"Unknown logic type: {logic}, defaulting to AND")
            return all(condition_results)
    
    def _get_field_value(self, field_path: str, data: Dict[str, Any]) -> Any:
        """
        Get field value from nested data using dot notation.
        
        Args:
            field_path: Field path (e.g., "user.profile.name")
            data: Data dictionary
            
        Returns:
            Field value or None if not found
        """
        try:
            value = data
            for field in field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(field)
                elif isinstance(value, list) and field.isdigit():
                    index = int(field)
                    value = value[index] if 0 <= index < len(value) else None
                else:
                    return None
                
                if value is None:
                    break
            
            return value
            
        except (KeyError, IndexError, ValueError, TypeError):
            return None
    
    def _evaluate_operator(self, condition: RuleCondition, field_value: Any) -> bool:
        """Evaluate operator-specific condition."""
        operator_type = condition.operator
        expected_value = condition.value
        
        # Handle case sensitivity for string operations
        if isinstance(field_value, str) and isinstance(expected_value, str) and not condition.case_sensitive:
            field_value = field_value.lower()
            expected_value = expected_value.lower()
        
        # Basic comparison operators
        if operator_type in self._operators:
            return self._operators[operator_type](field_value, expected_value)
        
        # String-specific operators
        elif operator_type == OperatorType.CONTAINS:
            return isinstance(field_value, str) and expected_value in field_value
        
        elif operator_type == OperatorType.NOT_CONTAINS:
            return isinstance(field_value, str) and expected_value not in field_value
        
        elif operator_type == OperatorType.STARTS_WITH:
            return isinstance(field_value, str) and field_value.startswith(expected_value)
        
        elif operator_type == OperatorType.ENDS_WITH:
            return isinstance(field_value, str) and field_value.endswith(expected_value)
        
        elif operator_type == OperatorType.REGEX_MATCH:
            if isinstance(field_value, str):
                flags = 0 if condition.case_sensitive else re.IGNORECASE
                return bool(re.search(expected_value, field_value, flags))
            return False
        
        # Collection operators
        elif operator_type == OperatorType.IN:
            return field_value in expected_value if hasattr(expected_value, '__contains__') else False
        
        elif operator_type == OperatorType.NOT_IN:
            return field_value not in expected_value if hasattr(expected_value, '__contains__') else True
        
        else:
            logger.warning(f"Unknown operator: {operator_type}")
            return False
    
    def _evaluate_custom_logic(self, condition_results: List[bool], custom_logic: str) -> bool:
        """
        Evaluate custom logic expression.
        
        Args:
            condition_results: Results of individual conditions
            custom_logic: Custom logic expression (e.g., "(0 AND 1) OR 2")
            
        Returns:
            Result of custom logic evaluation
        """
        try:
            # Replace condition indices with their results
            expression = custom_logic
            for i, result in enumerate(condition_results):
                expression = expression.replace(str(i), str(result))
            
            # Replace logical operators
            expression = expression.replace('AND', ' and ')
            expression = expression.replace('OR', ' or ')
            expression = expression.replace('NOT', ' not ')
            
            # Evaluate the expression safely
            # Note: In production, consider using a safer expression evaluator
            return eval(expression)
            
        except Exception as e:
            logger.error(f"Error evaluating custom logic '{custom_logic}': {e}")
            return False


class RuleActionExecutor:
    """Executes rule actions."""
    
    def __init__(self):
        """Initialize the action executor."""
        self._action_handlers: Dict[ActionType, Callable] = {
            ActionType.ROUTE_TO: self._handle_route_to,
            ActionType.TRANSFORM: self._handle_transform,
            ActionType.VALIDATE: self._handle_validate,
            ActionType.REJECT: self._handle_reject,
            ActionType.DELAY: self._handle_delay,
            ActionType.RETRY: self._handle_retry,
            ActionType.LOG: self._handle_log,
            ActionType.ALERT: self._handle_alert,
            ActionType.EXECUTE: self._handle_execute,
            ActionType.BRANCH: self._handle_branch,
            ActionType.MERGE: self._handle_merge,
            ActionType.SPLIT: self._handle_split,
        }
    
    async def execute_action(self, 
                           action: RuleAction, 
                           context: RuleExecutionContext) -> Dict[str, Any]:
        """
        Execute a single action.
        
        Args:
            action: Action to execute
            context: Execution context
            
        Returns:
            Action execution result
        """
        try:
            # Apply delay if specified
            if action.delay_seconds > 0:
                await asyncio.sleep(action.delay_seconds)
            
            # Get action handler
            handler = self._action_handlers.get(action.action_type)
            if not handler:
                raise ValueError(f"Unknown action type: {action.action_type}")
            
            # Execute action with timeout if specified
            if action.timeout_seconds:
                result = await asyncio.wait_for(
                    handler(action, context),
                    timeout=action.timeout_seconds
                )
            else:
                result = await handler(action, context)
            
            return {
                'success': True,
                'action_type': action.action_type.value,
                'result': result,
                'error': None
            }
            
        except asyncio.TimeoutError:
            error_msg = f"Action {action.action_type.value} timed out after {action.timeout_seconds}s"
            logger.error(error_msg)
            return {
                'success': False,
                'action_type': action.action_type.value,
                'result': None,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Error executing action {action.action_type.value}: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'action_type': action.action_type.value,
                'result': None,
                'error': error_msg
            }
    
    async def execute_actions(self, 
                            actions: List[RuleAction], 
                            context: RuleExecutionContext) -> List[Dict[str, Any]]:
        """
        Execute multiple actions.
        
        Args:
            actions: List of actions to execute
            context: Execution context
            
        Returns:
            List of action execution results
        """
        results = []
        
        for action in actions:
            # Handle retry logic
            retry_count = 0
            max_retries = action.retry_count
            
            while retry_count <= max_retries:
                result = await self.execute_action(action, context)
                
                if result['success'] or retry_count >= max_retries:
                    results.append(result)
                    break
                
                retry_count += 1
                if retry_count <= max_retries:
                    logger.info(f"Retrying action {action.action_type.value} (attempt {retry_count + 1})")
                    await asyncio.sleep(1.0 * retry_count)  # Exponential backoff
        
        return results
    
    # Action handlers
    
    async def _handle_route_to(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle routing action."""
        target = action.target or action.parameters.get('target')
        if not target:
            raise ValueError("Route action requires target")
        
        return {
            'routing_decision': target,
            'parameters': action.parameters,
            'context_data': context.data
        }
    
    async def _handle_transform(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle data transformation action."""
        transformation = action.parameters.get('transformation')
        if not transformation:
            raise ValueError("Transform action requires transformation specification")
        
        # Apply transformation to context data
        transformed_data = self._apply_transformation(context.data, transformation)
        
        return {
            'transformed_data': transformed_data,
            'original_data': context.data
        }
    
    async def _handle_validate(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle validation action."""
        validation_rules = action.parameters.get('validation_rules', [])
        
        validation_results = []
        for rule in validation_rules:
            field = rule.get('field')
            validator = rule.get('validator')
            
            if field and validator:
                field_value = self._get_nested_value(context.data, field)
                is_valid = self._validate_field(field_value, validator)
                
                validation_results.append({
                    'field': field,
                    'valid': is_valid,
                    'value': field_value,
                    'validator': validator
                })
        
        all_valid = all(result['valid'] for result in validation_results)
        
        return {
            'valid': all_valid,
            'validation_results': validation_results
        }
    
    async def _handle_reject(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle rejection action."""
        reason = action.parameters.get('reason', 'Rule condition not met')
        
        return {
            'rejected': True,
            'reason': reason,
            'context_id': context.execution_id
        }
    
    async def _handle_delay(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle delay action."""
        delay_seconds = action.parameters.get('delay_seconds', 1.0)
        await asyncio.sleep(delay_seconds)
        
        return {
            'delayed': True,
            'delay_seconds': delay_seconds
        }
    
    async def _handle_retry(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle retry action."""
        return {
            'retry_requested': True,
            'retry_parameters': action.parameters
        }
    
    async def _handle_log(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle logging action."""
        level = action.parameters.get('level', 'info')
        message = action.parameters.get('message', 'Rule executed')
        
        # Format message with context data
        formatted_message = message.format(**context.data)
        
        # Log at specified level
        if level == 'debug':
            logger.debug(formatted_message)
        elif level == 'info':
            logger.info(formatted_message)
        elif level == 'warning':
            logger.warning(formatted_message)
        elif level == 'error':
            logger.error(formatted_message)
        
        return {
            'logged': True,
            'level': level,
            'message': formatted_message
        }
    
    async def _handle_alert(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle alert action."""
        alert_type = action.parameters.get('type', 'info')
        message = action.parameters.get('message', 'Alert triggered')
        recipients = action.parameters.get('recipients', [])
        
        # Format message
        formatted_message = message.format(**context.data)
        
        # In a real implementation, this would send alerts via email, SMS, etc.
        logger.warning(f"ALERT [{alert_type}]: {formatted_message}")
        
        return {
            'alert_sent': True,
            'type': alert_type,
            'message': formatted_message,
            'recipients': recipients
        }
    
    async def _handle_execute(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle execution action."""
        command = action.parameters.get('command')
        if not command:
            raise ValueError("Execute action requires command")
        
        # In a real implementation, this would execute the command safely
        logger.info(f"Executing command: {command}")
        
        return {
            'executed': True,
            'command': command,
            'result': 'Command executed successfully'
        }
    
    async def _handle_branch(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle branching action."""
        branches = action.parameters.get('branches', [])
        condition_field = action.parameters.get('condition_field')
        
        if condition_field:
            condition_value = self._get_nested_value(context.data, condition_field)
            
            for branch in branches:
                if branch.get('condition_value') == condition_value:
                    return {
                        'branch_taken': branch.get('target'),
                        'condition_field': condition_field,
                        'condition_value': condition_value
                    }
        
        # Default branch
        default_branch = action.parameters.get('default_branch')
        return {
            'branch_taken': default_branch,
            'condition_field': condition_field,
            'condition_value': None
        }
    
    async def _handle_merge(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle merge action."""
        merge_sources = action.parameters.get('sources', [])
        merge_strategy = action.parameters.get('strategy', 'union')
        
        merged_data = {}
        
        for source in merge_sources:
            source_data = self._get_nested_value(context.data, source)
            if isinstance(source_data, dict):
                if merge_strategy == 'union':
                    merged_data.update(source_data)
                elif merge_strategy == 'intersection':
                    if not merged_data:
                        merged_data = source_data.copy()
                    else:
                        merged_data = {k: v for k, v in merged_data.items() if k in source_data}
        
        return {
            'merged_data': merged_data,
            'strategy': merge_strategy,
            'sources': merge_sources
        }
    
    async def _handle_split(self, action: RuleAction, context: RuleExecutionContext) -> Dict[str, Any]:
        """Handle split action."""
        split_field = action.parameters.get('field')
        split_criteria = action.parameters.get('criteria', [])
        
        if not split_field:
            raise ValueError("Split action requires field")
        
        field_value = self._get_nested_value(context.data, split_field)
        
        splits = []
        for criterion in split_criteria:
            condition = criterion.get('condition')
            target = criterion.get('target')
            
            if self._evaluate_split_condition(field_value, condition):
                splits.append({
                    'target': target,
                    'condition': condition,
                    'data': context.data
                })
        
        return {
            'splits': splits,
            'field': split_field,
            'value': field_value
        }
    
    # Helper methods
    
    def _apply_transformation(self, data: Dict[str, Any], transformation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transformation to data."""
        transformed = data.copy()
        
        # Handle field mappings
        field_mappings = transformation.get('field_mappings', {})
        for old_field, new_field in field_mappings.items():
            if old_field in transformed:
                transformed[new_field] = transformed.pop(old_field)
        
        # Handle value transformations
        value_transformations = transformation.get('value_transformations', {})
        for field, transform_func in value_transformations.items():
            if field in transformed:
                if transform_func == 'upper':
                    transformed[field] = str(transformed[field]).upper()
                elif transform_func == 'lower':
                    transformed[field] = str(transformed[field]).lower()
                elif transform_func == 'strip':
                    transformed[field] = str(transformed[field]).strip()
        
        return transformed
    
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested value from data using dot notation."""
        try:
            value = data
            for field in field_path.split('.'):
                value = value[field]
            return value
        except (KeyError, TypeError):
            return None
    
    def _validate_field(self, value: Any, validator: Dict[str, Any]) -> bool:
        """Validate field value against validator."""
        validator_type = validator.get('type')
        
        if validator_type == 'required':
            return value is not None and value != ''
        elif validator_type == 'type':
            expected_type = validator.get('expected_type')
            return type(value).__name__ == expected_type
        elif validator_type == 'range':
            min_val = validator.get('min')
            max_val = validator.get('max')
            return (min_val is None or value >= min_val) and (max_val is None or value <= max_val)
        elif validator_type == 'regex':
            pattern = validator.get('pattern')
            return bool(re.match(pattern, str(value))) if pattern else False
        
        return True
    
    def _evaluate_split_condition(self, value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate split condition."""
        condition_type = condition.get('type')
        
        if condition_type == 'equals':
            return value == condition.get('value')
        elif condition_type == 'greater_than':
            return value > condition.get('value')
        elif condition_type == 'contains':
            return condition.get('value') in str(value)
        
        return False


class TorchRulesEngine:
    """
    Advanced rules engine for conditional processing and routing.
    
    Features:
    - Rule-based decision making
    - Dynamic routing based on conditions
    - Complex condition evaluation with custom logic
    - Action execution with retry and timeout support
    - Rule dependency management
    - Performance monitoring and metrics
    - Rule versioning and lifecycle management
    - Context-aware execution
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Torch rules engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config or {}
        
        # Core components
        self.condition_evaluator = RuleConditionEvaluator()
        self.action_executor = RuleActionExecutor()
        
        # Rule storage
        self._rules: Dict[str, TorchRule] = {}
        self._rule_groups: Dict[str, List[str]] = defaultdict(list)
        self._rule_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Execution tracking
        self._execution_history: List[RuleExecutionResult] = []
        self._rule_cooldowns: Dict[str, datetime] = {}
        
        # Performance metrics
        self._metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time_ms': 0.0,
            'rules_count': 0,
            'active_rules_count': 0
        }
        
        # Configuration
        self.max_execution_history = self.config.get('max_execution_history', 1000)
        self.enable_metrics = self.config.get('enable_metrics', True)
        self.parallel_execution = self.config.get('parallel_execution', True)
    
    def add_rule(self, rule: TorchRule) -> bool:
        """
        Add a rule to the engine.
        
        Args:
            rule: Rule to add
            
        Returns:
            True if rule was added successfully
        """
        try:
            # Validate rule
            if not self._validate_rule(rule):
                return False
            
            # Store rule
            self._rules[rule.metadata.id] = rule
            
            # Update dependencies
            for dependency_id in rule.dependencies:
                self._rule_dependencies[dependency_id].add(rule.metadata.id)
            
            # Update metrics
            self._update_rule_metrics()
            
            logger.info(f"Added rule: {rule.metadata.id} ({rule.metadata.name})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule {rule.metadata.id}: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a rule from the engine.
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if rule was removed successfully
        """
        if rule_id not in self._rules:
            return False
        
        try:
            # Remove from dependencies
            rule = self._rules[rule_id]
            for dependency_id in rule.dependencies:
                self._rule_dependencies[dependency_id].discard(rule_id)
            
            # Remove rule
            del self._rules[rule_id]
            
            # Clean up cooldowns
            if rule_id in self._rule_cooldowns:
                del self._rule_cooldowns[rule_id]
            
            # Update metrics
            self._update_rule_metrics()
            
            logger.info(f"Removed rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing rule {rule_id}: {e}")
            return False
    
    def get_rule(self, rule_id: str) -> Optional[TorchRule]:
        """
        Get a rule by ID.
        
        Args:
            rule_id: Rule ID
            
        Returns:
            Rule or None if not found
        """
        return self._rules.get(rule_id)
    
    def list_rules(self, 
                   rule_type: Optional[RuleType] = None,
                   enabled_only: bool = True) -> List[TorchRule]:
        """
        List rules with optional filtering.
        
        Args:
            rule_type: Filter by rule type
            enabled_only: Only return enabled rules
            
        Returns:
            List of matching rules
        """
        rules = []
        
        for rule in self._rules.values():
            if enabled_only and not rule.metadata.enabled:
                continue
            
            if rule_type and rule.rule_type != rule_type:
                continue
            
            rules.append(rule)
        
        # Sort by priority
        rules.sort(key=lambda r: r.priority.value)
        
        return rules
    
    async def execute_rules(self, 
                          context: RuleExecutionContext,
                          rule_types: Optional[List[RuleType]] = None,
                          rule_ids: Optional[List[str]] = None) -> List[RuleExecutionResult]:
        """
        Execute rules against context.
        
        Args:
            context: Execution context
            rule_types: Filter by rule types
            rule_ids: Execute specific rules by ID
            
        Returns:
            List of execution results
        """
        start_time = datetime.now()
        
        try:
            # Get rules to execute
            rules_to_execute = self._get_executable_rules(rule_types, rule_ids)
            
            # Execute rules
            if self.parallel_execution and len(rules_to_execute) > 1:
                results = await self._execute_rules_parallel(rules_to_execute, context)
            else:
                results = await self._execute_rules_sequential(rules_to_execute, context)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_execution_metrics(results, execution_time)
            
            # Store execution history
            self._store_execution_history(results)
            
            logger.debug(f"Executed {len(rules_to_execute)} rules in {execution_time:.2f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Error executing rules: {e}")
            return []
    
    async def execute_rule(self, 
                         rule_id: str, 
                         context: RuleExecutionContext) -> Optional[RuleExecutionResult]:
        """
        Execute a single rule.
        
        Args:
            rule_id: Rule ID to execute
            context: Execution context
            
        Returns:
            Execution result or None if rule not found
        """
        rule = self._rules.get(rule_id)
        if not rule:
            return None
        
        return await self._execute_single_rule(rule, context)
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].metadata.enabled = True
            self._update_rule_metrics()
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].metadata.enabled = False
            self._update_rule_metrics()
            return True
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get engine metrics.
        
        Returns:
            Metrics dictionary
        """
        return self._metrics.copy()
    
    def get_execution_history(self, 
                            rule_id: Optional[str] = None,
                            limit: int = 100) -> List[RuleExecutionResult]:
        """
        Get execution history.
        
        Args:
            rule_id: Filter by rule ID
            limit: Maximum number of results
            
        Returns:
            List of execution results
        """
        history = self._execution_history
        
        if rule_id:
            history = [result for result in history if result.rule_id == rule_id]
        
        return history[-limit:] if limit else history
    
    # Private methods
    
    def _validate_rule(self, rule: TorchRule) -> bool:
        """Validate rule definition."""
        # Check required fields
        if not rule.metadata.id or not rule.metadata.name:
            logger.error("Rule must have ID and name")
            return False
        
        # Check for duplicate ID
        if rule.metadata.id in self._rules:
            logger.error(f"Rule with ID {rule.metadata.id} already exists")
            return False
        
        # Validate conditions
        if not rule.conditions:
            logger.warning(f"Rule {rule.metadata.id} has no conditions")
        
        # Validate actions
        if not rule.actions:
            logger.warning(f"Rule {rule.metadata.id} has no actions")
        
        # Validate dependencies
        for dependency_id in rule.dependencies:
            if dependency_id not in self._rules:
                logger.error(f"Rule {rule.metadata.id} depends on non-existent rule {dependency_id}")
                return False
        
        return True
    
    def _get_executable_rules(self, 
                            rule_types: Optional[List[RuleType]] = None,
                            rule_ids: Optional[List[str]] = None) -> List[TorchRule]:
        """Get rules that can be executed."""
        if rule_ids:
            # Execute specific rules
            rules = [self._rules[rule_id] for rule_id in rule_ids if rule_id in self._rules]
        else:
            # Get all enabled rules
            rules = [rule for rule in self._rules.values() if rule.metadata.enabled]
        
        # Filter by rule types
        if rule_types:
            rules = [rule for rule in rules if rule.rule_type in rule_types]
        
        # Filter by execution window
        now = datetime.now()
        rules = [rule for rule in rules if self._is_in_execution_window(rule, now)]
        
        # Filter by cooldown
        rules = [rule for rule in rules if self._is_cooldown_expired(rule, now)]
        
        # Filter by max executions
        rules = [rule for rule in rules if self._can_execute_more(rule)]
        
        # Sort by priority and dependencies
        rules = self._sort_rules_by_priority_and_dependencies(rules)
        
        return rules
    
    def _is_in_execution_window(self, rule: TorchRule, now: datetime) -> bool:
        """Check if rule is in execution window."""
        if not rule.execution_window:
            return True
        
        start_time, end_time = rule.execution_window
        return start_time <= now <= end_time
    
    def _is_cooldown_expired(self, rule: TorchRule, now: datetime) -> bool:
        """Check if rule cooldown has expired."""
        if rule.cooldown_seconds <= 0:
            return True
        
        last_execution = self._rule_cooldowns.get(rule.metadata.id)
        if not last_execution:
            return True
        
        cooldown_expires = last_execution + timedelta(seconds=rule.cooldown_seconds)
        return now >= cooldown_expires
    
    def _can_execute_more(self, rule: TorchRule) -> bool:
        """Check if rule can be executed more times."""
        if not rule.max_executions:
            return True
        
        return rule.metadata.execution_count < rule.max_executions
    
    def _sort_rules_by_priority_and_dependencies(self, rules: List[TorchRule]) -> List[TorchRule]:
        """Sort rules by priority and resolve dependencies."""
        # Simple topological sort for dependencies
        sorted_rules = []
        remaining_rules = rules.copy()
        
        while remaining_rules:
            # Find rules with no unresolved dependencies
            ready_rules = []
            for rule in remaining_rules:
                dependencies_satisfied = all(
                    dep_id not in [r.metadata.id for r in remaining_rules]
                    for dep_id in rule.dependencies
                )
                if dependencies_satisfied:
                    ready_rules.append(rule)
            
            if not ready_rules:
                # Circular dependency or missing dependency
                logger.warning("Circular dependency detected in rules")
                ready_rules = remaining_rules
            
            # Sort ready rules by priority
            ready_rules.sort(key=lambda r: r.priority.value)
            
            # Add to sorted list and remove from remaining
            sorted_rules.extend(ready_rules)
            for rule in ready_rules:
                remaining_rules.remove(rule)
        
        return sorted_rules
    
    async def _execute_rules_sequential(self, 
                                      rules: List[TorchRule], 
                                      context: RuleExecutionContext) -> List[RuleExecutionResult]:
        """Execute rules sequentially."""
        results = []
        
        for rule in rules:
            result = await self._execute_single_rule(rule, context)
            if result:
                results.append(result)
                
                # Update context with rule output if available
                if result.output_data:
                    context.data.update(result.output_data)
        
        return results
    
    async def _execute_rules_parallel(self, 
                                    rules: List[TorchRule], 
                                    context: RuleExecutionContext) -> List[RuleExecutionResult]:
        """Execute rules in parallel."""
        tasks = []
        
        for rule in rules:
            task = asyncio.create_task(self._execute_single_rule(rule, context))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = []
        for result in results:
            if isinstance(result, RuleExecutionResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Rule execution failed: {result}")
        
        return valid_results
    
    async def _execute_single_rule(self, 
                                 rule: TorchRule, 
                                 context: RuleExecutionContext) -> Optional[RuleExecutionResult]:
        """Execute a single rule."""
        start_time = datetime.now()
        
        try:
            # Check context requirements
            if not self._check_context_requirements(rule, context):
                return None
            
            # Evaluate conditions
            conditions_matched = self.condition_evaluator.evaluate_conditions(
                rule.conditions,
                context,
                rule.condition_logic,
                rule.custom_logic
            )
            
            # Initialize result
            result = RuleExecutionResult(
                rule_id=rule.metadata.id,
                matched=conditions_matched,
                executed=False,
                actions_performed=[],
                execution_time_ms=0.0
            )
            
            # Execute actions if conditions matched
            if conditions_matched:
                action_results = await self.action_executor.execute_actions(rule.actions, context)
                
                # Process action results
                successful_actions = []
                output_data = {}
                routing_decision = None
                next_rules = []
                
                for action, action_result in zip(rule.actions, action_results):
                    if action_result['success']:
                        successful_actions.append(action)
                        
                        # Extract routing decision
                        if action.action_type == ActionType.ROUTE_TO:
                            routing_decision = action_result['result'].get('routing_decision')
                        
                        # Extract output data
                        if 'result' in action_result and isinstance(action_result['result'], dict):
                            output_data.update(action_result['result'])
                
                result.executed = True
                result.actions_performed = successful_actions
                result.output_data = output_data
                result.routing_decision = routing_decision
                result.next_rules = next_rules
                
                # Update rule execution tracking
                rule.metadata.execution_count += 1
                rule.metadata.last_executed = datetime.now()
                rule.metadata.success_count += 1
                
                # Set cooldown
                if rule.cooldown_seconds > 0:
                    self._rule_cooldowns[rule.metadata.id] = datetime.now()
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            logger.debug(f"Rule {rule.metadata.id} executed: matched={conditions_matched}, executed={result.executed}")
            return result
            
        except Exception as e:
            error_msg = f"Error executing rule {rule.metadata.id}: {e}"
            logger.error(error_msg)
            
            # Update failure count
            rule.metadata.failure_count += 1
            
            # Return error result
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return RuleExecutionResult(
                rule_id=rule.metadata.id,
                matched=False,
                executed=False,
                actions_performed=[],
                execution_time_ms=execution_time,
                error=error_msg
            )
    
    def _check_context_requirements(self, rule: TorchRule, context: RuleExecutionContext) -> bool:
        """Check if context meets rule requirements."""
        for requirement in rule.context_requirements:
            if requirement not in context.data:
                logger.debug(f"Rule {rule.metadata.id} context requirement not met: {requirement}")
                return False
        
        return True
    
    def _update_rule_metrics(self):
        """Update rule-related metrics."""
        self._metrics['rules_count'] = len(self._rules)
        self._metrics['active_rules_count'] = sum(
            1 for rule in self._rules.values() if rule.metadata.enabled
        )
    
    def _update_execution_metrics(self, results: List[RuleExecutionResult], execution_time_ms: float):
        """Update execution metrics."""
        if not self.enable_metrics:
            return
        
        self._metrics['total_executions'] += len(results)
        
        successful_count = sum(1 for result in results if result.executed and not result.error)
        failed_count = len(results) - successful_count
        
        self._metrics['successful_executions'] += successful_count
        self._metrics['failed_executions'] += failed_count
        
        # Update average execution time
        total_executions = self._metrics['total_executions']
        if total_executions > 0:
            current_avg = self._metrics['average_execution_time_ms']
            self._metrics['average_execution_time_ms'] = (
                (current_avg * (total_executions - len(results)) + execution_time_ms) / total_executions
            )
    
    def _store_execution_history(self, results: List[RuleExecutionResult]):
        """Store execution history."""
        self._execution_history.extend(results)
        
        # Trim history if it exceeds maximum
        if len(self._execution_history) > self.max_execution_history:
            excess = len(self._execution_history) - self.max_execution_history
            self._execution_history = self._execution_history[excess:]


# Utility functions for creating rules

def create_simple_rule(rule_id: str,
                      name: str,
                      field: str,
                      operator: str,
                      value: Any,
                      action_type: str,
                      action_target: str = None,
                      priority: str = "MEDIUM") -> TorchRule:
    """
    Create a simple rule with one condition and one action.
    
    Args:
        rule_id: Unique rule ID
        name: Rule name
        field: Field to check
        operator: Comparison operator
        value: Value to compare against
        action_type: Type of action to perform
        action_target: Target for action (if applicable)
        priority: Rule priority
        
    Returns:
        TorchRule object
    """
    # Create metadata
    metadata = RuleMetadata(
        id=rule_id,
        name=name,
        description=f"Simple rule: {field} {operator} {value}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        created_by="system"
    )
    
    # Create condition
    condition = RuleCondition(
        field=field,
        operator=OperatorType(operator),
        value=value
    )
    
    # Create action
    action = RuleAction(
        action_type=ActionType(action_type),
        target=action_target
    )
    
    return TorchRule(
        metadata=metadata,
        rule_type=RuleType.CONDITION,
        priority=RulePriority[priority],
        conditions=[condition],
        actions=[action]
    )


def create_routing_rule(rule_id: str,
                       name: str,
                       conditions: List[Dict[str, Any]],
                       routes: Dict[str, str],
                       default_route: str = None) -> TorchRule:
    """
    Create a routing rule with multiple conditions and routes.
    
    Args:
        rule_id: Unique rule ID
        name: Rule name
        conditions: List of condition dictionaries
        routes: Mapping of condition results to routes
        default_route: Default route if no conditions match
        
    Returns:
        TorchRule object
    """
    # Create metadata
    metadata = RuleMetadata(
        id=rule_id,
        name=name,
        description=f"Routing rule with {len(conditions)} conditions",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        created_by="system"
    )
    
    # Create conditions
    rule_conditions = []
    for cond in conditions:
        rule_conditions.append(RuleCondition(
            field=cond['field'],
            operator=OperatorType(cond['operator']),
            value=cond['value']
        ))
    
    # Create routing actions
    actions = []
    for route_condition, target in routes.items():
        actions.append(RuleAction(
            action_type=ActionType.ROUTE_TO,
            target=target,
            parameters={'condition': route_condition}
        ))
    
    if default_route:
        actions.append(RuleAction(
            action_type=ActionType.ROUTE_TO,
            target=default_route,
            parameters={'default': True}
        ))
    
    return TorchRule(
        metadata=metadata,
        rule_type=RuleType.ROUTING,
        priority=RulePriority.MEDIUM,
        conditions=rule_conditions,
        actions=actions
    )
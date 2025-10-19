"""Core components of the Campfires framework."""

from .torch import Torch
from .camper import Camper
from .campfire import Campfire
from .state_manager import StateManager
from .openrouter import (
    OpenRouterClient, 
    OpenRouterConfig, 
    ChatMessage, 
    ChatRequest, 
    ChatResponse,
    LLMCamperMixin
)

# New orchestration and factory components
from .orchestration import (
    TaskComplexity,
    SubTask,
    RoleRequirement,
    TaskDecomposer,
    DynamicRoleGenerator,
    RoleAwareOrchestrator
)

from .factory import (
    CampfireTemplate,
    CampfireInstance,
    DynamicCamper,
    CampfireFactory
)

from .party_orchestrator import (
    ExecutionTopology,
    TaskStatus,
    ExecutionPlan,
    TaskExecution,
    PartyOrchestrator
)

# Configuration and validation components
from .manifest_loader import (
    CampfireManifest,
    OrchestrationManifest,
    PartyManifest,
    ManifestLoader
)

from .default_auditor import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    ValidationReport,
    TaskRequirement,
    AuditContext,
    DefaultAuditor
)

# Topology management
from .topology_manager import (
    TopologyType,
    NodeType,
    ExecutionStrategy,
    ExecutionNode,
    TopologyDefinition,
    ExecutionContext,
    TopologyExecutionResult,
    TopologyManager
)

# Context path support
from .context_path import (
    ContextType,
    AccessPattern,
    ContextMetadata,
    ContextItem,
    ContextPath,
    ContextQuery,
    ContextRetrievalResult,
    ContextPathManager
)

# Torch rules engine
from .torch_rules import (
    RuleType,
    OperatorType,
    ActionType,
    RulePriority,
    RuleCondition,
    RuleAction,
    RuleMetadata,
    TorchRule,
    RuleExecutionContext,
    RuleExecutionResult,
    RuleConditionEvaluator,
    RuleActionExecutor,
    TorchRulesEngine,
    create_simple_rule,
    create_routing_rule
)

__all__ = [
    # Original core components
    "Torch", 
    "Camper", 
    "Campfire", 
    "StateManager",
    "OpenRouterClient",
    "OpenRouterConfig",
    "ChatMessage",
    "ChatRequest", 
    "ChatResponse",
    "LLMCamperMixin",
    "quick_completion",
    "quick_chat",
    
    # Orchestration components
    "TaskComplexity",
    "SubTask",
    "RoleRequirement",
    "TaskDecomposer",
    "DynamicRoleGenerator",
    "RoleAwareOrchestrator",
    
    # Factory components
    "CampfireTemplate",
    "CampfireInstance",
    "DynamicCamper",
    "CampfireFactory",
    
    # Party orchestrator components
    "ExecutionTopology",
    "TaskStatus",
    "ExecutionPlan",
    "TaskExecution",
    "PartyOrchestrator",
    
    # Configuration and validation
    "CampfireManifest",
    "OrchestrationManifest",
    "PartyManifest",
    "ManifestLoader",
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "ValidationReport",
    "TaskRequirement",
    "AuditContext",
    "DefaultAuditor",
    
    # Topology management
    "TopologyType",
    "NodeType",
    "ExecutionStrategy",
    "ExecutionNode",
    "TopologyDefinition",
    "ExecutionContext",
    "TopologyExecutionResult",
    "TopologyManager",
    
    # Context path support
    "ContextType",
    "AccessPattern",
    "ContextMetadata",
    "ContextItem",
    "ContextPath",
    "ContextQuery",
    "ContextRetrievalResult",
    "ContextPathManager",
    
    # Torch rules engine
    "RuleType",
    "OperatorType",
    "ActionType",
    "RulePriority",
    "RuleCondition",
    "RuleAction",
    "RuleMetadata",
    "TorchRule",
    "RuleExecutionContext",
    "RuleExecutionResult",
    "RuleConditionEvaluator",
    "RuleActionExecutor",
    "TorchRulesEngine",
    "create_simple_rule",
    "create_routing_rule"
]
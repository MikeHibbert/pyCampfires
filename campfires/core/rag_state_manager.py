"""
RAG State Management System for Dynamic Context Tuning.

This module provides sophisticated RAG state management that allows campers to:
1. Save their current RAG context state
2. Dynamically tune their RAG context for specific tasks
3. Restore their original RAG context after task completion
4. Manage multiple context states for different scenarios
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib
import copy

logger = logging.getLogger(__name__)


class RAGContextType(Enum):
    """Types of RAG contexts."""
    SYSTEM_PROMPT = "system_prompt"
    ROLE_DEFINITION = "role_definition"
    EXPERTISE_CONTEXT = "expertise_context"
    TASK_SPECIFIC = "task_specific"
    DOMAIN_KNOWLEDGE = "domain_knowledge"
    BEHAVIORAL_TRAITS = "behavioral_traits"


@dataclass
class RAGContextState:
    """Represents a complete RAG context state for a camper."""
    camper_id: str
    state_id: str
    timestamp: datetime
    context_type: RAGContextType
    system_prompt: Optional[str] = None
    role_definition: Optional[str] = None
    expertise_areas: List[str] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    behavioral_traits: List[str] = field(default_factory=list)
    task_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGTuningProfile:
    """Profile for tuning RAG context for specific task types."""
    profile_id: str
    name: str
    description: str
    target_task_types: List[str]
    system_prompt_template: str
    role_enhancement_template: str
    expertise_focus_areas: List[str]
    behavioral_adjustments: List[str]
    context_variables: Dict[str, Any] = field(default_factory=dict)


class RAGStateManager:
    """
    Manages RAG context states for campers, enabling dynamic tuning and restoration.
    
    Features:
    - Save/restore complete RAG context states
    - Dynamic context tuning based on task requirements
    - Multiple state management (original, task-specific, etc.)
    - Context interpolation and blending
    - State versioning and history
    - Automatic cleanup and optimization
    """
    
    def __init__(self, storage_path: str = "./rag_states", config: Dict[str, Any] = None):
        """
        Initialize the RAG state manager.
        
        Args:
            storage_path: Directory to store RAG states
            config: Configuration options
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        
        # State storage
        self._camper_states: Dict[str, Dict[str, RAGContextState]] = {}
        self._original_states: Dict[str, RAGContextState] = {}
        self._tuning_profiles: Dict[str, RAGTuningProfile] = {}
        
        # Configuration
        self.max_states_per_camper = self.config.get('max_states_per_camper', 10)
        self.auto_cleanup_enabled = self.config.get('auto_cleanup_enabled', True)
        self.state_compression_enabled = self.config.get('state_compression_enabled', True)
        
        # Load existing states and profiles
        self._load_persistent_data()
        
        # Initialize default tuning profiles
        self._initialize_default_profiles()
    
    def save_camper_state(self, camper: Any, state_id: str = "original") -> str:
        """
        Save the current RAG context state of a camper.
        
        Args:
            camper: The camper instance to save state for
            state_id: Identifier for this state (default: "original")
            
        Returns:
            The generated state ID
        """
        try:
            camper_id = getattr(camper, 'name', str(id(camper)))
            
            # Extract current RAG context from camper
            context_state = RAGContextState(
                camper_id=camper_id,
                state_id=state_id,
                timestamp=datetime.now(),
                context_type=RAGContextType.SYSTEM_PROMPT,
                system_prompt=getattr(camper, '_rag_system_prompt', None),
                role_definition=getattr(camper, '_role', None),
                expertise_areas=getattr(camper, '_expertise_areas', []),
                domain_knowledge=getattr(camper, '_domain_knowledge', {}),
                behavioral_traits=getattr(camper, '_behavioral_traits', []),
                task_context=getattr(camper, '_task_context', {}),
                metadata={
                    'rag_document_path': getattr(camper, '_rag_document_path', None),
                    'config': getattr(camper, 'config', {}),
                    'conversation_context': getattr(camper, '_conversation_context', [])
                }
            )
            
            # Store the state
            if camper_id not in self._camper_states:
                self._camper_states[camper_id] = {}
            
            self._camper_states[camper_id][state_id] = context_state
            
            # Save original state if this is the first save
            if state_id == "original" or camper_id not in self._original_states:
                self._original_states[camper_id] = copy.deepcopy(context_state)
            
            # Persist to disk
            self._persist_state(context_state)
            
            logger.info(f"Saved RAG state '{state_id}' for camper '{camper_id}'")
            return f"{camper_id}:{state_id}"
            
        except Exception as e:
            logger.error(f"Failed to save camper state: {str(e)}")
            raise
    
    def restore_camper_state(self, camper: Any, state_id: str = "original") -> bool:
        """
        Restore a previously saved RAG context state to a camper.
        
        Args:
            camper: The camper instance to restore state to
            state_id: Identifier of the state to restore
            
        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            camper_id = getattr(camper, 'name', str(id(camper)))
            
            # Find the state to restore
            if camper_id not in self._camper_states:
                logger.warning(f"No saved states found for camper '{camper_id}'")
                return False
            
            if state_id not in self._camper_states[camper_id]:
                logger.warning(f"State '{state_id}' not found for camper '{camper_id}'")
                return False
            
            context_state = self._camper_states[camper_id][state_id]
            
            # Restore RAG context to camper
            if hasattr(camper, '_rag_system_prompt'):
                camper._rag_system_prompt = context_state.system_prompt
            
            if hasattr(camper, '_role'):
                camper._role = context_state.role_definition
            
            if hasattr(camper, '_expertise_areas'):
                camper._expertise_areas = context_state.expertise_areas.copy()
            
            if hasattr(camper, '_domain_knowledge'):
                camper._domain_knowledge = context_state.domain_knowledge.copy()
            
            if hasattr(camper, '_behavioral_traits'):
                camper._behavioral_traits = context_state.behavioral_traits.copy()
            
            if hasattr(camper, '_task_context'):
                camper._task_context = context_state.task_context.copy()
            
            # Restore metadata if available
            if context_state.metadata:
                if 'rag_document_path' in context_state.metadata and hasattr(camper, '_rag_document_path'):
                    camper._rag_document_path = context_state.metadata['rag_document_path']
                
                if 'conversation_context' in context_state.metadata and hasattr(camper, '_conversation_context'):
                    camper._conversation_context = context_state.metadata['conversation_context'].copy()
            
            logger.info(f"Restored RAG state '{state_id}' for camper '{camper_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore camper state: {str(e)}")
            return False
    
    def tune_camper_for_task(self, camper: Any, task_description: str, 
                           role_requirements: Dict[str, Any] = None,
                           profile_id: str = None) -> str:
        """
        Dynamically tune a camper's RAG context for a specific task.
        
        Args:
            camper: The camper instance to tune
            task_description: Description of the task to tune for
            role_requirements: Specific role requirements for the task
            profile_id: Optional tuning profile to use
            
        Returns:
            The state ID of the tuned context
        """
        try:
            camper_id = getattr(camper, 'name', str(id(camper)))
            
            # Save current state if not already saved
            if camper_id not in self._original_states:
                self.save_camper_state(camper, "original")
            
            # Generate task-specific state ID
            task_hash = hashlib.md5(task_description.encode()).hexdigest()[:8]
            tuned_state_id = f"task_{task_hash}"
            
            # Get tuning profile
            profile = None
            if profile_id and profile_id in self._tuning_profiles:
                profile = self._tuning_profiles[profile_id]
            else:
                profile = self._select_best_profile(task_description, role_requirements)
            
            # Create tuned context
            tuned_context = self._create_tuned_context(
                camper, task_description, role_requirements, profile
            )
            
            # Apply tuned context to camper
            self._apply_context_to_camper(camper, tuned_context)
            
            # Save the tuned state
            self.save_camper_state(camper, tuned_state_id)
            
            logger.info(f"Tuned camper '{camper_id}' for task (state: '{tuned_state_id}')")
            return tuned_state_id
            
        except Exception as e:
            logger.error(f"Failed to tune camper for task: {str(e)}")
            raise
    
    def restore_original_state(self, camper: Any) -> bool:
        """
        Restore a camper to its original RAG context state.
        
        Args:
            camper: The camper instance to restore
            
        Returns:
            True if restoration was successful, False otherwise
        """
        return self.restore_camper_state(camper, "original")
    
    def create_tuning_profile(self, profile: RAGTuningProfile) -> None:
        """
        Create a new tuning profile for specific task types.
        
        Args:
            profile: The tuning profile to create
        """
        self._tuning_profiles[profile.profile_id] = profile
        self._persist_profile(profile)
        logger.info(f"Created tuning profile '{profile.profile_id}'")
    
    def get_camper_states(self, camper_id: str) -> Dict[str, RAGContextState]:
        """
        Get all saved states for a specific camper.
        
        Args:
            camper_id: The camper identifier
            
        Returns:
            Dictionary of state_id -> RAGContextState
        """
        return self._camper_states.get(camper_id, {})
    
    def cleanup_old_states(self, max_age_days: int = 7) -> int:
        """
        Clean up old RAG states to free storage space.
        
        Args:
            max_age_days: Maximum age of states to keep
            
        Returns:
            Number of states cleaned up
        """
        if not self.auto_cleanup_enabled:
            return 0
        
        cleaned_count = 0
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)
        
        for camper_id in list(self._camper_states.keys()):
            states = self._camper_states[camper_id]
            
            for state_id in list(states.keys()):
                if state_id == "original":  # Never clean up original states
                    continue
                
                state = states[state_id]
                if state.timestamp.timestamp() < cutoff_date:
                    del states[state_id]
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old RAG states")
        return cleaned_count
    
    def _create_tuned_context(self, camper: Any, task_description: str,
                            role_requirements: Dict[str, Any] = None,
                            profile: RAGTuningProfile = None) -> RAGContextState:
        """Create a tuned context based on task requirements and profile."""
        camper_id = getattr(camper, 'name', str(id(camper)))
        original_state = self._original_states.get(camper_id)
        
        # Start with original context or create new one
        if original_state:
            tuned_context = copy.deepcopy(original_state)
        else:
            tuned_context = RAGContextState(
                camper_id=camper_id,
                state_id="tuned",
                timestamp=datetime.now(),
                context_type=RAGContextType.TASK_SPECIFIC
            )
        
        # Apply profile-based tuning
        if profile:
            # Enhance system prompt with profile template
            if profile.system_prompt_template:
                context_vars = {
                    'task_description': task_description,
                    'original_role': tuned_context.role_definition or '',
                    'expertise_areas': ', '.join(profile.expertise_focus_areas),
                    **profile.context_variables
                }
                
                enhanced_prompt = profile.system_prompt_template.format(**context_vars)
                
                if tuned_context.system_prompt:
                    tuned_context.system_prompt = f"{tuned_context.system_prompt}\n\n{enhanced_prompt}"
                else:
                    tuned_context.system_prompt = enhanced_prompt
            
            # Update expertise areas
            tuned_context.expertise_areas.extend(profile.expertise_focus_areas)
            tuned_context.expertise_areas = list(set(tuned_context.expertise_areas))  # Remove duplicates
            
            # Update behavioral traits
            tuned_context.behavioral_traits.extend(profile.behavioral_adjustments)
            tuned_context.behavioral_traits = list(set(tuned_context.behavioral_traits))
        
        # Apply role-specific requirements
        if role_requirements:
            if 'expertise_areas' in role_requirements:
                tuned_context.expertise_areas.extend(role_requirements['expertise_areas'])
                tuned_context.expertise_areas = list(set(tuned_context.expertise_areas))
            
            if 'required_capabilities' in role_requirements:
                tuned_context.domain_knowledge['required_capabilities'] = role_requirements['required_capabilities']
            
            if 'personality_traits' in role_requirements:
                tuned_context.behavioral_traits.extend(role_requirements['personality_traits'])
                tuned_context.behavioral_traits = list(set(tuned_context.behavioral_traits))
            
            if 'context_sources' in role_requirements:
                tuned_context.domain_knowledge['context_sources'] = role_requirements['context_sources']
        
        # Add task-specific context
        tuned_context.task_context = {
            'task_description': task_description,
            'tuning_timestamp': datetime.now().isoformat(),
            'profile_used': profile.profile_id if profile else None,
            'role_requirements': role_requirements or {}
        }
        
        tuned_context.timestamp = datetime.now()
        tuned_context.context_type = RAGContextType.TASK_SPECIFIC
        
        return tuned_context
    
    def _apply_context_to_camper(self, camper: Any, context: RAGContextState) -> None:
        """Apply a context state to a camper instance."""
        if hasattr(camper, '_rag_system_prompt'):
            camper._rag_system_prompt = context.system_prompt
        
        if hasattr(camper, '_role'):
            camper._role = context.role_definition
        
        if hasattr(camper, '_expertise_areas'):
            camper._expertise_areas = context.expertise_areas.copy()
        
        if hasattr(camper, '_domain_knowledge'):
            camper._domain_knowledge = context.domain_knowledge.copy()
        
        if hasattr(camper, '_behavioral_traits'):
            camper._behavioral_traits = context.behavioral_traits.copy()
        
        if hasattr(camper, '_task_context'):
            camper._task_context = context.task_context.copy()
    
    def _select_best_profile(self, task_description: str, 
                           role_requirements: Dict[str, Any] = None) -> Optional[RAGTuningProfile]:
        """Select the best tuning profile for a given task."""
        # Simple keyword-based matching for now
        # In a more sophisticated implementation, this could use embeddings or ML
        
        task_lower = task_description.lower()
        best_profile = None
        best_score = 0
        
        for profile in self._tuning_profiles.values():
            score = 0
            
            # Check if any target task types match
            for task_type in profile.target_task_types:
                if task_type.lower() in task_lower:
                    score += 2
            
            # Check expertise area relevance
            if role_requirements and 'expertise_areas' in role_requirements:
                for expertise in role_requirements['expertise_areas']:
                    if expertise.lower() in [area.lower() for area in profile.expertise_focus_areas]:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_profile = profile
        
        return best_profile
    
    def _initialize_default_profiles(self) -> None:
        """Initialize default tuning profiles for common task types."""
        profiles = [
            RAGTuningProfile(
                profile_id="data_analysis",
                name="Data Analysis Specialist",
                description="Tuning for data analysis and statistical tasks",
                target_task_types=["data analysis", "statistics", "visualization", "reporting"],
                system_prompt_template="""
Task Focus: {task_description}

You are now specialized for data analysis tasks. Your enhanced capabilities include:
- Statistical analysis and interpretation
- Data visualization and reporting
- Pattern recognition and insights generation
- Quantitative reasoning and methodology

Expertise Areas: {expertise_areas}

Approach this task with analytical rigor and attention to data quality.
                """,
                role_enhancement_template="Enhanced for data analysis and statistical reasoning",
                expertise_focus_areas=["statistics", "data_visualization", "quantitative_analysis"],
                behavioral_adjustments=["analytical", "detail-oriented", "methodical"]
            ),
            
            RAGTuningProfile(
                profile_id="software_development",
                name="Software Development Specialist",
                description="Tuning for software development tasks",
                target_task_types=["coding", "programming", "software", "development", "debugging"],
                system_prompt_template="""
Task Focus: {task_description}

You are now specialized for software development tasks. Your enhanced capabilities include:
- Code design and architecture
- Best practices and patterns
- Testing and debugging strategies
- Performance optimization

Expertise Areas: {expertise_areas}

Approach this task with engineering best practices and clean code principles.
                """,
                role_enhancement_template="Enhanced for software development and engineering",
                expertise_focus_areas=["software_engineering", "code_quality", "testing"],
                behavioral_adjustments=["systematic", "quality-focused", "problem-solving"]
            ),
            
            RAGTuningProfile(
                profile_id="research_analysis",
                name="Research Analysis Specialist", 
                description="Tuning for research and analytical tasks",
                target_task_types=["research", "analysis", "investigation", "study"],
                system_prompt_template="""
Task Focus: {task_description}

You are now specialized for research and analysis tasks. Your enhanced capabilities include:
- Literature review and synthesis
- Critical thinking and evaluation
- Hypothesis formation and testing
- Evidence-based reasoning

Expertise Areas: {expertise_areas}

Approach this task with scientific rigor and comprehensive analysis.
                """,
                role_enhancement_template="Enhanced for research and analytical thinking",
                expertise_focus_areas=["research_methodology", "critical_analysis", "evidence_evaluation"],
                behavioral_adjustments=["thorough", "objective", "evidence-based"]
            )
        ]
        
        for profile in profiles:
            if profile.profile_id not in self._tuning_profiles:
                self._tuning_profiles[profile.profile_id] = profile
    
    def _persist_state(self, state: RAGContextState) -> None:
        """Persist a RAG context state to disk."""
        try:
            state_file = self.storage_path / f"{state.camper_id}_{state.state_id}.json"
            
            with open(state_file, 'w', encoding='utf-8') as f:
                # Convert dataclass to dict for JSON serialization
                state_dict = asdict(state)
                state_dict['timestamp'] = state.timestamp.isoformat()
                json.dump(state_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to persist RAG state: {str(e)}")
    
    def _persist_profile(self, profile: RAGTuningProfile) -> None:
        """Persist a tuning profile to disk."""
        try:
            profile_file = self.storage_path / f"profile_{profile.profile_id}.json"
            
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(profile), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to persist tuning profile: {str(e)}")
    
    def _load_persistent_data(self) -> None:
        """Load persistent states and profiles from disk."""
        try:
            # Load states
            for state_file in self.storage_path.glob("*.json"):
                if state_file.name.startswith("profile_"):
                    continue
                
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state_dict = json.load(f)
                    
                    # Convert back to dataclass
                    state_dict['timestamp'] = datetime.fromisoformat(state_dict['timestamp'])
                    state_dict['context_type'] = RAGContextType(state_dict['context_type'])
                    
                    state = RAGContextState(**state_dict)
                    
                    if state.camper_id not in self._camper_states:
                        self._camper_states[state.camper_id] = {}
                    
                    self._camper_states[state.camper_id][state.state_id] = state
                    
                    if state.state_id == "original":
                        self._original_states[state.camper_id] = state
                        
                except Exception as e:
                    logger.warning(f"Failed to load state from {state_file}: {str(e)}")
            
            # Load profiles
            for profile_file in self.storage_path.glob("profile_*.json"):
                try:
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        profile_dict = json.load(f)
                    
                    profile = RAGTuningProfile(**profile_dict)
                    self._tuning_profiles[profile.profile_id] = profile
                    
                except Exception as e:
                    logger.warning(f"Failed to load profile from {profile_file}: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"Failed to load persistent data: {str(e)}")
"""
Base Camper class for individual models or tools within a campfire.
"""

import json
import yaml
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from jinja2 import Template, Environment, FileSystemLoader

from .torch import Torch

if TYPE_CHECKING:
    from ..party_box.box_driver import BoxDriver


class Camper(ABC):
    """
    Base class for individual models or tools within a campfire.
    
    Campers collaborate to produce a single, refined output (torch).
    Each camper can load RAG templates, process prompts, and interact
    with the Party Box for asset management.
    """
    
    def __init__(self, party_box: "BoxDriver", config: Dict[str, Any]):
        """
        Initialize a camper.
        
        Args:
            party_box: Reference to the Party Box for asset storage
            config: Configuration dictionary for this camper
        """
        self.party_box = party_box
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.jinja_env = Environment(
            loader=FileSystemLoader(config.get("template_dir", "templates"))
        )
        
        # RAG document support
        self._rag_document_path = config.get("rag_document_path")
        self._rag_system_prompt = None
        self._load_rag_document()
        
        # Zeitgeist functionality
        self._zeitgeist_engine = None
        self._role = config.get("role", "general")
        self._conversation_context: List[str] = []
        self._zeitgeist_enabled = config.get("zeitgeist_enabled", True)
        
    def load_rag(self, template_path: str, **kwargs) -> str:
        """
        Load a JSON/YAML template and embed dynamic values.
        
        Args:
            template_path: Path to the template file
            **kwargs: Dynamic values to embed in the template
            
        Returns:
            Formatted prompt string
        """
        template_file = Path(template_path)
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        # Load template content
        with open(template_file, 'r', encoding='utf-8') as f:
            if template_file.suffix.lower() in ['.yaml', '.yml']:
                template_data = yaml.safe_load(f)
            elif template_file.suffix.lower() == '.json':
                template_data = json.load(f)
            else:
                # Treat as plain text template
                template_content = f.read()
                template = Template(template_content)
                return template.render(**kwargs)
        
        # Add default dynamic values
        default_values = {
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': int(time.time()),
            'camper_name': self.name,
            **kwargs
        }
        
        # If template_data is a dict, look for a 'prompt' or 'template' field
        if isinstance(template_data, dict):
            prompt_template = template_data.get('prompt', template_data.get('template', ''))
            if isinstance(prompt_template, str):
                template = Template(prompt_template)
                return template.render(**default_values)
            else:
                # Return the entire template data as JSON string
                template = Template(json.dumps(template_data, indent=2))
                return template.render(**default_values)
        else:
            # Template data is a string
            template = Template(str(template_data))
            return template.render(**default_values)
    
    def _load_rag_document(self) -> None:
        """
        Load RAG document content for system prompt if specified.
        
        This method loads the RAG document and prepares it for use as a system prompt.
        The document content will be used to define the camper's role and knowledge base.
        """
        if not self._rag_document_path:
            return
            
        try:
            rag_file = Path(self._rag_document_path)
            
            if not rag_file.exists():
                raise FileNotFoundError(f"RAG document not found: {self._rag_document_path}")
            
            # Load document content based on file type
            with open(rag_file, 'r', encoding='utf-8') as f:
                if rag_file.suffix.lower() in ['.yaml', '.yml']:
                    rag_data = yaml.safe_load(f)
                elif rag_file.suffix.lower() == '.json':
                    rag_data = json.load(f)
                else:
                    # Treat as plain text
                    rag_data = f.read()
            
            # Process RAG data for system prompt
            if isinstance(rag_data, dict):
                # Look for system_prompt, role, or instructions fields
                self._rag_system_prompt = (
                    rag_data.get('system_prompt') or 
                    rag_data.get('role') or 
                    rag_data.get('instructions') or
                    rag_data.get('persona') or
                    str(rag_data)
                )
            else:
                # Use the entire content as system prompt
                self._rag_system_prompt = str(rag_data)
                
        except Exception as e:
            # Log error but don't fail initialization
            print(f"Warning: Failed to load RAG document {self._rag_document_path}: {str(e)}")
            self._rag_system_prompt = None
    
    def get_system_prompt(self, **kwargs) -> Optional[str]:
        """
        Get the system prompt for this camper, including RAG document content.
        
        Args:
            **kwargs: Additional context variables for template rendering
            
        Returns:
            System prompt string or None if no system prompt is configured
        """
        if not self._rag_system_prompt:
            return None
            
        # Add default context variables
        context = {
            'camper_name': self.name,
            'role': self._role,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': int(time.time()),
            **kwargs
        }
        
        # Render the system prompt with context if it contains template variables
        try:
            template = Template(self._rag_system_prompt)
            return template.render(**context)
        except Exception:
            # If template rendering fails, return the raw content
            return self._rag_system_prompt
    
    def set_rag_document_path(self, path: str) -> None:
        """
        Set a new RAG document path and reload the document.
        
        Args:
            path: Path to the RAG document file
        """
        self._rag_document_path = path
        self._load_rag_document()
    
    @abstractmethod
    async def override_prompt(self, raw_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Custom API calls for this camper.
        
        Developers implement this method to integrate their existing
        model wrappers (e.g., OpenRouter, local models, APIs).
        
        Args:
            raw_prompt: The formatted prompt to process
            system_prompt: Optional system prompt from RAG document or configuration
            
        Returns:
            Dictionary containing the response and any metadata
        """
        pass
    
    async def process(self, input_torch: Optional[Torch] = None) -> Torch:
        """
        Main processing logic for the camper.
        
        Args:
            input_torch: Optional input torch from previous campfire
            
        Returns:
            Output torch with this camper's results
        """
        try:
            # Prepare context from input torch
            context = {}
            if input_torch:
                context.update({
                    'input_claim': input_torch.claim,
                    'input_path': input_torch.path,
                    'input_metadata': input_torch.metadata,
                    'input_confidence': input_torch.confidence
                })
            
            # Load and format prompt if template is specified
            prompt = ""
            if 'template_path' in self.config:
                prompt = self.load_rag(self.config['template_path'], **context)
            elif 'prompt' in self.config:
                template = Template(self.config['prompt'])
                prompt = template.render(**context)
            else:
                # Use a default prompt
                prompt = f"Process the following input: {context}"
            
            # Get system prompt from RAG document if available
            system_prompt = self.get_system_prompt(**context)
            
            # Call the custom override_prompt method
            response = await self.override_prompt(prompt, system_prompt)
            
            # Extract claim and other data from response
            claim = response.get('claim', response.get('content', str(response)))
            confidence = response.get('confidence', 1.0)
            metadata = response.get('metadata', {})
            asset_path = response.get('path')
            
            # Store any assets in Party Box if provided
            if 'asset_data' in response:
                asset_hash = await self.party_box.put(
                    f"{self.name}_{int(time.time())}", 
                    response['asset_data']
                )
                asset_path = f"./party_box/{asset_hash}"
            
            # Create output torch
            output_torch = Torch(
                claim=claim,
                path=asset_path,
                confidence=confidence,
                metadata={
                    'camper_name': self.name,
                    'processing_time': time.time(),
                    **metadata
                },
                source_campfire=self.config.get('campfire_name', 'unknown'),
                channel=self.config.get('output_channel', 'default')
            )
            
            return output_torch
            
        except Exception as e:
            # Return error torch
            error_torch = Torch(
                claim=f"Error in {self.name}: {str(e)}",
                confidence=0.0,
                metadata={
                    'error': True,
                    'error_type': type(e).__name__,
                    'camper_name': self.name
                },
                source_campfire=self.config.get('campfire_name', 'unknown'),
                channel=self.config.get('output_channel', 'default')
            )
            return error_torch
    
    async def store_asset(self, data: bytes, filename: str) -> str:
        """
        Store an asset in the Party Box.
        
        Args:
            data: Asset data as bytes
            filename: Suggested filename
            
        Returns:
            Asset hash/key for retrieval
        """
        return await self.party_box.put(filename, data)
    
    async def get_asset(self, asset_key: str):
        """
        Retrieve an asset from the Party Box.
        
        Args:
            asset_key: Asset hash/key
            
        Returns:
            Asset data or path
        """
        return await self.party_box.get(asset_key)
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def set_party_box(self, party_box: "BoxDriver") -> None:
        """
        Set the party box reference for this camper.
        
        Args:
            party_box: The party box driver instance
        """
        self.party_box = party_box
    
    def set_campfire_name(self, campfire_name: str) -> None:
        """
        Set the campfire name for this camper.
        
        Args:
            campfire_name: Name of the campfire this camper belongs to
        """
        self.campfire_name = campfire_name
    
    def set_role(self, role: str) -> None:
        """
        Set the role for this camper (affects Zeitgeist search behavior).
        
        Args:
            role: The role this camper plays (e.g., 'developer', 'designer', 'manager')
        """
        self._role = role
    
    def get_role(self) -> str:
        """
        Get the current role of this camper.
        
        Returns:
            The camper's role
        """
        return self._role
    
    def enable_zeitgeist(self, enabled: bool = True) -> None:
        """
        Enable or disable Zeitgeist functionality for this camper.
        
        Args:
            enabled: Whether to enable Zeitgeist
        """
        self._zeitgeist_enabled = enabled
    
    def add_conversation_context(self, context: str) -> None:
        """
        Add context from the current conversation for better Zeitgeist searches.
        
        Args:
            context: Context string to add
        """
        self._conversation_context.append(context)
        # Keep only the last 10 context items to avoid memory bloat
        if len(self._conversation_context) > 10:
            self._conversation_context = self._conversation_context[-10:]
    
    async def get_zeitgeist(self, 
                           topic: str, 
                           context: str = "",
                           search_types: List[str] = None) -> Dict[str, Any]:
        """
        Get current zeitgeist (opinions, trends, beliefs) about a topic
        relevant to this camper's role.
        
        Args:
            topic: The topic to search for
            context: Additional context for the search
            search_types: Types of searches to perform ('general', 'tools', 'opinions')
            
        Returns:
            Dictionary containing zeitgeist analysis
        """
        if not self._zeitgeist_enabled:
            return {
                'error': 'Zeitgeist functionality is disabled for this camper',
                'enabled': False
            }
        
        try:
            # Lazy import to avoid circular dependencies
            if self._zeitgeist_engine is None:
                from ..zeitgeist import ZeitgeistEngine
                self._zeitgeist_engine = ZeitgeistEngine()
            
            # Combine provided context with conversation context
            full_context = context
            if self._conversation_context:
                full_context = f"{context} {' '.join(self._conversation_context[-3:])}"
            
            # Get zeitgeist information
            zeitgeist_info = await self._zeitgeist_engine.get_zeitgeist(
                role=self._role,
                topic=topic,
                context=full_context.strip(),
                search_types=search_types
            )
            
            # Add this search to conversation context
            self.add_conversation_context(f"searched for {topic}")
            
            return zeitgeist_info
            
        except Exception as e:
            return {
                'error': f'Failed to get zeitgeist information: {str(e)}',
                'topic': topic,
                'role': self._role
            }
    
    async def get_role_opinions(self, topic: str) -> Dict[str, Any]:
        """
        Get opinions specifically relevant to this camper's role about a topic.
        
        Args:
            topic: The topic to get opinions about
            
        Returns:
            Dictionary containing role-specific opinions
        """
        return await self.get_zeitgeist(
            topic=topic,
            search_types=['opinions']
        )
    
    async def get_trending_tools(self, topic: str = "") -> Dict[str, Any]:
        """
        Get trending tools and methods relevant to this camper's role.
        
        Args:
            topic: Optional topic to focus the tool search
            
        Returns:
            Dictionary containing trending tools information
        """
        search_topic = f"{self._role} tools" if not topic else f"{topic} {self._role} tools"
        return await self.get_zeitgeist(
            topic=search_topic,
            search_types=['tools']
        )
    
    async def get_expert_perspectives(self, topic: str) -> Dict[str, Any]:
        """
        Get expert perspectives on a topic relevant to this camper's role.
        
        Args:
            topic: The topic to get expert perspectives on
            
        Returns:
            Dictionary containing expert perspectives
        """
        return await self.get_zeitgeist(
            topic=topic,
            context="expert professional industry leader",
            search_types=['general', 'opinions']
        )
    
    async def get_personal_outlook(self, topic: str = "") -> Dict[str, Any]:
        """
        Get this camper's personal outlook, self-perception, and current knowledge state.
        This method captures how the camper sees themselves, their role, and their understanding
        of topics before or after zeitgeist research.
        
        Args:
            topic: Optional specific topic to focus the outlook on
            
        Returns:
            Dictionary containing the camper's personal outlook and self-perception
        """
        # Create a prompt that asks the camper to reflect on their current state
        context_prompt = f"As {self.name} in the role of {self._role}, reflect on your current outlook and self-perception."
        
        if topic:
            context_prompt += f" Specifically consider your understanding and perspective on: {topic}"
        
        context_prompt += """
        
        Please provide your honest self-assessment covering:
        1. How you see yourself in your role
        2. Your current knowledge and expertise level
        3. Your main concerns and priorities
        4. Your confidence in handling relevant challenges
        5. Your perspective on the topic (if provided)
        
        Be authentic and specific to your role and personality."""
        
        # Use the override_prompt method to get the camper's response
        response = await self.override_prompt(context_prompt)
        
        return {
            'camper_name': self.name,
            'role': self._role,
            'topic': topic if topic else "general",
            'outlook_response': response,
            'timestamp': self._get_timestamp(),
            'metadata': {
                'method': 'personal_reflection',
                'context': 'self_assessment'
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for tracking purposes."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def __str__(self) -> str:
        """String representation of the camper."""
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name={self.name}, config={self.config})"


class SimpleCamper(Camper):
    """
    A simple camper implementation for testing and basic use cases.
    """
    

    
    async def override_prompt(self, raw_prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Simple implementation that echoes the prompt.
        
        Args:
            raw_prompt: The prompt to process
            system_prompt: Optional system prompt from RAG document
            
        Returns:
            Dictionary with echoed content
        """
        return {
            'claim': f"Processed: {raw_prompt}",
            'confidence': 0.8,
            'metadata': {
                'prompt_length': len(raw_prompt),
                'processing_method': 'echo'
            }
        }
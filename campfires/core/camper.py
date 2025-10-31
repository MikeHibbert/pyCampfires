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
from .graph_store import GraphStore, Node, Edge

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
        self._in_memory_rag: Dict[str, str] = {}
        self._load_rag_document()
        
        # Zeitgeist functionality
        self._zeitgeist_engine = None
        self._role = config.get("role", "general")
        self._conversation_context: List[str] = []
        self._zeitgeist_enabled = config.get("zeitgeist_enabled", True)
        self._address: Optional[str] = None # Unique address for the camper
        
        # Graph memory integration (optional)
        self._graph_store: Optional[GraphStore] = None
        self._graph_enabled: bool = config.get("graph_enabled", True)
        self._topic_aliases: Dict[str, str] = {
            **{k.lower(): v for k, v in (config.get("topic_aliases") or {}).items()}
        }

    def set_address(self, address: str) -> None:
        """
        Sets the unique address for this camper.
        """
        self._address = address

    def get_address(self) -> Optional[str]:
        """
        Returns the unique address of this camper.
        """
        return self._address

    def set_graph_store(self, graph_store: GraphStore) -> None:
        """
        Inject a GraphStore for sharing findings and retrieving shared context.
        """
        self._graph_store = graph_store

    def add_in_memory_rag(self, key: str, content: str) -> None:
        """
        Adds content to the camper's in-memory RAG storage.
        """
        self._in_memory_rag[key] = content

    def get_in_memory_rag(self, key: str) -> Optional[str]:
        """
        Retrieves content from the camper's in-memory RAG storage.
        """
        return self._in_memory_rag.get(key)

    def get_all_in_memory_rag(self) -> Dict[str, str]:
        """
        Returns all content in the camper's in-memory RAG storage.
        """
        return self._in_memory_rag

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
        Prioritizes in-memory RAG if available.
        """
        if self._in_memory_rag:
            # Combine in-memory RAG into a single system prompt or use a specific key
            self._rag_system_prompt = "\n".join(self._in_memory_rag.values())
            return

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

    def set_rag_document_path(self, path: Optional[str]) -> None:
        """
        Set a new RAG document path and reload the document.
        If path is None, it clears the document path and reloads.
        """
        self._rag_document_path = path
        self._load_rag_document()

    def set_in_memory_rag(self, rag_data: Dict[str, str]) -> None:
        """
        Sets the entire in-memory RAG storage.
        """
        self._in_memory_rag = rag_data
        self._load_rag_document()

    def clear_in_memory_rag(self) -> None:
        """
        Clears the in-memory RAG storage.
        """
        self._in_memory_rag = {}
        self._load_rag_document()

    async def review_rag(self, context: Dict[str, Any]) -> str:
        """
        Camper reviews the provided RAG document and offers feedback.
        
        Args:
            context: A dictionary containing context for the review, including the RAG document.
            
        Returns:
            A string containing the camper's review of the RAG document.
        """
        # This is a placeholder. Actual implementation will involve
        # using the LLM to review the RAG document based on the context.
        print(f"{self.name} is reviewing RAG document...")
        return f"Review from {self.name}: The RAG document seems relevant. {context.get('rag_document_content', '')}"

    async def solve(self, problem_statement: str, context: Dict[str, Any]) -> str:
        """
        Camper attempts to solve a given problem statement.
        
        Args:
            problem_statement: The problem to be solved.
            context: Additional context for solving the problem.
            
        Returns:
            A string containing the camper's proposed solution.
        """
        print(f"{self.name} is solving the problem...")
        return f"Solution from {self.name}: I propose the following solution to {problem_statement}. {context.get('additional_info', '')}"

    async def design(self, problem_statement: str, solution_proposal: str, context: Dict[str, Any]) -> str:
        """
        Camper designs a solution based on a problem statement and a solution proposal.
        
        Args:
            problem_statement: The problem statement.
            solution_proposal: The proposed solution to design from.
            context: Additional context for the design process.
            
        Returns:
            A string containing the camper's design for the solution.
        """
        print(f"{self.name} is designing the solution...")
        return f"Design from {self.name}: Based on the problem {problem_statement} and proposal {solution_proposal}, here is the design. {context.get('design_constraints', '')}"

    async def implement(self, design_document: str, context: Dict[str, Any]) -> str:
        """
        Camper implements the solution based on a design document.
        
        Args:
            design_document: The design document to implement.
            context: Additional context for the implementation process.
            
        Returns:
            A string containing the camper's implementation details or code.
        """
        print(f"{self.name} is implementing the design...")
        return f"Implementation from {self.name}: Following the design {design_document}, here is the implementation. {context.get('coding_guidelines', '')}"

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
            
            # Share process results to graph if enabled and available
            try:
                await self._maybe_share_process_result(output_torch, input_torch)
            except Exception:
                # Do not fail processing if graph sharing encounters issues
                pass
            
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

    async def _maybe_share_process_result(self, output_torch: Torch, input_torch: Optional[Torch]) -> None:
        """
        Optionally share the camper's process result to the graph for Zeitgeist-style
        collaboration across campers.
        """
        if not self._graph_enabled or self._graph_store is None:
            return
        
        # Determine topic from torch metadata or camper role
        raw_topic = (
            output_torch.metadata.get('topic')
            or (input_torch.metadata.get('topic') if input_torch else None)
            or self._role
        )
        topic = self._normalize_topic(raw_topic)
        
        await self.share_finding(
            topic=topic,
            summary=output_torch.claim,
            details={
                'confidence': output_torch.confidence,
                'metadata': output_torch.metadata,
                'path': output_torch.path,
                'source_campfire': output_torch.source_campfire,
                'channel': output_torch.channel,
            },
            tags=output_torch.metadata.get('tags', []),
        )

    async def share_finding(
        self,
        topic: str,
        summary: str,
        details: Dict[str, Any],
        importance: float = 0.5,
        tags: List[str] = None,
    ) -> None:
        """
        Share a finding/result to the graph for other campers to retrieve.
        Creates Camper, Topic, and Finding nodes, and connects them with edges.
        """
        if not self._graph_enabled or self._graph_store is None:
            return
        tags = tags or []
        
        # Normalize topic and build stable IDs
        topic = self._normalize_topic(topic)
        camper_id = self._address or f"camper:{self.config.get('campfire_name','unknown')}:{self.name}"
        topic_id = f"topic:{(topic or 'general').strip().lower()}"
        # Use timestamp for uniqueness; could be replaced with hash if needed
        ts = int(time.time())
        finding_id = f"finding:{camper_id}:{ts}"
        
        # Upsert nodes
        await self._graph_store.upsert_node(Node(
            id=camper_id,
            label="Camper",
            properties={
                'id': camper_id,
                'name': self.name,
                'role': self._role,
                'campfire': self.config.get('campfire_name', 'unknown'),
                'address': self._address,
            }
        ))
        await self._graph_store.upsert_node(Node(
            id=topic_id,
            label="Topic",
            properties={
                'id': topic_id,
                'name': topic,
            }
        ))
        await self._graph_store.upsert_node(Node(
            id=finding_id,
            label="Finding",
            properties={
                'id': finding_id,
                'summary': summary,
                'importance': importance,
                'tags': tags,
                'timestamp': ts,
                'camper': camper_id,
                'topic': topic_id,
            }
        ))
        
        # Connect nodes both ways to enable neighbor queries from topic
        await self._graph_store.upsert_edge(Edge(
            src=camper_id,
            dst=finding_id,
            type="MADE",
            properties={}
        ))
        await self._graph_store.upsert_edge(Edge(
            src=finding_id,
            dst=topic_id,
            type="ABOUT",
            properties={}
        ))
        await self._graph_store.upsert_edge(Edge(
            src=topic_id,
            dst=finding_id,
            type="HAS_FINDING",
            properties={}
        ))
        
        # Store details as node context
        await self._graph_store.add_context(finding_id, details)

    async def get_shared_findings(self, topic: str, limit: int = 10, weight_importance: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve shared findings for a topic to support Zeitgeist-informed collaboration.
        """
        if not self._graph_enabled or self._graph_store is None:
            return []
        topic_id = f"topic:{(self._normalize_topic(topic) or 'general').strip().lower()}"
        neighbors = await self._graph_store.query_neighbors(topic_id, edge_type="HAS_FINDING")
        results: List[Dict[str, Any]] = []
        for n in neighbors:
            ctx = await self._graph_store.get_context(n.id)
            item = {
                'id': n.id,
                'summary': n.properties.get('summary'),
                'importance': n.properties.get('importance'),
                'tags': n.properties.get('tags', []),
                'timestamp': n.properties.get('timestamp'),
                'context': ctx,
            }
            # Compute a simple weighted score if confidence present in context
            importance = float(item.get('importance') or 0)
            confidence = 0.0
            if isinstance(item.get('context'), dict):
                confidence = float(item['context'].get('confidence') or 0)
            score = weight_importance * importance + (1 - weight_importance) * confidence
            item['score'] = score
            results.append(item)
        # Sort by score desc and apply limit
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results[:max(0, int(limit))]

    def _normalize_topic(self, topic: Optional[str]) -> str:
        """
        Normalize topic using alias mapping. Falls back to lowercased stripped text.
        """
        if not topic:
            return "general"
        t = topic.strip().lower()
        return self._topic_aliases.get(t, t)
    
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
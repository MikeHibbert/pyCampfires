"""
ManifestLoader for Dockerfile-like YAML configuration management.

This module provides a declarative configuration system similar to Docker's approach,
allowing users to define campfire configurations, orchestration topologies, and
execution parameters using YAML manifests.
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from .orchestration import TaskComplexity
from .party_orchestrator import ExecutionTopology


logger = logging.getLogger(__name__)


@dataclass
class CampfireManifest:
    """Represents a campfire configuration manifest."""
    version: str
    name: str
    description: str
    base_config: Dict[str, Any]
    campers: List[Dict[str, Any]]
    environment: Dict[str, str]
    resources: Dict[str, Any]
    networking: Dict[str, Any]
    volumes: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationManifest:
    """Represents an orchestration configuration manifest."""
    version: str
    name: str
    description: str
    topology: str
    tasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    environment: Dict[str, str]
    resources: Dict[str, Any]
    timeout_minutes: int
    retry_policy: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PartyManifest:
    """Represents a complete party configuration with multiple campfires."""
    version: str
    name: str
    description: str
    campfires: List[str]  # References to campfire manifests
    orchestration: str    # Reference to orchestration manifest
    shared_environment: Dict[str, str]
    shared_resources: Dict[str, Any]
    networking: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ManifestLoader:
    """
    Loads and validates YAML manifests for campfire configuration.
    
    Supports Dockerfile-like syntax with instructions for:
    - FROM: Base configuration inheritance
    - ENV: Environment variables
    - COPY: File and configuration copying
    - RUN: Setup commands
    - EXPOSE: Network port exposure
    - VOLUME: Data volume mounting
    - WORKDIR: Working directory setting
    - CMD: Default execution command
    """
    
    SUPPORTED_VERSIONS = ['1.0', '1.1']
    REQUIRED_FIELDS = {
        'campfire': ['version', 'name', 'campers'],
        'orchestration': ['version', 'name', 'topology', 'tasks'],
        'party': ['version', 'name', 'campfires']
    }
    
    def __init__(self, manifest_dir: str = None, config: Dict[str, Any] = None):
        """
        Initialize the manifest loader.
        
        Args:
            manifest_dir: Directory containing manifest files
            config: Loader configuration
        """
        self.manifest_dir = Path(manifest_dir) if manifest_dir else Path.cwd()
        self.config = config or {}
        
        # Caches
        self._campfire_cache: Dict[str, CampfireManifest] = {}
        self._orchestration_cache: Dict[str, OrchestrationManifest] = {}
        self._party_cache: Dict[str, PartyManifest] = {}
        
        # Base configurations for inheritance
        self._base_configs: Dict[str, Dict[str, Any]] = {
            'minimal': {
                'resources': {'memory': 'low', 'cpu': 'low'},
                'timeout_minutes': 15,
                'max_concurrent_tasks': 1
            },
            'standard': {
                'resources': {'memory': 'medium', 'cpu': 'medium'},
                'timeout_minutes': 30,
                'max_concurrent_tasks': 3
            },
            'performance': {
                'resources': {'memory': 'high', 'cpu': 'high'},
                'timeout_minutes': 60,
                'max_concurrent_tasks': 5
            }
        }
    
    def load_campfire_manifest(self, manifest_path: str) -> CampfireManifest:
        """
        Load a campfire manifest from a YAML file.
        
        Args:
            manifest_path: Path to the manifest file
            
        Returns:
            Parsed campfire manifest
        """
        if manifest_path in self._campfire_cache:
            return self._campfire_cache[manifest_path]
        
        full_path = self.manifest_dir / manifest_path
        if not full_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
            
            # Process Dockerfile-like instructions
            processed_data = self._process_dockerfile_instructions(raw_data)
            
            # Validate manifest
            self._validate_manifest(processed_data, 'campfire')
            
            # Create manifest object
            manifest = CampfireManifest(
                version=processed_data['version'],
                name=processed_data['name'],
                description=processed_data.get('description', ''),
                base_config=processed_data.get('base_config', {}),
                campers=processed_data.get('campers', []),
                environment=processed_data.get('environment', {}),
                resources=processed_data.get('resources', {}),
                networking=processed_data.get('networking', {}),
                volumes=processed_data.get('volumes', []),
                metadata=processed_data.get('metadata', {})
            )
            
            # Cache the manifest
            self._campfire_cache[manifest_path] = manifest
            
            logger.info(f"Loaded campfire manifest: {manifest.name}")
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load campfire manifest {manifest_path}: {e}")
            raise
    
    def load_orchestration_manifest(self, manifest_path: str) -> OrchestrationManifest:
        """
        Load an orchestration manifest from a YAML file.
        
        Args:
            manifest_path: Path to the manifest file
            
        Returns:
            Parsed orchestration manifest
        """
        if manifest_path in self._orchestration_cache:
            return self._orchestration_cache[manifest_path]
        
        full_path = self.manifest_dir / manifest_path
        if not full_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
            
            # Process Dockerfile-like instructions
            processed_data = self._process_dockerfile_instructions(raw_data)
            
            # Validate manifest
            self._validate_manifest(processed_data, 'orchestration')
            
            # Create manifest object
            manifest = OrchestrationManifest(
                version=processed_data['version'],
                name=processed_data['name'],
                description=processed_data.get('description', ''),
                topology=processed_data['topology'],
                tasks=processed_data['tasks'],
                dependencies=processed_data.get('dependencies', {}),
                environment=processed_data.get('environment', {}),
                resources=processed_data.get('resources', {}),
                timeout_minutes=processed_data.get('timeout_minutes', 30),
                retry_policy=processed_data.get('retry_policy', {}),
                metadata=processed_data.get('metadata', {})
            )
            
            # Cache the manifest
            self._orchestration_cache[manifest_path] = manifest
            
            logger.info(f"Loaded orchestration manifest: {manifest.name}")
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load orchestration manifest {manifest_path}: {e}")
            raise
    
    def load_party_manifest(self, manifest_path: str) -> PartyManifest:
        """
        Load a party manifest from a YAML file.
        
        Args:
            manifest_path: Path to the manifest file
            
        Returns:
            Parsed party manifest
        """
        if manifest_path in self._party_cache:
            return self._party_cache[manifest_path]
        
        full_path = self.manifest_dir / manifest_path
        if not full_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {full_path}")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
            
            # Process Dockerfile-like instructions
            processed_data = self._process_dockerfile_instructions(raw_data)
            
            # Validate manifest
            self._validate_manifest(processed_data, 'party')
            
            # Create manifest object
            manifest = PartyManifest(
                version=processed_data['version'],
                name=processed_data['name'],
                description=processed_data.get('description', ''),
                campfires=processed_data['campfires'],
                orchestration=processed_data.get('orchestration', ''),
                shared_environment=processed_data.get('shared_environment', {}),
                shared_resources=processed_data.get('shared_resources', {}),
                networking=processed_data.get('networking', {}),
                metadata=processed_data.get('metadata', {})
            )
            
            # Cache the manifest
            self._party_cache[manifest_path] = manifest
            
            logger.info(f"Loaded party manifest: {manifest.name}")
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load party manifest {manifest_path}: {e}")
            raise
    
    def _process_dockerfile_instructions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Dockerfile-like instructions in the manifest.
        
        Args:
            data: Raw manifest data
            
        Returns:
            Processed manifest data
        """
        processed = data.copy()
        
        # Process FROM instruction for base configuration inheritance
        if 'FROM' in data:
            base_name = data['FROM']
            if base_name in self._base_configs:
                base_config = self._base_configs[base_name].copy()
                # Merge base config with current config
                for key, value in base_config.items():
                    if key not in processed:
                        processed[key] = value
                    elif isinstance(value, dict) and isinstance(processed[key], dict):
                        merged = value.copy()
                        merged.update(processed[key])
                        processed[key] = merged
            
            # Remove FROM instruction from final config
            del processed['FROM']
        
        # Process ENV instructions
        if 'ENV' in data:
            env_vars = data['ENV']
            if 'environment' not in processed:
                processed['environment'] = {}
            
            if isinstance(env_vars, dict):
                processed['environment'].update(env_vars)
            elif isinstance(env_vars, list):
                for env_item in env_vars:
                    if isinstance(env_item, dict):
                        processed['environment'].update(env_item)
                    elif isinstance(env_item, str) and '=' in env_item:
                        key, value = env_item.split('=', 1)
                        processed['environment'][key.strip()] = value.strip()
            
            del processed['ENV']
        
        # Process COPY instructions for configuration copying
        if 'COPY' in data:
            copy_instructions = data['COPY']
            if 'volumes' not in processed:
                processed['volumes'] = []
            
            for copy_item in copy_instructions:
                if isinstance(copy_item, dict):
                    processed['volumes'].append(copy_item)
                elif isinstance(copy_item, str):
                    # Parse "source:destination" format
                    if ':' in copy_item:
                        source, dest = copy_item.split(':', 1)
                        processed['volumes'].append({
                            'source': source.strip(),
                            'destination': dest.strip()
                        })
            
            del processed['COPY']
        
        # Process RUN instructions for setup commands
        if 'RUN' in data:
            run_commands = data['RUN']
            if 'setup_commands' not in processed:
                processed['setup_commands'] = []
            
            if isinstance(run_commands, list):
                processed['setup_commands'].extend(run_commands)
            else:
                processed['setup_commands'].append(run_commands)
            
            del processed['RUN']
        
        # Process EXPOSE instructions for networking
        if 'EXPOSE' in data:
            exposed_ports = data['EXPOSE']
            if 'networking' not in processed:
                processed['networking'] = {}
            if 'exposed_ports' not in processed['networking']:
                processed['networking']['exposed_ports'] = []
            
            if isinstance(exposed_ports, list):
                processed['networking']['exposed_ports'].extend(exposed_ports)
            else:
                processed['networking']['exposed_ports'].append(exposed_ports)
            
            del processed['EXPOSE']
        
        # Process VOLUME instructions
        if 'VOLUME' in data:
            volumes = data['VOLUME']
            if 'volumes' not in processed:
                processed['volumes'] = []
            
            if isinstance(volumes, list):
                for vol in volumes:
                    if isinstance(vol, str):
                        processed['volumes'].append({'destination': vol})
                    else:
                        processed['volumes'].append(vol)
            else:
                processed['volumes'].append({'destination': volumes})
            
            del processed['VOLUME']
        
        # Process WORKDIR instruction
        if 'WORKDIR' in data:
            if 'base_config' not in processed:
                processed['base_config'] = {}
            processed['base_config']['working_directory'] = data['WORKDIR']
            del processed['WORKDIR']
        
        # Process CMD instruction for default commands
        if 'CMD' in data:
            if 'base_config' not in processed:
                processed['base_config'] = {}
            processed['base_config']['default_command'] = data['CMD']
            del processed['CMD']
        
        return processed
    
    def _validate_manifest(self, data: Dict[str, Any], manifest_type: str):
        """
        Validate a manifest against required fields and constraints.
        
        Args:
            data: Manifest data to validate
            manifest_type: Type of manifest ('campfire', 'orchestration', 'party')
        """
        # Check version
        if 'version' not in data:
            raise ValueError("Manifest must specify a version")
        
        if data['version'] not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported manifest version: {data['version']}")
        
        # Check required fields
        required_fields = self.REQUIRED_FIELDS.get(manifest_type, [])
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from {manifest_type} manifest")
        
        # Type-specific validations
        if manifest_type == 'orchestration':
            self._validate_orchestration_manifest(data)
        elif manifest_type == 'campfire':
            self._validate_campfire_manifest(data)
        elif manifest_type == 'party':
            self._validate_party_manifest(data)
    
    def _validate_orchestration_manifest(self, data: Dict[str, Any]):
        """Validate orchestration-specific fields."""
        # Validate topology
        topology = data.get('topology', '').lower()
        valid_topologies = [t.value for t in ExecutionTopology]
        if topology not in valid_topologies:
            raise ValueError(f"Invalid topology '{topology}'. Must be one of: {valid_topologies}")
        
        # Validate tasks
        tasks = data.get('tasks', [])
        if not isinstance(tasks, list) or len(tasks) == 0:
            raise ValueError("Orchestration manifest must have at least one task")
        
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                raise ValueError(f"Task {i} must be a dictionary")
            if 'description' not in task:
                raise ValueError(f"Task {i} must have a 'description' field")
    
    def _validate_campfire_manifest(self, data: Dict[str, Any]):
        """Validate campfire-specific fields."""
        # Validate campers
        campers = data.get('campers', [])
        if not isinstance(campers, list) or len(campers) == 0:
            raise ValueError("Campfire manifest must have at least one camper")
        
        for i, camper in enumerate(campers):
            if not isinstance(camper, dict):
                raise ValueError(f"Camper {i} must be a dictionary")
            if 'type' not in camper:
                raise ValueError(f"Camper {i} must have a 'type' field")
    
    def _validate_party_manifest(self, data: Dict[str, Any]):
        """Validate party-specific fields."""
        # Validate campfires
        campfires = data.get('campfires', [])
        if not isinstance(campfires, list) or len(campfires) == 0:
            raise ValueError("Party manifest must reference at least one campfire")
    
    def create_sample_manifests(self, output_dir: str = None):
        """
        Create sample manifest files for reference.
        
        Args:
            output_dir: Directory to create sample files in
        """
        output_path = Path(output_dir) if output_dir else self.manifest_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample campfire manifest
        campfire_sample = {
            'version': '1.0',
            'FROM': 'standard',
            'name': 'analysis-campfire',
            'description': 'Campfire specialized for data analysis tasks',
            'ENV': {
                'MODEL_TYPE': 'meta-llama/llama-3.2-3b-instruct:free',
                'MAX_TOKENS': '2048'
            },
            'campers': [
                {
                    'type': 'dynamic',
                    'name': 'data-analyst',
                    'role': 'data_analyst',
                    'expertise': ['statistics', 'data_visualization', 'python'],
                    'capabilities': ['data_processing', 'chart_generation', 'statistical_analysis']
                }
            ],
            'resources': {
                'memory': 'medium',
                'cpu': 'medium',
                'timeout_minutes': 45
            },
            'EXPOSE': [8080, 8081],
            'VOLUME': ['/data', '/outputs'],
            'metadata': {
                'tags': ['analysis', 'data', 'statistics'],
                'author': 'campfires-team'
            }
        }
        
        # Sample orchestration manifest
        orchestration_sample = {
            'version': '1.0',
            'name': 'data-pipeline',
            'description': 'Multi-stage data analysis pipeline',
            'topology': 'sequential',
            'ENV': {
                'PIPELINE_MODE': 'production',
                'LOG_LEVEL': 'info'
            },
            'tasks': [
                {
                    'id': 'data-collection',
                    'description': 'Collect and validate input data',
                    'priority': 8,
                    'estimated_duration': 10,
                    'context': {'data_source': 'api', 'validation_rules': 'strict'}
                },
                {
                    'id': 'data-processing',
                    'description': 'Clean and transform the collected data',
                    'priority': 7,
                    'estimated_duration': 15,
                    'context': {'processing_type': 'etl', 'output_format': 'parquet'}
                },
                {
                    'id': 'analysis',
                    'description': 'Perform statistical analysis on processed data',
                    'priority': 6,
                    'estimated_duration': 20,
                    'context': {'analysis_type': 'descriptive', 'confidence_level': 0.95}
                }
            ],
            'dependencies': {
                'data-processing': ['data-collection'],
                'analysis': ['data-processing']
            },
            'timeout_minutes': 60,
            'retry_policy': {
                'max_retries': 3,
                'backoff_factor': 2
            }
        }
        
        # Sample party manifest
        party_sample = {
            'version': '1.0',
            'name': 'research-party',
            'description': 'Multi-campfire research and analysis party',
            'campfires': [
                'analysis-campfire.yaml',
                'research-campfire.yaml'
            ],
            'orchestration': 'research-pipeline.yaml',
            'shared_environment': {
                'PROJECT_NAME': 'research-project',
                'DATA_PATH': '/shared/data'
            },
            'shared_resources': {
                'memory_pool': 'large',
                'storage': 'distributed'
            },
            'networking': {
                'internal_network': 'campfire-net',
                'external_access': True
            }
        }
        
        # Write sample files
        samples = [
            ('campfire-sample.yaml', campfire_sample),
            ('orchestration-sample.yaml', orchestration_sample),
            ('party-sample.yaml', party_sample)
        ]
        
        for filename, content in samples:
            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(content, f, default_flow_style=False, indent=2)
            logger.info(f"Created sample manifest: {file_path}")
    
    def validate_manifest_file(self, manifest_path: str, manifest_type: str) -> bool:
        """
        Validate a manifest file without loading it into cache.
        
        Args:
            manifest_path: Path to the manifest file
            manifest_type: Type of manifest to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            full_path = self.manifest_dir / manifest_path
            if not full_path.exists():
                logger.error(f"Manifest file not found: {full_path}")
                return False
            
            with open(full_path, 'r', encoding='utf-8') as f:
                raw_data = yaml.safe_load(f)
            
            processed_data = self._process_dockerfile_instructions(raw_data)
            self._validate_manifest(processed_data, manifest_type)
            
            logger.info(f"Manifest {manifest_path} is valid")
            return True
            
        except Exception as e:
            logger.error(f"Manifest validation failed for {manifest_path}: {e}")
            return False
    
    def list_manifests(self, manifest_type: str = None) -> List[str]:
        """
        List available manifest files in the manifest directory.
        
        Args:
            manifest_type: Filter by manifest type (optional)
            
        Returns:
            List of manifest file paths
        """
        manifest_files = []
        
        for file_path in self.manifest_dir.glob("*.yaml"):
            if file_path.is_file():
                # Try to determine manifest type from content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    # Simple heuristic to determine manifest type
                    if manifest_type is None:
                        manifest_files.append(str(file_path.relative_to(self.manifest_dir)))
                    elif manifest_type == 'campfire' and 'campers' in data:
                        manifest_files.append(str(file_path.relative_to(self.manifest_dir)))
                    elif manifest_type == 'orchestration' and 'topology' in data:
                        manifest_files.append(str(file_path.relative_to(self.manifest_dir)))
                    elif manifest_type == 'party' and 'campfires' in data:
                        manifest_files.append(str(file_path.relative_to(self.manifest_dir)))
                        
                except Exception:
                    # Skip files that can't be parsed
                    continue
        
        return sorted(manifest_files)
    
    def clear_cache(self):
        """Clear all cached manifests."""
        self._campfire_cache.clear()
        self._orchestration_cache.clear()
        self._party_cache.clear()
        logger.info("Manifest cache cleared")
    
    def save_campfire_manifest(self, manifest: CampfireManifest, file_path: str) -> None:
        """
        Save a CampfireManifest to a YAML file.
        
        Args:
            manifest: CampfireManifest instance to save
            file_path: Path where to save the YAML file
        """
        # Convert manifest to dictionary
        manifest_dict = {
            'version': manifest.version,
            'kind': 'CampfireManifest',
            'metadata': {
                'name': manifest.name,
                'description': manifest.description,
                'created_at': datetime.utcnow().isoformat()
            },
            'spec': {
                'name': manifest.name,
                'description': manifest.description,
                'campers': manifest.campers,
                'config': manifest.config,
                'environment': manifest.environment,
                'resources': manifest.resources,
                'networking': manifest.networking,
                'volumes': manifest.volumes
            }
        }
        
        # Add optional fields if present
        if hasattr(manifest, 'party_box') and manifest.party_box:
            manifest_dict['spec']['party_box'] = manifest.party_box
        
        if hasattr(manifest, 'mcp') and manifest.mcp:
            manifest_dict['spec']['mcp'] = manifest.mcp
        
        # Ensure directory exists
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML file
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(manifest_dict, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"CampfireManifest saved to: {file_path}")
    
    def save_orchestration_manifest(self, manifest: OrchestrationManifest, file_path: str) -> None:
        """
        Save an OrchestrationManifest to a YAML file.
        
        Args:
            manifest: OrchestrationManifest instance to save
            file_path: Path where to save the YAML file
        """
        # Convert manifest to dictionary
        manifest_dict = {
            'version': manifest.version,
            'kind': 'OrchestrationManifest',
            'metadata': {
                'name': manifest.name,
                'description': manifest.description,
                'created_at': datetime.utcnow().isoformat()
            },
            'spec': {
                'name': manifest.name,
                'description': manifest.description,
                'topology': manifest.topology,
                'tasks': manifest.tasks,
                'dependencies': manifest.dependencies,
                'timeout_minutes': manifest.timeout_minutes,
                'retry_policy': manifest.retry_policy,
                'environment': manifest.environment,
                'resources': manifest.resources,
                'networking': manifest.networking,
                'volumes': manifest.volumes
            }
        }
        
        # Ensure directory exists
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML file
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(manifest_dict, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"OrchestrationManifest saved to: {file_path}")
    
    def save_party_manifest(self, manifest: PartyManifest, file_path: str) -> None:
        """
        Save a PartyManifest to a YAML file.
        
        Args:
            manifest: PartyManifest instance to save
            file_path: Path where to save the YAML file
        """
        # Convert manifest to dictionary
        manifest_dict = {
            'version': manifest.version,
            'kind': 'PartyManifest',
            'metadata': {
                'name': manifest.name,
                'description': manifest.description,
                'created_at': datetime.utcnow().isoformat()
            },
            'spec': {
                'name': manifest.name,
                'description': manifest.description,
                'campfires': manifest.campfires,
                'shared_environment': manifest.shared_environment,
                'shared_resources': manifest.shared_resources,
                'networking': manifest.networking,
                'volumes': manifest.volumes
            }
        }
        
        # Ensure directory exists
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write YAML file
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(manifest_dict, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"PartyManifest saved to: {file_path}")
    
    def save_manifest(self, manifest: Union[CampfireManifest, OrchestrationManifest, PartyManifest], 
                     file_path: str) -> None:
        """
        Save any type of manifest to a YAML file.
        
        Args:
            manifest: Manifest instance to save
            file_path: Path where to save the YAML file
        """
        if isinstance(manifest, CampfireManifest):
            self.save_campfire_manifest(manifest, file_path)
        elif isinstance(manifest, OrchestrationManifest):
            self.save_orchestration_manifest(manifest, file_path)
        elif isinstance(manifest, PartyManifest):
            self.save_party_manifest(manifest, file_path)
        else:
            raise ValueError(f"Unsupported manifest type: {type(manifest)}")
    
    def create_campfire_manifest_from_campfire(self, campfire) -> CampfireManifest:
        """
        Create a CampfireManifest from an existing Campfire instance.
        
        Args:
            campfire: Campfire instance to convert
            
        Returns:
            CampfireManifest instance
        """
        # Get YAML config from campfire
        yaml_config = campfire.to_yaml_config()
        spec = yaml_config.get('spec', {})
        metadata = yaml_config.get('metadata', {})
        
        # Create manifest
        manifest = CampfireManifest(
            version=yaml_config.get('version', '1.0'),
            name=spec.get('name', campfire.name),
            description=metadata.get('description', f'Manifest for {campfire.name}'),
            campers=spec.get('campers', []),
            config=spec.get('config', {}),
            environment=spec.get('environment', {}),
            resources=spec.get('resources', {}),
            networking=spec.get('networking', {}),
            volumes=spec.get('volumes', [])
        )
        
        return manifest
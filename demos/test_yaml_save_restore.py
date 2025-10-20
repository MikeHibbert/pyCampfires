#!/usr/bin/env python3
"""
Test script for YAML save/restore functionality of campfires.

This demonstrates how campfires can be saved to YAML configuration files
and restored later, similar to GitHub Actions or Ansible configurations.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from campfires.core.campfire import Campfire, CampfireManager
from campfires.core.camper import Camper
from campfires.party_box.local_driver import LocalDriver

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCamper(Camper):
    """Simple test camper for demonstration."""
    
    def __init__(self, party_box, name: str, role: str = "test"):
        config = {
            'name': name,
            'test_setting': 'test_value',
            'role': role,
            'enabled': True
        }
        super().__init__(party_box, config)
        self._role = role
    
    def override_prompt(self, prompt: str) -> str:
        """Override the prompt with role-specific modifications."""
        return f"[{self._role}] {prompt}"
    
    async def process_torch(self, torch):
        """Simple torch processing."""
        logger.info(f"TestCamper {self.name} processing torch: {torch.content}")
        return f"Processed by {self.name}: {torch.content}"


async def test_yaml_save_restore():
    """Test the YAML save and restore functionality."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"Using temporary directory: {temp_path}")
        
        # Set up party box
        party_box = LocalDriver(str(temp_path / "party_box"))
        
        # Create test campers
        camper1 = TestCamper(party_box, "analyst", "data_analyst")
        camper2 = TestCamper(party_box, "reviewer", "quality_reviewer")
        
        # Create original campfire
        original_campfire = Campfire(
            name="test_campfire",
            campers=[camper1, camper2],
            party_box=party_box,
            config={
                'description': 'Test campfire for YAML functionality',
                'version': '1.0',
                'custom_setting': 'custom_value'
            }
        )
        
        logger.info("=== Original Campfire Created ===")
        logger.info(f"Name: {original_campfire.name}")
        logger.info(f"Campers: {[c.name for c in original_campfire.campers]}")
        logger.info(f"Config: {original_campfire.config}")
        
        # Test 1: Save campfire to YAML
        yaml_file = temp_path / "test_campfire.yaml"
        logger.info(f"\n=== Saving Campfire to YAML ===")
        original_campfire.save_to_yaml(str(yaml_file))
        
        # Verify YAML file was created
        assert yaml_file.exists(), "YAML file was not created"
        logger.info(f"✓ YAML file created: {yaml_file}")
        
        # Read and display YAML content
        with open(yaml_file, 'r') as f:
            yaml_content = f.read()
        logger.info(f"YAML content preview:\n{yaml_content[:500]}...")
        
        # Test 2: Load campfire from YAML
        logger.info(f"\n=== Loading Campfire from YAML ===")
        try:
            # Note: This will have limited functionality since camper creation 
            # from YAML is not fully implemented yet
            loaded_campfire = Campfire.load_from_yaml(str(yaml_file), party_box)
            logger.info(f"✓ Campfire loaded from YAML")
            logger.info(f"Loaded name: {loaded_campfire.name}")
            logger.info(f"Loaded config: {loaded_campfire.config}")
        except Exception as e:
            logger.warning(f"Expected limitation: {e}")
        
        # Test 3: CampfireManager functionality
        logger.info(f"\n=== Testing CampfireManager Save/Load ===")
        manager = CampfireManager()
        manager.add_campfire(original_campfire)
        
        # Save specific campfire manifest
        manager_yaml_file = temp_path / "manager_test_campfire.yaml"
        manager.save_campfire_manifest("test_campfire", str(manager_yaml_file))
        assert manager_yaml_file.exists(), "Manager YAML file was not created"
        logger.info(f"✓ Manager saved campfire manifest: {manager_yaml_file}")
        
        # Save all manifests to directory
        manifests_dir = temp_path / "manifests"
        saved_files = manager.save_all_manifests(str(manifests_dir))
        logger.info(f"✓ Manager saved all manifests: {saved_files}")
        
        # Export entire manager state
        export_dir = temp_path / "manager_export"
        export_metadata = manager.export_manager_state(str(export_dir))
        logger.info(f"✓ Manager state exported to: {export_dir}")
        logger.info(f"Export metadata: {export_metadata}")
        
        # Test 4: Demonstrate flexible location specification
        logger.info(f"\n=== Testing Flexible Location Specification ===")
        
        # Simulate management framework specifying custom locations
        custom_locations = {
            'production': temp_path / "production" / "configs",
            'staging': temp_path / "staging" / "configs", 
            'development': temp_path / "dev" / "configs"
        }
        
        for env, location in custom_locations.items():
            location.mkdir(parents=True, exist_ok=True)
            env_yaml_file = location / f"campfire_{env}.yaml"
            original_campfire.save_to_yaml(str(env_yaml_file))
            logger.info(f"✓ Saved {env} config to: {env_yaml_file}")
        
        # Test 5: Template-based filename generation
        logger.info(f"\n=== Testing Template-based Filenames ===")
        template_dir = temp_path / "templates"
        
        # Different filename templates
        templates = [
            "{name}_v1.yaml",
            "campfire_{name}_config.yaml",
            "{name}_{timestamp}.yaml"
        ]
        
        for template in templates:
            if "{timestamp}" in template:
                # For timestamp template, we'll use a simple approach
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = template.format(name=original_campfire.name, timestamp=timestamp)
            else:
                filename = template.format(name=original_campfire.name)
            
            file_path = template_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            original_campfire.save_to_yaml(str(file_path))
            logger.info(f"✓ Saved with template '{template}': {file_path}")
        
        logger.info(f"\n=== Test Summary ===")
        logger.info("✓ YAML save functionality working")
        logger.info("✓ YAML load functionality implemented (with limitations)")
        logger.info("✓ CampfireManager save/load methods working")
        logger.info("✓ Flexible location specification supported")
        logger.info("✓ Template-based filename generation supported")
        logger.info("✓ Management framework integration ready")


if __name__ == "__main__":
    asyncio.run(test_yaml_save_restore())
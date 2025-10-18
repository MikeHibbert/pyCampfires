#!/usr/bin/env python3
"""
Simple test to verify Camper class initialization.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import campfires
sys.path.append(str(Path(__file__).parent.parent))

from campfires.core.camper import Camper
from campfires.party_box.local_driver import LocalDriver

class SimpleCamper(Camper):
    """A simple test camper."""
    
    def __init__(self, party_box, config):
        super().__init__(party_box, config)
    
    async def override_prompt(self, raw_prompt: str, system_prompt: str = None) -> dict:
        return {"system": "test", "user": raw_prompt}
    
    async def process(self, torch):
        return torch

def test_simple_camper():
    """Test basic camper initialization."""
    print("Creating LocalDriver...")
    party_box = LocalDriver()
    
    print("Creating config...")
    config = {
        "name": "SimpleCamper",
        "template_dir": "templates",
        "rag_document_path": "C:\\Users\\Mike\\Documents\\Python\\Campfires\\demos\\rag_examples\\medical_expert.yaml"
    }
    
    print("Creating SimpleCamper...")
    camper = SimpleCamper(party_box, config)
    
    print(f"Camper created successfully!")
    print(f"Has _rag_system_prompt: {hasattr(camper, '_rag_system_prompt')}")
    if hasattr(camper, '_rag_system_prompt'):
        print(f"_rag_system_prompt: {camper._rag_system_prompt}")

if __name__ == "__main__":
    test_simple_camper()
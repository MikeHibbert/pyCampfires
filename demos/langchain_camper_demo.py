#!/usr/bin/env python3
"""
LangChain-backed Camper Demo (feature-flagged)

Prerequisites:
- Optional: pip install langchain langchain-ollama
- Ollama running at http://localhost:11434 (default)

Enable via:
- Env: set CAMPFIRES_ENABLE_LANGCHAIN=1
- Or config: {'enable_langchain': True}
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import campfires
sys.path.append(str(Path(__file__).parent.parent))

from campfires.core.factory import CampfireFactory
from campfires.party_box.local_driver import LocalDriver
from campfires.core.orchestration import SubTask, RoleRequirement
from campfires.core.torch import Torch


async def main():
    # Enable the feature flag via env if desired
    os.environ.setdefault('CAMPFIRES_ENABLE_LANGCHAIN', '1')

    party_box = LocalDriver()

    # Factory config: you can also set enable_langchain=True here
    factory = CampfireFactory(
        party_box=party_box,
        config={
            'enable_langchain': True,
            'ollama_base_url': 'http://localhost:11434',
            'ollama_model': 'llama3',
        }
    )

    # Define a minimal subtask and role requirement
    subtask = SubTask(
        id='demo-subtask-1',
        description='Generate a short greeting using LangChain-backed camper',
        priority=1,
        estimated_effort=1,
        required_capabilities=['task_processing'],
        metadata={},
    )

    role = RoleRequirement(
        role_name='langchain_greeter',
        expertise_areas=['general'],
        required_capabilities=['task_processing'],
        personality_traits=['friendly'],
        context_sources=[],
    )

    # Create a campfire for the subtask
    instance_id = await factory.create_campfire_for_subtask(subtask, role)

    # Prepare an input torch
    input_torch = Torch(
        claim='Say hello to the world in one sentence.',
        confidence=0.5,
        metadata={'demo': 'langchain_camper'},
        source_campfire='demo',
        channel='demo',
    )

    # Process the torch
    result = await factory.process_torch_in_campfire(instance_id, input_torch)
    if result:
        print('Output torch:', result.claim)
        print('Metadata:', result.metadata)
    else:
        print('No output torch produced.')

    # Cleanup
    await factory.terminate_campfire(instance_id)
    await factory.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
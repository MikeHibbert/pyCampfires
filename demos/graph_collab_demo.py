"""
Graph Collaboration Demo
------------------------

This demo spins up a simple Campfire with two campers that
collaborate via the shared GraphStore. The first camper publishes
findings about a topic; the second camper retrieves those findings
to inform its own processing.

Prerequisites
- Python 3.10+
- Optional: Neo4j running per docker-compose.neo4j.yml (otherwise uses in-memory graph)

Run
python -m demos.graph_collab_demo
"""

import asyncio
from typing import Any, Dict, Optional

from campfires.core.torch import Torch
from campfires.core.camper import Camper
from campfires.core.campfire import Campfire
from campfires.core.graph_store import InMemoryGraphStore


class DesignCamper(Camper):
    async def override_prompt(self, torch: Torch) -> str:
        # Publish a design outline for the torch topic
        topic = torch.metadata.get("topic") or "general"
        outline = f"Design outline for {topic}: goals, components, data flow, risks."
        # Share with importance and tags; include confidence in context
        await self.share_finding(
            topic=topic,
            summary=outline,
            importance=0.8,
            tags=["design", "outline"],
            details={"confidence": 0.75, "source": "DesignCamper"},
        )
        return outline


class QACamper(Camper):
    async def override_prompt(self, torch: Torch) -> str:
        topic = torch.metadata.get("topic") or "general"
        findings = await self.get_shared_findings(topic=topic, limit=5)
        if findings:
            top = findings[0]
            context_summary = top.get("summary")
            return f"QA plan for {topic}, informed by design: {context_summary}"
        return f"QA plan for {topic}: baseline tests, edge cases, performance."


async def main() -> None:
    # Use in-memory graph by default; Neo4j is supported via Factory
    graph = InMemoryGraphStore()

    # Create a campfire with two collaborating campers
    design = DesignCamper(name="designer", role="architect", config={"graph_enabled": True})
    qa = QACamper(name="qa", role="tester", config={"graph_enabled": True})
    campfire = Campfire(name="collab-demo", campers=[design, qa], graph_store=graph)

    # Build a torch with a topic; both campers will process it sequentially
    torch = Torch.claim(
        claim="Implement Feature X",
        metadata={"topic": "feature-x"},
    )

    # Process: DesignCamper publishes finding, QACamper consumes it
    result = await campfire.process_torch(torch)

    print("=== Collaboration Demo Results ===")
    for name, out in result.results.items():
        print(f"[{name}] {out.message}")


if __name__ == "__main__":
    asyncio.run(main())
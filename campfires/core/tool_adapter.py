from typing import Any, Dict, Optional


class ToolAdapter:
    async def invoke(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

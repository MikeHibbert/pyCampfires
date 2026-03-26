from typing import Any, Dict, Optional
from dataclasses import dataclass
from .torch import Torch


@dataclass
class SecurityHookResult:
    action: str
    torch: Optional[Torch] = None
    reason: Optional[str] = None


class SecurityHooks:
    async def pre_receive_torch(self, torch: Torch, context: Dict[str, Any]) -> SecurityHookResult:
        return SecurityHookResult(action="allow", torch=torch)

    async def pre_send_torch(self, torch: Torch, context: Dict[str, Any]) -> SecurityHookResult:
        return SecurityHookResult(action="allow", torch=torch)

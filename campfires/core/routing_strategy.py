from typing import Any, Dict, Optional
from dataclasses import dataclass
from .torch import Torch


@dataclass
class TargetDecision:
    channel: Optional[str] = None
    metadata: Dict[str, Any] = None


class RoutingStrategy:
    async def choose_target(self, torch: Torch, session: Optional[Dict[str, Any]] = None) -> Optional[TargetDecision]:
        return None

    async def update_health(self, target: str, status: str) -> None:
        return None


class BasicRuleBasedRouting(RoutingStrategy):
    async def choose_target(self, torch: Torch, session: Optional[Dict[str, Any]] = None) -> Optional[TargetDecision]:
        return None

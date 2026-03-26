from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ChannelEvent:
    event_id: str
    type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class SendResult:
    success: bool
    event_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class ChannelAdapter:
    async def connect(self, config: Dict[str, Any]) -> None:
        return None

    async def receive(self) -> ChannelEvent:
        raise NotImplementedError

    async def send(self, target: str, message: Dict[str, Any]) -> SendResult:
        raise NotImplementedError

    async def ack(self, event_id: str) -> None:
        return None

    async def fail(self, event_id: str, reason: str) -> None:
        return None

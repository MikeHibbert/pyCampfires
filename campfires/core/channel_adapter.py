from typing import Any, Dict, Optional


class ChannelEvent:
    def __init__(self, event_type: str, data: Dict[str, Any], channel: Optional[str] = None):
        self.event_type = event_type
        self.data = data
        self.channel = channel


class SendResult:
    def __init__(self, ok: bool, message_id: Optional[str] = None, error: Optional[str] = None):
        self.ok = ok
        self.message_id = message_id
        self.error = error


class ChannelAdapter:
    async def connect(self, config: Dict[str, Any]) -> None:
        return None

    async def receive(self) -> Optional[ChannelEvent]:
        return None

    async def send(self, target: str, message: Dict[str, Any]) -> SendResult:
        return SendResult(ok=True)

    async def ack(self, event_id: str) -> None:
        return None

    async def fail(self, event_id: str, reason: str) -> None:
        return None

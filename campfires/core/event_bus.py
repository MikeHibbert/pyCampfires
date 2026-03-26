from typing import Any, Callable, Dict, List, Optional, Tuple
import asyncio


class Event:
    def __init__(self, event_type: str, payload: Dict[str, Any]):
        self.event_type = event_type
        self.payload = payload


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Any]]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        await self._queue.put(Event(event_type, payload))

    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)

    async def _loop(self) -> None:
        while self._running:
            try:
                event = await self._queue.get()
                handlers = self._subscribers.get(event.event_type, [])
                for handler in handlers:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
            except asyncio.CancelledError:
                break
            except Exception:
                continue

from typing import Awaitable, Callable, Dict
import asyncio


class AsyncScheduler:
    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for t in list(self._tasks.values()):
            t.cancel()
        self._tasks.clear()

    async def register_interval(self, task_id: str, interval_seconds: float, callback: Callable[[], Awaitable[None]]) -> None:
        if not self._running:
            await self.start()

        async def _runner():
            while self._running:
                try:
                    await callback()
                except Exception:
                    pass
                await asyncio.sleep(interval_seconds)

        task = asyncio.create_task(_runner())
        self._tasks[task_id] = task

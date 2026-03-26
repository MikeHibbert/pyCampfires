from typing import Any, Dict, Optional
import asyncio


class Session:
    def __init__(self, session_id: str, data: Optional[Dict[str, Any]] = None):
        self.session_id = session_id
        self.data = data or {}


class SessionManager:
    async def create(self, session_id: str, data: Optional[Dict[str, Any]] = None) -> Session:
        return Session(session_id, data or {})

    async def get(self, session_id: str) -> Optional[Session]:
        return None

    async def update(self, session_id: str, data: Dict[str, Any]) -> Optional[Session]:
        return None

    async def close(self, session_id: str) -> None:
        return None


class InMemorySessionManager(SessionManager):
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self, session_id: str, data: Optional[Dict[str, Any]] = None) -> Session:
        async with self._lock:
            sess = Session(session_id, data or {})
            self._sessions[session_id] = sess
            return sess

    async def get(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def update(self, session_id: str, data: Dict[str, Any]) -> Optional[Session]:
        async with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return None
            sess.data.update(data or {})
            return sess

    async def close(self, session_id: str) -> None:
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

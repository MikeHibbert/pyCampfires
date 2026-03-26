from typing import Any, Callable, Dict, Optional


class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, Any] = {}
        self._active: Optional[str] = None
        self._failover: Optional[Callable[[str, Exception], Optional[str]]] = None

    def register(self, name: str, provider: Any) -> None:
        self._providers[name] = provider
        if self._active is None:
            self._active = name

    def set_active(self, name: str) -> None:
        if name in self._providers:
            self._active = name

    def get_active(self) -> Optional[Any]:
        if self._active is None:
            return None
        return self._providers.get(self._active)

    def set_failover(self, callback: Callable[[str, Exception], Optional[str]]) -> None:
        self._failover = callback

    def failover(self, error: Exception) -> Optional[Any]:
        if not self._failover or self._active is None:
            return None
        target = self._failover(self._active, error)
        if target and target in self._providers:
            self._active = target
            return self._providers[target]
        return None

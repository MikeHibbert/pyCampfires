import asyncio
from typing import Optional

# Optional LangChain import guarded to avoid hard dependency
try:
    from langchain_community.chat_models import ChatOllama  # type: ignore
    from langchain_core.messages import HumanMessage  # type: ignore
except Exception:
    ChatOllama = None  # type: ignore
    HumanMessage = None  # type: ignore


class LangChainAdapter:
    """
    Minimal adapter around LangChain for simple prompt completions.
    Currently supports the Ollama backend to keep local defaults intact.
    """

    def __init__(self, provider: str = "ollama", model: Optional[str] = None, base_url: Optional[str] = None):
        if provider != "ollama":
            raise ValueError("LangChainAdapter currently supports only 'ollama' provider")
        if ChatOllama is None or HumanMessage is None:
            raise ImportError("LangChain is not installed or missing Ollama integration. Install 'langchain' and 'langchain-community'.")

        self._model = model or "gemma3"
        self._base_url = base_url or "http://localhost:11434"
        self._client = ChatOllama(model=self._model, base_url=self._base_url)

    def generate(self, prompt: str) -> str:
        """Synchronous generation using LangChain's ChatOllama."""
        msg = HumanMessage(content=prompt)
        result = self._client.invoke([msg])
        # result is an AIMessage with .content
        return getattr(result, "content", str(result))

    async def async_generate(self, prompt: str) -> str:
        """Async wrapper to avoid blocking event loop when LangChain client is sync."""
        return await asyncio.to_thread(self.generate, prompt)
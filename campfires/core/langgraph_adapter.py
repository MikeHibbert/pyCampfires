from typing import Dict, Any, Optional, Callable

try:
    # Optional: LangGraph integration
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception:
    StateGraph = None  # type: ignore
    END = None  # type: ignore


class LangGraphAdapter:
    """
    Minimal adapter to run a simple LangGraph workflow when available.

    This scaffolds a single-node graph that invokes a provided callable to
    perform LLM completion, keeping Ollama as the default backend via
    existing campers or the LangChainAdapter when configured.
    """

    def __init__(self, llm_callable: Callable[[str, Optional[str]], str]):
        if StateGraph is None:
            raise ImportError("langgraph not installed. Install with `pip install langgraph`. ")
        self.llm_callable = llm_callable

        def node_fn(state: Dict[str, Any]) -> Dict[str, Any]:
            prompt = state.get("prompt", "")
            system = state.get("system", None)
            output = self.llm_callable(prompt, system)
            return {"output": output}

        graph = StateGraph(dict)
        graph.add_node("llm_node", node_fn)
        graph.set_entry_point("llm_node")
        graph.add_edge("llm_node", END)
        self._graph = graph.compile()

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        result = self._graph.invoke({"prompt": prompt, "system": system_prompt})
        return {"text": result.get("output", "")}
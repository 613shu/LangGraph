"""Build and compile the spam-cleaner graph."""
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from nodes import classify, execute, fetch, review, route_empty, route_more
from state import CleanerState

RECURSION_LIMIT = 1000   # each round = 4 steps; the first run may hold hundreds of messages


def _is_rate_limit(exc: Exception) -> bool:
    """Retry only on Gemini quota errors (HTTP 429 / RESOURCE_EXHAUSTED)."""
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


# ---------- 4.1 build_graph ----------

def build_graph(checkpointer):
    """fetch -> classify -> review (interrupt) -> execute, looping back to fetch while messages remain."""
    builder = StateGraph(CleanerState)

    builder.add_node("fetch", fetch)
    builder.add_node("classify", classify,
                     retry_policy=RetryPolicy(max_attempts=4, initial_interval=5.0,
                                              retry_on=_is_rate_limit))
    builder.add_node("review", review)
    builder.add_node("execute", execute)

    builder.add_edge(START, "fetch")
    builder.add_conditional_edges("fetch", route_empty, {"empty": END, "classify": "classify"})
    builder.add_edge("classify", "review")
    builder.add_edge("review", "execute")
    builder.add_conditional_edges("execute", route_more, {"more": "fetch", "done": END})

    return builder.compile(checkpointer=checkpointer)

"""Build and compile the LangGraph state graph for Qing.

Exports:
- get_graph() → compiled LangGraph with in-memory checkpoint persistence
- INTENT_HANDLERS → maps intent string to async streaming generator
"""
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.state import QingState
from app.graph.nodes.route import route_intent
from app.graph.nodes.retrieve import do_retrieve, do_rerank, do_compress
from app.config import CHECKPOINT_DIR

logger = logging.getLogger(__name__)

# ── Module-level compiled graph singleton ──
_qing_graph = None


def _build_graph() -> StateGraph:
    """Build the Qing LangGraph workflow.

    Graph structure:
        START → route ─┬─ question → retrieve → rerank → compress → END
                        └─ (all other intents) → END

    The answer generation happens outside the graph for streaming support.
    After the graph finishes, the endpoint streams tokens and then calls
    graph.aupdate_state() to persist the final response.
    """
    wf = StateGraph(QingState)

    # ── Nodes ──
    wf.add_node("route", route_intent)
    wf.add_node("retrieve", do_retrieve)
    wf.add_node("rerank", do_rerank)
    wf.add_node("compress", do_compress)

    # ── Entry ──
    wf.set_entry_point("route")

    # ── Conditional edges based on intent ──
    wf.add_conditional_edges(
        "route",
        _route_by_intent,
        {
            "question": "retrieve",
            "end": END,
        },
    )

    # ── RAG pipeline for questions ──
    wf.add_edge("retrieve", "rerank")
    wf.add_edge("rerank", "compress")
    wf.add_edge("compress", END)

    return wf


def _route_by_intent(state: QingState) -> str:
    """Route to appropriate graph branch based on intent.

    Only 'question' intent goes through the RAG pipeline.
    All other intents (summary, mindmap, plan, etc.) exit the graph
    and are handled by external streaming handlers.
    """
    intent = state.get("intent", "question")
    if intent == "question":
        return "question"
    return "end"


def get_graph():
    """Get or create the compiled LangGraph singleton with checkpoint persistence.

    Uses InMemorySaver for async-compatible state persistence between turns.
    Messages accumulate via the add_messages reducer automatically.
    State is lost on server restart (sessions.json retains UI metadata).
    """
    global _qing_graph
    if _qing_graph is None:
        graph = _build_graph()
        checkpointer = InMemorySaver()
        _qing_graph = graph.compile(checkpointer=checkpointer)
        logger.info("LangGraph compiled with InMemory checkpoint persistence")
    return _qing_graph


# ── Intent-to-handler mapping for FastAPI dispatch ──
# These async generators do the actual LLM generation + streaming.
# They receive the state (from graph checkpoint) and yield SSE chunks.

from app.graph.nodes.answer import answer_stream
from app.graph.nodes.summary import generate_summary
from app.graph.nodes.generate import (
    generate_mindmap,
    generate_plan,
    generate_practice,
    generate_flashcards,
    generate_formulas,
    generate_compare,
    generate_mnemonic,
    generate_sprint,
)

INTENT_HANDLERS = {
    "question": answer_stream,
    "summary": generate_summary,
    "mindmap": generate_mindmap,
    "plan": generate_plan,
    "practice": generate_practice,
    "flashcard": generate_flashcards,
    "formula": generate_formulas,
    "compare": generate_compare,
    "mnemonic": generate_mnemonic,
    "sprint": generate_sprint,
}


# ── Intent-to-handler mapping for FastAPI dispatch ──
# These async generators do the actual LLM generation + streaming.
# They receive the state (from graph checkpoint) and yield SSE chunks.

from app.graph.nodes.answer import answer_stream
from app.graph.nodes.summary import generate_summary
from app.graph.nodes.generate import (
    generate_mindmap,
    generate_plan,
    generate_practice,
    generate_flashcards,
    generate_formulas,
    generate_compare,
    generate_mnemonic,
    generate_sprint,
)

INTENT_HANDLERS = {
    "question": answer_stream,
    "summary": generate_summary,
    "mindmap": generate_mindmap,
    "plan": generate_plan,
    "practice": generate_practice,
    "flashcard": generate_flashcards,
    "formula": generate_formulas,
    "compare": generate_compare,
    "mnemonic": generate_mnemonic,
    "sprint": generate_sprint,
}

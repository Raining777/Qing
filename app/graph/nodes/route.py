"""Route intent, auto-detect course, and expand follow-up queries with conversation context."""
import logging
from langchain_core.messages import HumanMessage

from app.graph.state import QingState
from app.services.vectordb import list_courses, find_course_for_query
from app.services.embedder import get_embedder

logger = logging.getLogger(__name__)

# ── Follow-up indicators for detecting short contextual questions ──
_FOLLOW_UP_MARKERS = [
    "那", "这", "它", "这个", "那个", "这些", "那些",
    "上面", "前面", "刚才", "刚刚", "之前", "上次",
    "怎么用", "怎么做", "为什么", "然后呢", "还有呢",
    "再", "还有", "继续", "接着", "详细", "具体",
    "举例", "比如", "例如", "能再", "多说",
    "what about", "how about", "and", "then", "also", "so",
    "explain more", "tell me more", "elaborate", "go on",
    "why", "how", "when", "where", "which",
]


def _get_recent_topic(messages, last_n: int = 6) -> str:
    """Extract the most recent discussion topic from conversation history.

    Returns a short summary string or empty string.
    """
    if not messages:
        return ""

    recent = []
    for m in messages[-last_n:]:
        if isinstance(m, dict):
            content = m.get("content", "")
            role = m.get("role", "user")
        elif hasattr(m, "content"):
            content = m.content or ""
            role = "user" if getattr(m, "type", "") == "human" else "assistant"
        else:
            continue
        # Truncate long messages for the context summary
        recent.append(f"[{role}]: {content[:200]}")

    return " | ".join(recent)


def _is_follow_up(query: str) -> bool:
    """Check if the query looks like a follow-up question that needs context."""
    ql = query.lower().strip()
    # Very short queries are almost always follow-ups
    if len(query) < 15:
        return True
    # Check for follow-up indicators
    for marker in _FOLLOW_UP_MARKERS:
        if marker in ql:
            return True
    return False


def _expand_query(query: str, messages) -> str:
    """Expand short follow-up queries using recent conversation context.

    For retrieval purposes, a bare "怎么用" needs to become something like
    "在[二叉树遍历]的上下文中: 怎么用", so the vector search can match.

    Returns the expanded query, or the original if no expansion is needed.
    """
    if not _is_follow_up(query):
        return query

    ctx = _get_recent_topic(messages, last_n=6)
    if not ctx:
        return query

    # Build an expanded query that carries the conversation context
    # Format: inject topic keywords from history
    expanded = f"对话上下文: {ctx}\n用户追问: {query}"
    logger.info(f"Query expanded: [{query[:60]}] → [{expanded[:150]}...]")
    return expanded


async def route_intent(state: QingState) -> dict:
    """Determine intent and auto-detect course if needed.

    For question intent with conversation history, expands short follow-up
    queries so the retrieval step can find relevant documents.
    """
    intent = state.get("intent", "question")
    messages = state.get("messages", [])

    # Prefer the current request query. Fall back to message history only when
    # a caller did not provide query explicitly.
    query = (state.get("query") or "").strip()
    if not query:
        for m in reversed(messages):
            if isinstance(m, dict):
                if m.get("role") == "user":
                    query = (m.get("content") or "").strip()
                    break
            elif hasattr(m, "content") and getattr(m, "type", "") == "human":
                query = (m.content or "").strip()
                break

    # ── Query expansion for follow-up questions ──
    expanded_query = query
    if intent == "question" and query and messages:
        expanded_query = _expand_query(query, messages)

    # ── Auto-detect course ──
    target_course = state.get("target_course", "")
    available = list_courses()

    if intent == "question" and not target_course and query and available:
        embedder = get_embedder()
        q_emb = embedder.encode_query(query)
        target_course = find_course_for_query(q_emb)

    return {
        "query": query,
        "expanded_query": expanded_query,
        "target_course": target_course or "Other",
        "available_courses": available,
    }

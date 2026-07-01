"""Chat router: RAG Q&A with SSE streaming and LangGraph checkpoint persistence.

Architecture:
1. graph.ainvoke() runs route → retrieve → rerank → compress through the graph
   with SQLite checkpointing via thread_id. Messages accumulate automatically.
2. answer_stream() streams tokens to the client (outside graph — for SSE).
3. graph.aupdate_state() saves the final assistant response to the checkpoint.
4. Lightweight session metadata stored in JSON for the sidebar session list.
"""
import json
import traceback
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.builder import INTENT_HANDLERS, get_graph
from app.graph.state import QingState
from app.services.vectordb import list_courses
from app.config import get_default_provider, DATA_DIR

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

# ── Lightweight session metadata (sidebar UI only, not messages/state) ──
_SESSIONS_FILE = DATA_DIR / "sessions.json"
_sessions: dict[str, dict] = {}
_sessions_loaded = False


def _load_sessions() -> dict:
    try:
        if _SESSIONS_FILE.exists():
            return json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_sessions(sessions: dict):
    try:
        _SESSIONS_FILE.write_text(
            json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _ensure_sessions_loaded():
    global _sessions, _sessions_loaded
    if not _sessions_loaded:
        _sessions = _load_sessions()
        _sessions_loaded = True


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    course: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


@router.post("/chat")
async def chat(req: ChatRequest):
    """RAG Q&A with SSE streaming response.

    Uses LangGraph for the retrieval pipeline with automatic state persistence.
    Multi-turn conversations work because:
    - The graph checkpoint accumulates messages via add_messages reducer
    - route.py expands short follow-up queries with conversation context
    - answer.py sees up to 10 turns of history for coherent replies
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is required")
    if len(question) > 10000:
        raise HTTPException(400, "Question too long (max 10000 characters)")

    sid = req.session_id or uuid.uuid4().hex[:12]
    graph = get_graph()
    config = {"configurable": {"thread_id": sid}}

    # ── Load existing checkpoint to preserve persistent fields ──
    existing_state: dict = {}
    try:
        checkpoint = await graph.aget_state(config)
        if checkpoint and checkpoint.values:
            existing_state = checkpoint.values
    except Exception:
        pass  # First turn for this thread — no checkpoint yet

    # ── Build state for this turn ──
    # messages: HumanMessage gets appended to checkpoint history by add_messages
    # target_course: preserved from checkpoint unless explicitly overridden
    # chapter_summaries: always preserved (cache from summary generation)
    state: QingState = {
        "messages": [HumanMessage(content=question)],
        "session_id": sid,
        "intent": "question",
        "provider_id": req.provider or existing_state.get("provider_id") or get_default_provider(),
        "model_name": req.model or existing_state.get("model_name", ""),
        "query": question,
        "expanded_query": question,
        "target_course": req.course or existing_state.get("target_course", ""),
        "retrieved_docs": [],
        "reranked_docs": [],
        "compressed_context": "",
        "response": "",
        "sources": [],
        "token_usage": {},
        "available_courses": list_courses(),
        "practice_type": "mixed",
        "practice_count": 5,
        "practice_topic": "",
        "generated_questions": [],
        "user_answers": [],
        "review_chapter": "",
        "review_rating": 0,
        "concept_a": "",
        "concept_b": "",
        "exam_date": "",
        "sprint_course": "",
        "chapter_summaries": existing_state.get("chapter_summaries", {}),
    }

    # ── Run RAG pipeline through the graph ──
    # The graph: route (expand query, detect course) → retrieve → rerank → compress
    # All intermediate state is checkpointed automatically.
    try:
        pipeline_result = await graph.ainvoke(state, config)
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"RAG pipeline error: {e}")

    handler = INTENT_HANDLERS["question"]

    async def event_stream():
        # ── Progress notifications ──
        yield f"data: {json.dumps({'status': '正在分析意图...'})}\n\n"
        yield f"data: {json.dumps({'status': '正在检索知识库...'})}\n\n"
        yield f"data: {json.dumps({'status': '正在排序...'})}\n\n"
        yield f"data: {json.dumps({'status': '正在提取关键信息...'})}\n\n"
        yield f"data: {json.dumps({'status': '正在生成回答...'})}\n\n"

        # ── Stream the answer ──
        final_response = ""
        try:
            async for chunk in handler(pipeline_result):
                yield f"data: {json.dumps(chunk, default=str)}\n\n"
                if "response" in chunk:
                    final_response = chunk["response"]
        except Exception as e:
            final_response = f"生成回答时出错：{e}"
            yield f"data: {json.dumps({'response': final_response})}\n\n"

        # ── Persist assistant response to checkpoint ──
        # add_messages reducer appends this AIMessage to the conversation history
        try:
            await graph.aupdate_state(
                config,
                {
                    "messages": [AIMessage(content=final_response)],
                    "response": final_response,
                },
            )
        except Exception:
            pass  # Non-fatal: checkpoint update failure doesn't break the response

        # ── Update lightweight session metadata (sidebar UI) ──
        try:
            _ensure_sessions_loaded()
            existing = _sessions.get(sid, {})
            _sessions[sid] = {
                "id": sid,
                "preview": question[:50],
                "course": pipeline_result.get("target_course", ""),
                "created": existing.get("created", str(uuid.uuid4())),
            }
            _save_sessions(_sessions)
        except Exception:
            pass

        yield f"data: {json.dumps({'done': True, 'session_id': sid})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/regenerate")
async def regenerate(req: ChatRequest):
    """Regenerate the last answer (re-runs the same query)."""
    return await chat(req)


# ── Session management (lightweight metadata for sidebar UI) ──

@router.get("/sessions")
async def list_sessions():
    """List saved conversations from the lightweight metadata store."""
    _ensure_sessions_loaded()
    return sorted(
        [
            {
                "id": sid,
                "preview": s.get("preview", ""),
                "course": s.get("course", ""),
            }
            for sid, s in _sessions.items()
        ],
        key=lambda x: x["id"],
        reverse=True,
    )[:50]


@router.post("/sessions/new")
async def new_session():
    """Create a new conversation (generates a fresh thread_id)."""
    _ensure_sessions_loaded()
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = {
        "id": sid,
        "preview": "New conversation",
        "course": "",
        "created": str(uuid.uuid4()),
    }
    _save_sessions(_sessions)
    return {"session_id": sid}


@router.delete("/sessions/{sid}")
async def delete_session(sid: str):
    """Delete a conversation from the UI list."""
    _ensure_sessions_loaded()
    _sessions.pop(sid, None)
    _save_sessions(_sessions)
    # Note: LangGraph SQLite checkpointer doesn't expose a simple
    # delete-by-thread API, so checkpoint data is retained on disk.
    # A new session with the same thread_id would resume the old state.
    return {"ok": True}

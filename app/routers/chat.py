"""Chat router — unified /api/chat endpoint with SSE streaming."""
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.intents import detect_intent
from app.pipeline import run_rag_pipeline
from app.generators import INTENT_HANDLERS
from app.history import get_sessions
from app.config import get_default_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str = ""
    question: str
    course: str = ""
    provider: str = ""
    model: str = ""
    # Action-specific fields
    practice_count: int = 5
    practice_topic: str = ""
    practice_type: str = "mixed"
    concept_a: str = ""
    concept_b: str = ""
    exam_date: str = ""


def _require_course(course: str) -> str:
    course = (course or "").strip()
    if not course or len(course) > 200:
        raise HTTPException(400, "Valid course name is required")
    return course


@router.post("/chat")
async def chat(req: ChatRequest):
    """Unified endpoint: intent detection → dispatch → SSE stream.

    - question intent → RAG pipeline (retrieve + answer)
    - other intents → direct LLM generation
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question is required")
    if len(question) > 10000:
        raise HTTPException(400, "Question too long (max 10000 characters)")

    store = get_sessions()
    sid = req.session_id or store.create(preview=question[:50])
    provider = req.provider or get_default_provider()

    # Detect intent
    intent = detect_intent(question)

    async def event_stream():
        # Yield progress status
        if intent == "question":
            yield f"data: {json.dumps({'status': '正在检索知识库...'})}\n\n"

        try:
            if intent == "question":
                # ── RAG pipeline ──
                async for chunk in run_rag_pipeline(
                    query=question,
                    course=req.course,
                    session_id=sid,
                    provider_id=provider,
                    model_name=req.model,
                ):
                    yield f"data: {json.dumps(chunk, default=str)}\n\n"

            elif intent in INTENT_HANDLERS:
                # ── Action generators ──
                handler = INTENT_HANDLERS[intent]
                course = req.course or store.get_course(sid)
                _require_course(course)

                kwargs = dict(course=course, provider_id=provider, model_name=req.model, session_id=sid)
                # Add intent-specific params
                if intent == "practice":
                    kwargs.update(topic=req.practice_topic, count=req.practice_count, qtype=req.practice_type)
                elif intent == "plan":
                    kwargs["exam_date"] = req.exam_date
                elif intent == "compare":
                    kwargs.update(concept_a=req.concept_a, concept_b=req.concept_b)
                elif intent == "mnemonic":
                    kwargs = dict(query=question, provider_id=provider, model_name=req.model)
                elif intent == "sprint":
                    kwargs["course"] = req.course or store.get_course(sid)
                    kwargs["session_id"] = sid

                async for chunk in handler(**kwargs):
                    yield f"data: {json.dumps(chunk, default=str)}\n\n"

                # Cache last generation for export
                if "response" in chunk:
                    _cache_generation(intent, course, chunk["response"])

                # Save to session history
                store.add_message(sid, "user", question)
                if "response" in chunk:
                    store.add_message(sid, "assistant", chunk["response"])

            else:
                yield f"data: {json.dumps({'error': f'Unknown intent: {intent}'})}\n\n"

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"

        yield f"data: {json.dumps({'done': True, 'session_id': sid})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Session management ──

@router.get("/sessions")
async def list_sessions():
    return get_sessions().list_sessions()


@router.post("/sessions/new")
async def new_session():
    sid = get_sessions().create()
    return {"session_id": sid}


@router.delete("/sessions/{sid}")
async def delete_session(sid: str):
    get_sessions().delete(sid)
    return {"ok": True}


# ── Generation cache for export endpoints ──
_gen_cache: dict[str, str] = {}


def _cache_generation(intent: str, course: str, content: str):
    if content and len(content) > 100:
        _gen_cache[f"{intent}:{course}"] = content


def get_cached(intent: str, course: str) -> str:
    return _gen_cache.get(f"{intent}:{course}", "")


def clear_cache():
    _gen_cache.clear()

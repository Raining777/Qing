"""Actions router: export endpoints + clear. Core action endpoints now
go through /api/chat with intent detection. This file keeps export/download
and backward-compatible direct endpoints for quick-action buttons."""
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response

from app.generators import INTENT_HANDLERS
from app.history import get_sessions
from app.config import get_default_provider
from app.services.vectordb import get_course_chunks_text, clear_all
from app.routers.chat import get_cached, clear_cache, _require_course

import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["actions"])


# ── Stream helper ──

async def _stream_action(intent: str, **kwargs):
    """Common SSE streaming wrapper for action endpoints."""
    handler = INTENT_HANDLERS.get(intent)
    if not handler:
        yield f"data: {json.dumps({'error': f'Unknown intent: {intent}'})}\n\n"
        return

    async for chunk in handler(**kwargs):
        yield f"data: {json.dumps(chunk, default=str)}\n\n"

    course = kwargs.get("course", "")
    final_response = chunk.get("response", "") if "response" in chunk else ""

    # Cache for export
    if final_response and course:
        from app.routers.chat import _cache_generation
        _cache_generation(intent, course, final_response)

    # Save to session
    sid = kwargs.get("session_id", "")
    if sid and "query" in kwargs:
        get_sessions().add_message(sid, "user", kwargs.get("query", ""))
    if sid and final_response:
        get_sessions().add_message(sid, "assistant", final_response)

    yield f"data: {json.dumps({'done': True})}\n\n"


# ── Direct action endpoints (backward-compatible with quick-action buttons) ──

@router.post("/summary")
async def summary(req: dict):
    course = _require_course(req.get("course", ""))
    sid = req.get("session_id", "")
    return StreamingResponse(
        _stream_action("summary", course=course,
                       provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""), session_id=sid),
        media_type="text/event-stream",
    )


@router.post("/mindmap")
async def mindmap(req: dict):
    course = _require_course(req.get("course", ""))
    sid = req.get("session_id", "")
    return StreamingResponse(
        _stream_action("mindmap", course=course,
                       provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""), session_id=sid),
        media_type="text/event-stream",
    )


@router.post("/plan")
async def plan(req: dict):
    course = _require_course(req.get("course", ""))
    sid = req.get("session_id", "")
    return StreamingResponse(
        _stream_action("plan", course=course,
                       provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""), session_id=sid,
                       exam_date=req.get("exam_date", "")),
        media_type="text/event-stream",
    )


@router.post("/practice")
async def practice(req: dict):
    course = _require_course(req.get("course", ""))
    count = req.get("practice_count", 5)
    if not isinstance(count, int) or count < 1 or count > 20:
        count = 5
    return StreamingResponse(
        _stream_action("practice", course=course,
                       provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""),
                       topic=req.get("practice_topic", ""),
                       count=count, qtype=req.get("practice_type", "mixed")),
        media_type="text/event-stream",
    )


@router.post("/flashcards")
async def flashcards(req: dict):
    course = _require_course(req.get("course", ""))
    sid = req.get("session_id", "")
    return StreamingResponse(
        _stream_action("flashcard", course=course,
                       provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""), session_id=sid),
        media_type="text/event-stream",
    )


@router.post("/formulas")
async def formulas(req: dict):
    course = _require_course(req.get("course", ""))
    sid = req.get("session_id", "")
    return StreamingResponse(
        _stream_action("formula", course=course,
                       provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""), session_id=sid),
        media_type="text/event-stream",
    )


@router.post("/compare")
async def compare(req: dict):
    course = _require_course(req.get("course", ""))
    if not req.get("concept_a") or not req.get("concept_b"):
        raise HTTPException(400, "concept_a and concept_b are required")
    return StreamingResponse(
        _stream_action("compare", course=course,
                       provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""),
                       concept_a=req.get("concept_a", ""),
                       concept_b=req.get("concept_b", "")),
        media_type="text/event-stream",
    )


@router.post("/mnemonic")
async def mnemonic(req: dict):
    query = (req.get("query") or "").strip()
    if not query or len(query) > 1000:
        raise HTTPException(400, "Valid query is required")
    return StreamingResponse(
        _stream_action("mnemonic", provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""), query=query),
        media_type="text/event-stream",
    )


@router.post("/sprint")
async def sprint(req: dict):
    course = _require_course(req.get("course") or req.get("sprint_course", ""))
    sid = req.get("session_id", "")
    return StreamingResponse(
        _stream_action("sprint", course=course,
                       provider_id=req.get("provider", get_default_provider()),
                       model_name=req.get("model", ""), session_id=sid),
        media_type="text/event-stream",
    )


# ── Export endpoints ──

@router.get("/export/summary")
async def export_summary(course: str):
    cached = get_cached("summary", course)
    if cached:
        return Response(content=cached, media_type="text/markdown",
                        headers={"Content-Disposition": f"attachment; filename={course}_summary.md"})
    text = get_course_chunks_text(course)
    if not text:
        raise HTTPException(404, "No content found. Generate a summary first.")
    return Response(content=text[:100000], media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename={course}_summary.md"})


@router.get("/export/plan")
async def export_plan(course: str):
    cached = get_cached("plan", course)
    if not cached:
        raise HTTPException(404, "No plan generated yet.")
    return Response(content=cached, media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename={course}_plan.md"})


@router.get("/export/flashcards")
async def export_flashcards(course: str):
    cached = get_cached("flashcard", course)
    if not cached:
        raise HTTPException(404, "No flashcards generated yet.")
    return Response(content=cached, media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={course}_flashcards.csv"})


@router.get("/export/formulas")
async def export_formulas(course: str):
    cached = get_cached("formula", course)
    if not cached:
        raise HTTPException(404, "No formula sheet generated yet.")
    return Response(content=cached, media_type="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename={course}_formulas.md"})


# ── Clear all ──

@router.post("/clear")
async def clear_all_data():
    from app.services.upload_queue import get_upload_queue
    clear_all()
    queue = get_upload_queue()
    queue._seen_md5.clear()
    clear_cache()
    return {"ok": True}

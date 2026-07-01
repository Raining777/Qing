"""Actions router: summary, mindmap, plan, practice, flashcard, formula, compare, mnemonic, sprint.

All actions now integrate with the LangGraph checkpointer so that:
- target_course persists across intents
- chapter_summaries from summary generation are cached for mindmap/flashcard/formula
- session context is available for follow-up questions after an action
"""
import json
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.builder import INTENT_HANDLERS, get_graph
from app.graph.state import QingState
from app.services.vectordb import list_courses
from app.config import get_default_provider

router = APIRouter(prefix="/api", tags=["actions"])

# ── Generation cache for export endpoints ──
_gen_cache: dict[str, str] = {}


def _cache_key(intent: str, course: str) -> str:
    return f"{intent}:{course}"


def _cache_generation(intent: str, course: str, content: str):
    if content and len(content) > 100:
        _gen_cache[_cache_key(intent, course)] = content


def _get_cached(intent: str, course: str) -> Optional[str]:
    return _gen_cache.get(_cache_key(intent, course))


async def _ensure_graph_state(
    intent: str,
    session_id: str,
    provider: str,
    model: str,
    course: str = "",
    **extra,
) -> dict:
    """Run the graph to create/update a checkpoint for this session + intent.

    Returns (state, config) tuple for downstream use.
    Even though non-question intents exit the graph at route → END,
    running through the graph ensures the checkpoint is initialized
    and target_course / chapter_summaries are preserved across turns.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Load existing state from checkpoint
    existing: dict = {}
    try:
        cp = await graph.aget_state(config)
        if cp and cp.values:
            existing = cp.values
    except Exception:
        pass

    # Build state
    state: QingState = {
        "messages": [],
        "session_id": session_id,
        "intent": intent,
        "provider_id": provider or existing.get("provider_id") or get_default_provider(),
        "model_name": model or existing.get("model_name", ""),
        "query": extra.get("query", ""),
        "expanded_query": extra.get("query", ""),
        "target_course": course or existing.get("target_course", ""),
        "retrieved_docs": [],
        "reranked_docs": [],
        "compressed_context": "",
        "response": "",
        "sources": [],
        "token_usage": {},
        "available_courses": list_courses(),
        "practice_type": extra.get("practice_type", "mixed"),
        "practice_count": extra.get("practice_count", 5),
        "practice_topic": extra.get("practice_topic", ""),
        "generated_questions": [],
        "user_answers": [],
        "review_chapter": extra.get("review_chapter", ""),
        "review_rating": extra.get("review_rating", 0),
        "concept_a": extra.get("concept_a", ""),
        "concept_b": extra.get("concept_b", ""),
        "exam_date": extra.get("exam_date", ""),
        "sprint_course": extra.get("sprint_course", ""),
        "chapter_summaries": existing.get("chapter_summaries", {}),
    }

    # Run graph (for non-question intents: route → END, just saves checkpoint)
    try:
        result = await graph.ainvoke(state, config)
    except Exception:
        result = state  # Fallback: use raw state if graph fails

    return {"state": result, "config": config}


async def _stream_action(intent: str, state: QingState, config: dict = None):
    """Common SSE streaming wrapper for action endpoints.

    Streams the action result and persists chapter_summaries
    to the graph checkpoint for cross-intent reuse.
    """
    handler = INTENT_HANDLERS.get(intent)
    if not handler:
        yield f"data: {json.dumps({'error': f'Unknown intent: {intent}'})}\n\n"
        return

    final_response = ""
    chapter_summaries_update = None

    async for chunk in handler(state):
        yield f"data: {json.dumps(chunk, default=str)}\n\n"
        if "response" in chunk:
            final_response = chunk["response"]
        # Summary handler yields chapter_summaries at the end
        if "chapter_summaries" in chunk:
            chapter_summaries_update = chunk["chapter_summaries"]

    # Cache for export
    course = state.get("target_course", "") or state.get("sprint_course", "")
    if final_response and course:
        _cache_generation(intent, course, final_response)

    # Persist to graph checkpoint
    if config:
        try:
            graph = get_graph()
            update = {"response": final_response}
            if chapter_summaries_update:
                # Merge with existing summaries to avoid losing
                # summaries from other courses already in the checkpoint
                existing_summaries = state.get("chapter_summaries", {})
                merged = {**existing_summaries, **chapter_summaries_update}
                update["chapter_summaries"] = merged
            await graph.aupdate_state(config, update)
        except Exception:
            pass

    yield f"data: {json.dumps({'done': True})}\n\n"


def _require_course(req: dict) -> str:
    course = (req.get("course") or "").strip()
    if not course or len(course) > 200:
        raise HTTPException(400, "Valid course name is required")
    return course


# ── Summary ──
@router.post("/summary")
async def summary(req: dict):
    course = _require_course(req)
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state("summary", sid, req.get("provider", ""), req.get("model", ""), course=course)
    return StreamingResponse(
        _stream_action("summary", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Mindmap ──
@router.post("/mindmap")
async def mindmap(req: dict):
    course = _require_course(req)
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state("mindmap", sid, req.get("provider", ""), req.get("model", ""), course=course)
    return StreamingResponse(
        _stream_action("mindmap", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Study Plan ──
@router.post("/plan")
async def plan(req: dict):
    course = _require_course(req)
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state(
        "plan", sid, req.get("provider", ""), req.get("model", ""),
        course=course, exam_date=req.get("exam_date", ""),
    )
    return StreamingResponse(
        _stream_action("plan", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Practice ──
@router.post("/practice")
async def practice(req: dict):
    course = _require_course(req)
    count = req.get("practice_count", 5)
    if not isinstance(count, int) or count < 1 or count > 20:
        count = 5
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state(
        "practice", sid, req.get("provider", ""), req.get("model", ""),
        course=course, practice_count=count, practice_topic=req.get("practice_topic", ""),
    )
    return StreamingResponse(
        _stream_action("practice", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Flashcards ──
@router.post("/flashcards")
async def flashcards(req: dict):
    course = _require_course(req)
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state("flashcard", sid, req.get("provider", ""), req.get("model", ""), course=course)
    return StreamingResponse(
        _stream_action("flashcard", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Formula Sheet ──
@router.post("/formulas")
async def formulas(req: dict):
    course = _require_course(req)
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state("formula", sid, req.get("provider", ""), req.get("model", ""), course=course)
    return StreamingResponse(
        _stream_action("formula", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Compare ──
@router.post("/compare")
async def compare(req: dict):
    course = _require_course(req)
    if not req.get("concept_a") or not req.get("concept_b"):
        raise HTTPException(400, "concept_a and concept_b are required")
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state(
        "compare", sid, req.get("provider", ""), req.get("model", ""),
        course=course, concept_a=req.get("concept_a", ""), concept_b=req.get("concept_b", ""),
    )
    return StreamingResponse(
        _stream_action("compare", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Mnemonic ──
@router.post("/mnemonic")
async def mnemonic(req: dict):
    query = (req.get("query") or "").strip()
    if not query or len(query) > 1000:
        raise HTTPException(400, "Valid query is required")
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state("mnemonic", sid, req.get("provider", ""), req.get("model", ""), query=query)
    return StreamingResponse(
        _stream_action("mnemonic", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Sprint ──
@router.post("/sprint")
async def sprint(req: dict):
    course = (req.get("course") or req.get("sprint_course") or "").strip()
    if not course:
        raise HTTPException(400, "Course is required")
    sid = req.get("session_id", uuid.uuid4().hex[:12])
    result = await _ensure_graph_state(
        "sprint", sid, req.get("provider", ""), req.get("model", ""),
        course=course, sprint_course=course,
    )
    return StreamingResponse(
        _stream_action("sprint", result["state"], result["config"]),
        media_type="text/event-stream",
    )


# ── Clear all ──
@router.post("/clear")
async def clear_all():
    """Clear all courses and data."""
    from app.services.vectordb import clear_all
    from app.services.upload_queue import get_upload_queue
    clear_all()
    queue = get_upload_queue()
    queue._seen_md5.clear()
    _gen_cache.clear()
    return {"ok": True}


# ── Export ──

@router.get("/export/summary")
async def export_summary(course: str):
    """Download knowledge summary as Markdown."""
    cached = _get_cached("summary", course)
    if cached:
        return Response(
            content=cached, media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={course}_summary.md"},
        )
    from app.services.vectordb import get_course_chunks_text
    text = get_course_chunks_text(course)
    if not text:
        raise HTTPException(404, "No content found. Generate a summary first.")
    return Response(
        content=text[:100000], media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={course}_summary.md"},
    )


@router.get("/export/plan")
async def export_plan(course: str):
    """Download study plan as Markdown."""
    cached = _get_cached("plan", course)
    if cached:
        return Response(
            content=cached, media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={course}_plan.md"},
        )
    raise HTTPException(404, "No plan generated yet. Generate a study plan first.")


@router.get("/export/flashcards")
async def export_flashcards(course: str):
    """Download flashcards as CSV (Anki-compatible)."""
    cached = _get_cached("flashcard", course)
    if cached:
        return Response(
            content=cached, media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={course}_flashcards.csv"},
        )
    raise HTTPException(404, "No flashcards generated yet. Generate flashcards first.")


@router.get("/export/formulas")
async def export_formulas(course: str):
    """Download formula sheet as Markdown."""
    cached = _get_cached("formula", course)
    if cached:
        return Response(
            content=cached, media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename={course}_formulas.md"},
        )
    raise HTTPException(404, "No formula sheet generated yet. Generate formulas first.")

"""Upload router: file upload, progress tracking, course management."""
import asyncio
import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB
from app.services.upload_queue import get_upload_queue
from app.services.classifier import classify_files
from app.services.llm import create_llm
from app.services.vectordb import list_courses, delete_course, delete_file_from_course

import json

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload multiple files. Saves them and starts async processing."""
    if not files:
        raise HTTPException(400, "No files uploaded")

    saved_paths = []
    saved_names = []

    for f in files:
        if not f.filename:
            continue

        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        # Save file
        file_id = uuid.uuid4().hex[:8]
        safe_name = f"{file_id}_{f.filename}"
        save_path = UPLOAD_DIR / safe_name

        content = await f.read()
        # Check size
        mb = len(content) / (1024 * 1024)
        if mb > MAX_FILE_SIZE_MB:
            continue

        save_path.write_bytes(content)
        saved_paths.append(str(save_path))
        saved_names.append(f.filename)

    if not saved_paths:
        raise HTTPException(400, "No valid files uploaded")

    # Classify files (1 API call)
    llm = create_llm()
    course_map = await classify_files(saved_paths, llm)

    # Enqueue for processing
    queue = get_upload_queue()
    await queue.enqueue(saved_paths, course_map)

    return {
        "ok": True,
        "files": [{"name": n, "course": course_map.get(n, "Other")} for n in saved_names],
        "total": len(saved_paths),
    }


@router.get("/upload/progress")
async def upload_progress():
    """SSE endpoint for real-time upload processing progress."""
    queue = get_upload_queue()

    async def event_stream():
        async for progress in queue.subscribe():
            yield f"data: {json.dumps(progress)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/courses")
async def get_courses():
    """List all courses with stats."""
    return list_courses()


@router.put("/courses/{name}")
async def rename_course(name: str, new_name: str):
    """Rename a course (merge collections)."""
    # Read old chunks, add to new collection, delete old
    from app.services.vectordb import get_course_chunks_with_meta, add_chunks, delete_course
    from app.services.embedder import get_embedder

    old_chunks = get_course_chunks_with_meta(name)
    if old_chunks:
        embedder = get_embedder()
        texts = [c["text"] for c in old_chunks]
        metas = [c["meta"] for c in old_chunks]
        embs = embedder.encode_documents(texts)
        add_chunks(new_name, embs, texts, metas)

    delete_course(name)
    return {"ok": True, "name": new_name}


@router.post("/courses/merge")
async def merge_courses(sources: list[str], target: str):
    """Merge multiple courses into one."""
    from app.services.vectordb import get_course_chunks_with_meta, add_chunks, delete_course
    from app.services.embedder import get_embedder

    embedder = get_embedder()
    for src in sources:
        if src == target:
            continue
        chunks = get_course_chunks_with_meta(src)
        if chunks:
            texts = [c["text"] for c in chunks]
            metas = [c["meta"] for c in chunks]
            embs = embedder.encode_documents(texts)
            add_chunks(target, embs, texts, metas)
        delete_course(src)

    return {"ok": True, "target": target}


@router.delete("/courses/{name}")
async def remove_course(name: str):
    """Delete a course and all its data."""
    # Also delete associated upload files
    from app.services.vectordb import get_course_chunks_with_meta
    chunks = get_course_chunks_with_meta(name)
    files_to_check = set()
    for c in chunks:
        if c.get("meta", {}).get("file"):
            files_to_check.add(c["meta"]["file"])

    delete_course(name)
    return {"ok": True}


@router.delete("/courses/{name}/files/{file_id}")
async def remove_file_from_course(name: str, file_id: str):
    """Remove a single file from a course."""
    delete_file_from_course(name, file_id)
    return {"ok": True}

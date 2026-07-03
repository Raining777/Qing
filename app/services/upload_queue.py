"""Async upload queue — processes files one by one, pushes progress via SSE."""
import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

from app.services.parser import parse_file, file_md5, file_size_ok
from app.services.chunker import chunk_document
from app.services.embedder import get_embedder
from app.services.vectordb import add_chunks
from app.services.classifier import classify_files
from app.services.llm import create_llm

logger = logging.getLogger(__name__)


class UploadQueue:
    """Manages file upload processing queue with progress tracking."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._progress: dict = {"done": 0, "total": 0, "current": "", "step": "", "phase": "idle"}
        self._results: dict = {}  # filename -> {ok, error, etc}
        self._listeners: list[asyncio.Queue] = []  # SSE listeners
        self._task: asyncio.Task = None
        self._seen_md5: set[str] = set()

    async def enqueue(self, file_paths: list[str], course_map: dict[str, str]):
        """Add files to queue. course_map: {filename: course_name}"""
        for fp in file_paths:
            name = Path(fp).name
            course = course_map.get(name, "Other")
            # Deduplicate
            try:
                md5 = file_md5(fp)
                if md5 in self._seen_md5:
                    self._results[name] = {"ok": True, "course": course, "skipped": True, "reason": "duplicate"}
                    continue
                self._seen_md5.add(md5)
            except Exception:
                pass

            await self._queue.put({"path": fp, "name": name, "course": course})

        # Start processing if not already running
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._process_all())

    async def _process_all(self):
        """Process all queued files sequentially."""
        total = self._queue.qsize()
        done = 0
        self._update_progress(done, total, "", "starting", "processing")

        while not self._queue.empty():
            item = await self._queue.get()
            self._update_progress(done, total + done, item["name"], "parsing", "processing")

            try:
                await self._process_one(item)
                self._results[item["name"]] = {"ok": True, "course": item["course"]}
            except Exception as e:
                logger.error(f"Failed to process {item['name']}: {e}")
                self._results[item["name"]] = {"ok": False, "course": item["course"], "error": str(e)}

            done += 1
            self._update_progress(done, total + done, item["name"], "done", "processing")

        self._update_progress(done, total, "", "complete", "complete")
        logger.info(f"Upload processing complete: {done} files")

    async def _process_one(self, item: dict):
        """Process a single file: parse -> chunk -> embed -> store."""
        file_path = item["path"]
        name = item["name"]
        course = item["course"]

        # Step 1: Parse
        self._update_progress(self._progress["done"], self._progress["total"], name, "parsing", "processing")
        from app.services.parser import get_file_type, parse_file as do_parse, parse_file_async, parse_scanned_pdf
        ft = get_file_type(file_path)

        # Use async parser for types that need Vision (images, scanned PDFs)
        if ft in ("image",):
            llm = create_llm()
            result = await parse_file_async(file_path, llm)
        elif ft == "text_pdf":
            # Check if scanned first
            result = do_parse(file_path)
            if result.get("is_scanned"):
                self._update_progress(self._progress["done"], self._progress["total"], name, "ocr_scanning", "processing")
                llm = create_llm()
                ocr_text = await parse_scanned_pdf(file_path, llm)
                if ocr_text and ocr_text.strip():
                    result["text"] = ocr_text
        else:
            result = do_parse(file_path)

        if not result["ok"]:
            raise Exception(result.get("error", "Parse failed"))

        # Step 2: Chunk
        self._update_progress(self._progress["done"], self._progress["total"], name, "chunking", "processing")
        from app.services.chunker import chunk_text
        chunks = chunk_text(result["text"], {"file": name, "page": result.get("page", ""), "type": result["type"]})
        if not chunks:
            return

        # Step 3: Embed
        self._update_progress(self._progress["done"], self._progress["total"], name, "embedding", "processing")
        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        metas = [c["meta"] for c in chunks]
        embeddings = embedder.encode_documents(texts)

        # Step 4: Store
        self._update_progress(self._progress["done"], self._progress["total"], name, "storing", "processing")
        add_chunks(course, embeddings, texts, metas)

    def _update_progress(self, done: int, total: int, current: str, step: str, phase: str):
        self._progress = {"done": done, "total": total, "current": current, "step": step, "phase": phase}
        # Notify SSE listeners
        for q in self._listeners:
            try:
                q.put_nowait(self._progress.copy())
            except asyncio.QueueFull:
                pass

    async def subscribe(self) -> AsyncIterator[dict]:
        """SSE generator for progress updates."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._listeners.append(q)
        # Send current state immediately
        await q.put(self._progress.copy())
        try:
            while True:
                try:
                    update = await asyncio.wait_for(q.get(), timeout=30)
                    yield update
                except asyncio.TimeoutError:
                    yield self._progress.copy()  # heartbeat
        finally:
            self._listeners.remove(q)

    def get_progress(self) -> dict:
        return self._progress.copy()

    def get_results(self) -> dict:
        return self._results.copy()


# Global instance
_upload_queue: UploadQueue = None


def get_upload_queue() -> UploadQueue:
    global _upload_queue
    if _upload_queue is None:
        _upload_queue = UploadQueue()
    return _upload_queue

"""ChromaDB vector store with per-course collections."""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
import numpy as np

from app.config import CHROMA_DIR

logger = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_bm25_cache: dict[str, tuple[int, object]] = {}  # collection_name -> (chunk_count, BM25Okapi)
_course_centroids: dict[str, np.ndarray] = {}    # course_name -> mean embedding vector


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def collection_name(course: str) -> str:
    """Convert course name to valid ChromaDB collection name."""
    # Replace spaces/special chars, lowercase
    safe = "".join(c if c.isalnum() else "_" for c in course.lower())
    return f"course__{safe}"


def list_collections() -> list:
    """Return list of collection objects in ChromaDB."""
    client = _get_client()
    return list(client.list_collections())


def list_courses() -> list[dict]:
    """Return list of courses with file/chunk counts."""
    client = _get_client()
    courses = []
    for col in client.list_collections():
        count = col.count()
        if count == 0:
            continue
        # Extract course name from collection name
        name = col.name.replace("course__", "").replace("_", " ").title()
        files = set()
        metadatas = col.get(include=["metadatas"])
        if metadatas and metadatas["metadatas"]:
            for m in metadatas["metadatas"]:
                if m and "file" in m:
                    files.add(m["file"])
        courses.append({
            "name": name,
            "collection": col.name,
            "file_count": len(files),
            "chunk_count": count,
            "files": sorted(files),
        })
    return courses


def get_course_chunks_text(course: str) -> str:
    """Get all chunk text for a course (concatenated for summary generation)."""
    client = _get_client()
    cname = collection_name(course)
    try:
        col = client.get_collection(cname)
        result = col.get(include=["documents"])
        if result and result["documents"]:
            return "\n\n---\n\n".join(result["documents"])
        return ""
    except Exception:
        return ""


def get_course_chunks_with_meta(course: str) -> list[dict]:
    """Get all chunks with metadata for a course."""
    client = _get_client()
    cname = collection_name(course)
    try:
        col = client.get_collection(cname)
        result = col.get(include=["documents", "metadatas"])
        chunks = []
        if result and result["documents"]:
            for i, doc in enumerate(result["documents"]):
                meta = (result["metadatas"][i] if result["metadatas"] and i < len(result["metadatas"]) else {}) or {}
                chunks.append({"text": doc, "meta": meta})
        return chunks
    except Exception:
        return []


def add_chunks(course: str, embeddings: np.ndarray, documents: list[str], metadatas: list[dict]):
    """Add embedded chunks to a course collection."""
    client = _get_client()
    cname = collection_name(course)

    try:
        col = client.get_collection(cname)
    except Exception:
        col = client.create_collection(name=cname, metadata={"hnsw:space": "cosine"})

    ids = [f"{cname}-{uuid.uuid4().hex[:12]}" for _ in documents]
    col.add(ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas)
    _bm25_invalidate(course)
    compute_course_centroid(course)


def search(course: str, query_embedding: np.ndarray, top_k: int = 8) -> list[dict]:
    """Semantic search within a course collection."""
    client = _get_client()
    cname = collection_name(course)
    try:
        col = client.get_collection(cname)
        results = col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, col.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = (results["metadatas"][0][i] if results["metadatas"] and len(results["metadatas"][0]) > i else {}) or {}
                dist = results["distances"][0][i] if results.get("distances") and len(results["distances"][0]) > i else 0
                docs.append({"content": doc, "meta": meta, "score": float(1.0 - dist)})
        return docs
    except Exception as e:
        logger.error(f"Search error in {course}: {e}")
        return []


def _bm25_invalidate(course: str):
    """Invalidate BM25 cache and centroid for a course."""
    cname = collection_name(course)
    _bm25_cache.pop(cname, None)
    _course_centroids.pop(course, None)


def search_keyword(course: str, query: str, top_k: int = 8) -> list[dict]:
    """Keyword-based (BM25) search within a course. Caches BM25 index per course."""
    try:
        from rank_bm25 import BM25Okapi
        cname = collection_name(course)
        chunks = get_course_chunks_with_meta(course)
        if not chunks:
            return []

        chunk_count = len(chunks)

        # Use cached BM25 index if chunk count hasn't changed
        cached = _bm25_cache.get(cname)
        if cached and cached[0] == chunk_count:
            bm25 = cached[1]
        else:
            texts = [c["text"] for c in chunks]
            tokenized = [t.lower().split() for t in texts]
            bm25 = BM25Okapi(tokenized)
            _bm25_cache[cname] = (chunk_count, bm25)

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k by score
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        max_score = float(max(scores)) if max(scores) > 0 else 1.0
        for idx, score in indexed:
            if score > 0:
                results.append({**chunks[idx], "score": float(score) / max_score})
        return results
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return []


def hybrid_search(course: str, query_embedding: np.ndarray, query_text: str, top_k: int = 8) -> list[dict]:
    """Combine semantic and keyword search, deduplicate, and merge by weighted score."""
    semantic_results = search(course, query_embedding, top_k)
    keyword_results = search_keyword(course, query_text, top_k)

    # Merge: semantic weight 0.7, keyword weight 0.3
    seen = set()
    merged = {}

    for doc in semantic_results:
        text = (doc.get("content") or doc.get("text") or "")
        key = text[:100]
        seen.add(key)
        merged[key] = {"content": text, "meta": doc.get("meta", {}), "score": doc.get("score", 0) * 0.7}

    for doc in keyword_results:
        text = (doc.get("content") or doc.get("text") or "")
        if not text:
            continue
        key = text[:100]
        kw_score = doc.get("score", 0) * 0.3
        if key in merged:
            merged[key]["score"] += kw_score
        else:
            merged[key] = {"content": text, "meta": doc.get("meta", {}), "score": kw_score}

    return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def compute_course_centroid(course: str):
    """Pre-compute the mean embedding for a course (call after adding chunks)."""
    cname = collection_name(course)
    try:
        col = _get_client().get_collection(cname)
        result = col.get(include=["embeddings"])
        if result and result["embeddings"] and len(result["embeddings"]) > 0:
            embs = np.array(result["embeddings"])
            _course_centroids[course] = np.mean(embs, axis=0)
    except Exception:
        _course_centroids.pop(course, None)


def find_course_for_query(query_embedding: np.ndarray) -> Optional[str]:
    """Find best course via centroid dot-product (O(N) vs O(N) vector searches).

    Falls back to full search if centroids aren't computed yet.
    """
    courses = list_courses()
    if not courses:
        return None
    if len(courses) == 1:
        return courses[0]["name"]

    # Try centroid-based detection first (fast)
    if _course_centroids:
        best_course, best_score = None, -1
        for course_info in courses:
            centroid = _course_centroids.get(course_info["name"])
            if centroid is not None:
                # Normalize and compute cosine similarity
                q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                score = float(np.dot(q_norm, c_norm))
                if score > best_score:
                    best_score = score
                    best_course = course_info["name"]
        if best_course and best_score > 0.3:
            return best_course

    # Fallback: per-course search
    best_course, best_score = None, -1
    for course_info in courses:
        results = search(course_info["name"], query_embedding, top_k=1)
        if results and results[0]["score"] > best_score:
            best_score = results[0]["score"]
            best_course = course_info["name"]

    return best_course if best_score > 0.3 else courses[0]["name"]


def update_metadata(course: str, chunk_index: int, updates: dict):
    """Update metadata for a specific chunk (e.g., review tracking, errors)."""
    client = _get_client()
    cname = collection_name(course)
    try:
        col = client.get_collection(cname)
        result = col.get(include=["metadatas"])
        if result and result["metadatas"] and chunk_index < len(result["metadatas"]):
            meta = result["metadatas"][chunk_index] or {}
            meta.update(updates)
            col.update(ids=[result["ids"][chunk_index]], metadatas=[meta])
    except Exception as e:
        logger.error(f"Metadata update error: {e}")


def delete_course(course: str):
    """Delete an entire course collection."""
    client = _get_client()
    cname = collection_name(course)
    try:
        client.delete_collection(cname)
        _bm25_invalidate(course)
        logger.info(f"Deleted collection: {cname}")
    except Exception as e:
        logger.error(f"Delete course error: {e}")


def delete_file_from_course(course: str, filename: str):
    """Delete all chunks belonging to a specific file within a course."""
    client = _get_client()
    cname = collection_name(course)
    try:
        col = client.get_collection(cname)
        result = col.get(include=["metadatas"])
        if not result or not result["ids"]:
            return

        ids_to_delete = []
        for i, meta in enumerate(result["metadatas"]):
            if meta and meta.get("file") == filename:
                ids_to_delete.append(result["ids"][i])

        if ids_to_delete:
            col.delete(ids=ids_to_delete)
            _bm25_invalidate(course)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for {filename} from {course}")
    except Exception as e:
        logger.error(f"Delete file error: {e}")


def clear_all():
    """Delete all collections."""
    client = _get_client()
    for col in client.list_collections():
        try:
            client.delete_collection(col.name)
        except Exception:
            pass
    _bm25_cache.clear()
    logger.info("Cleared all collections")

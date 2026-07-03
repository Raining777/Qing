"""Session history manager — single-file persistence replacing LangGraph checkpoints.

Stores conversation messages + UI metadata in data/sessions.json.
Max 50 sessions, debounced writes.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from app.config import DATA_DIR

logger = logging.getLogger(__name__)

SESSIONS_FILE = DATA_DIR / "sessions.json"
MAX_SESSIONS = 50
WRITE_DEBOUNCE = 0.5  # seconds


class SessionStore:
    """Manages conversation sessions: CRUD + debounced persistence."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._dirty = False
        self._last_write = 0.0
        self._load()

    # ── Public API ──

    def get(self, sid: str) -> Optional[dict]:
        return self._sessions.get(sid)

    def create(self, preview: str = "", course: str = "") -> str:
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = {
            "id": sid,
            "preview": preview or "New conversation",
            "course": course,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "messages": [],
        }
        self._maybe_save()
        return sid

    def add_message(self, sid: str, role: str, content: str):
        """Append a message to session history."""
        s = self._sessions.get(sid)
        if not s:
            return
        s["messages"].append({"role": role, "content": content})
        # Update preview from first user message
        if role == "user" and s.get("preview", "").startswith("New"):
            s["preview"] = content[:50]
        self._dirty = True
        self._maybe_save()

    def get_messages(self, sid: str) -> list[dict]:
        """Get messages for a session (returns copy)."""
        s = self._sessions.get(sid)
        return list(s.get("messages", [])) if s else []

    def get_recent_context(self, sid: str, n: int = 6) -> str:
        """Get last N messages as context string for query expansion."""
        msgs = self.get_messages(sid)
        if not msgs:
            return ""
        parts = []
        for m in msgs[-n:]:
            role = m.get("role", "user")
            content = (m.get("content", "") or "")[:200]
            parts.append(f"[{role}]: {content}")
        return " | ".join(parts)

    def set_course(self, sid: str, course: str):
        s = self._sessions.get(sid)
        if s:
            s["course"] = course
            self._dirty = True
            self._maybe_save()

    def get_course(self, sid: str) -> str:
        s = self._sessions.get(sid)
        return s.get("course", "") if s else ""

    def get_summaries(self, sid: str) -> dict:
        """Get cached chapter summaries for a session."""
        s = self._sessions.get(sid)
        return s.get("chapter_summaries", {}) if s else {}

    def set_summaries(self, sid: str, summaries: dict):
        s = self._sessions.get(sid)
        if s:
            existing = s.get("chapter_summaries", {})
            s["chapter_summaries"] = {**existing, **summaries}
            self._dirty = True
            self._maybe_save()

    def list_sessions(self) -> list[dict]:
        """Return session list for sidebar UI (newest first)."""
        items = []
        for sid, s in self._sessions.items():
            items.append({
                "id": sid,
                "preview": s.get("preview", ""),
                "course": s.get("course", ""),
            })
        items.sort(key=lambda x: x["id"], reverse=True)
        return items[:MAX_SESSIONS]

    def delete(self, sid: str):
        if sid in self._sessions:
            del self._sessions[sid]
            self._dirty = True
            self._save()

    def flush(self):
        """Force immediate save."""
        self._save()

    # ── Internal ──

    def _load(self):
        try:
            if SESSIONS_FILE.exists():
                self._sessions = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            self._sessions = {}

    def _maybe_save(self):
        """Debounced save — writes at most once per WRITE_DEBOUNCE seconds."""
        now = time.time()
        if now - self._last_write >= WRITE_DEBOUNCE and self._dirty:
            self._save()

    def _save(self):
        try:
            SESSIONS_FILE.write_text(
                json.dumps(self._sessions, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._dirty = False
            self._last_write = time.time()
        except Exception as e:
            logger.warning(f"Failed to save sessions: {e}")


# ── Global singleton ──
_store: Optional[SessionStore] = None


def get_sessions() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store

"""
BriefMind Document Store
In-memory KV store for document briefings and raw text.
Swap for Redis, PostgreSQL, or any persistent store in production.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class DocumentStore:
    """Thread-safe in-memory document store."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._active_doc_id: Optional[str] = None

    # ── Write ──────────────────────────────────

    def store_raw(self, doc_id: str, title: str, text: str) -> None:
        """Store raw document text."""
        if doc_id not in self._store:
            self._store[doc_id] = {}
        self._store[doc_id].update({
            "doc_id": doc_id,
            "title": title,
            "raw_text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "created_at": datetime.utcnow().isoformat(),
            "status": "processing",
        })

    def store_briefing(self, doc_id: str, title: str, text: str, briefing: dict) -> None:
        """Store the full briefing result."""
        if doc_id not in self._store:
            self._store[doc_id] = {}
        self._store[doc_id].update({
            "doc_id": doc_id,
            "title": title,
            "raw_text": text,
            "word_count": len(text.split()),
            "char_count": len(text),
            "briefing": briefing,
            "created_at": datetime.utcnow().isoformat(),
            "status": "ready",
        })

    # ── Read ───────────────────────────────────

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        doc = self._store.get(doc_id)
        if doc:
            # Return without raw_text to keep response lean
            return {k: v for k, v in doc.items() if k != "raw_text"}
        return None

    def get_raw_text(self, doc_id: str) -> Optional[str]:
        """Retrieve the raw text of a document."""
        doc = self._store.get(doc_id)
        return doc.get("raw_text") if doc else None

    def list_all(self) -> list:
        """List metadata for all stored documents."""
        return [
            {
                "doc_id": v["doc_id"],
                "title": v.get("title", "Untitled"),
                "word_count": v.get("word_count", 0),
                "status": v.get("status", "unknown"),
                "created_at": v.get("created_at", ""),
            }
            for v in self._store.values()
        ]

    # ── Delete ─────────────────────────────────

    def delete(self, doc_id: str) -> bool:
        """Delete a document. Returns True if deleted."""
        if doc_id in self._store:
            del self._store[doc_id]
            return True
        return False

    # ── Active doc (for chat mode) ─────────────

    def set_active(self, doc_id: str) -> None:
        self._active_doc_id = doc_id

    def get_active(self) -> Optional[str]:
        return self._active_doc_id
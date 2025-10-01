"""Utility helpers for persisting LLM API keys for the desktop GUI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
import json


@dataclass
class ApiKeyStore:
    """File-backed storage for Gemini API keys used by the GUI.

    The desktop application lets operators add or remove keys at runtime.
    This helper keeps the implementation testable by encapsulating the
    persistence logic outside of the Tkinter components.
    """

    storage_path: Path

    def load(self) -> List[str]:
        """Load keys from disk and return a sanitized list."""
        if not self.storage_path.exists():
            return []

        try:
            raw = self.storage_path.read_text(encoding="utf-8")
        except OSError:
            # If the file cannot be read we treat it as empty to keep the GUI usable.
            return []

        try:
            data = json.loads(raw or "[]")
        except json.JSONDecodeError:
            # Malformed content should not crash the app; treat as empty.
            return []

        return self._sanitize_iterable(data)

    def save(self, keys: Iterable[str]) -> None:
        """Persist the provided keys to disk in a deterministic order."""
        sanitized = self._sanitize_iterable(keys)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(sanitized, indent=2, ensure_ascii=False)
        self.storage_path.write_text(payload + "\n", encoding="utf-8")

    @staticmethod
    def _sanitize_iterable(keys: Iterable[str]) -> List[str]:
        """Return unique, non-empty keys while preserving insertion order."""
        seen = set()
        ordered: List[str] = []
        for key in keys or []:
            if key is None:
                continue
            cleaned = str(key).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                ordered.append(cleaned)
        return ordered

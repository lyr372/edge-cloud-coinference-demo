from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CacheRecord:
    text: str
    hits: int


class PersistentKVCache:
    """Simple persisted KV cache for hot queries and incremental updates."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, CacheRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for key, value in raw.items():
            self._store[key] = CacheRecord(text=value["text"], hits=value.get("hits", 0))

    def _flush(self) -> None:
        payload = {key: {"text": rec.text, "hits": rec.hits} for key, rec in self._store.items()}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str) -> CacheRecord | None:
        record = self._store.get(key)
        if record:
            record.hits += 1
            self._flush()
        return record

    def upsert(self, key: str, text: str, min_hits_for_persist: int = 1) -> None:
        existing = self._store.get(key)
        if existing:
            existing.text = text
            existing.hits += 1
        else:
            existing = CacheRecord(text=text, hits=1)
            self._store[key] = existing
        if existing.hits >= min_hits_for_persist:
            self._flush()

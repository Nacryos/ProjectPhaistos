"""Simple file-based checkpoint/resume support."""

from __future__ import annotations

from pathlib import Path

import orjson


class Checkpoint:
    """Tracks which items have been processed so stages can resume."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._done: set[str] = set()
        if self._path.exists():
            data = orjson.loads(self._path.read_bytes())
            self._done = set(data.get("done", []))

    def is_done(self, key: str) -> bool:
        return key in self._done

    def mark_done(self, key: str) -> None:
        self._done.add(key)
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(orjson.dumps({"done": sorted(self._done)}))

    def reset(self) -> None:
        self._done.clear()
        if self._path.exists():
            self._path.unlink()

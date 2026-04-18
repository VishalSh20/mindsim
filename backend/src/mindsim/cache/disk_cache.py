"""Content-addressed disk cache.

Used for LLM responses, scrape results, and research. Cache keys are sha256
hashes of the stage name plus canonicalised JSON of the input; this means the
same input produces the same key regardless of dict ordering.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path.home() / ".mindsim" / "cache"


class DiskCache:
    def __init__(
        self,
        namespace: str,
        ttl_seconds: int,
        root: Path | None = None,
    ):
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self.root = (root or DEFAULT_ROOT) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, key_input: Any) -> str:
        canonical = json.dumps(key_input, sort_keys=True, default=str)
        payload = f"{self.namespace}\x00{canonical}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.pkl"

    def get(self, key_input: Any) -> Any | None:
        key = self._key(key_input)
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                record = pickle.load(f)
        except (pickle.PickleError, EOFError, OSError) as exc:
            logger.warning("cache read failed at %s: %s", path, exc)
            return None
        age = time.time() - record["written_at"]
        if age >= self.ttl_seconds:
            logger.debug("cache expired (%.1fs > %ds): %s", age, self.ttl_seconds, key)
            return None
        return record["value"]

    def set(self, key_input: Any, value: Any) -> None:
        key = self._key(key_input)
        path = self._path(key)
        record = {"written_at": time.time(), "value": value}
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(record, f)
        tmp.replace(path)

    def clear(self) -> int:
        removed = 0
        for p in self.root.glob("*.pkl"):
            try:
                p.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("could not remove %s: %s", p, exc)
        return removed

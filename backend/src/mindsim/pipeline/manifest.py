"""Run manifest: reproducibility + observability for a single pipeline run.

Records stage timings, LLM call counts, cache hits, scrape budget usage, and
warnings. Written alongside the diagnostic dump as `run_manifest.json`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RunManifest(BaseModel):
    run_id: str
    timestamp: str
    product_hash: str
    seed: int | None = None
    git_sha: str | None = None
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    llm_calls: dict[str, int] = Field(default_factory=dict)
    cache_hits: dict[str, int] = Field(default_factory=dict)
    scrape_budget_used: int = 0
    total_cost_usd: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class ManifestRecorder:
    """Mutable accumulator. Call finalize() or write() once per run."""

    def __init__(self) -> None:
        self._stage_start: dict[str, float] = {}
        self.stage_timings_ms: dict[str, float] = {}
        self.llm_calls: dict[str, int] = {}
        self.cache_hits: dict[str, int] = {}
        self.scrape_budget_used: int = 0
        self.total_cost_usd: float = 0.0
        self.warnings: list[str] = []

    def start_stage(self, name: str) -> None:
        self._stage_start[name] = time.perf_counter()

    def end_stage(self, name: str) -> None:
        start = self._stage_start.pop(name, None)
        if start is None:
            logger.warning("end_stage(%r) called without matching start_stage", name)
            return
        self.stage_timings_ms[name] = round((time.perf_counter() - start) * 1000.0, 2)

    def record_llm_call(self, agent: str) -> None:
        self.llm_calls[agent] = self.llm_calls.get(agent, 0) + 1

    def record_cache_hit(self, namespace: str) -> None:
        self.cache_hits[namespace] = self.cache_hits.get(namespace, 0) + 1

    def record_scrape(self, count: int = 1) -> None:
        self.scrape_budget_used += count

    def record_cost(self, usd: float) -> None:
        self.total_cost_usd += usd

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def finalize(
        self,
        product_description: str,
        seed: int | None = None,
    ) -> RunManifest:
        return RunManifest(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            product_hash=_hash_text(product_description),
            seed=seed,
            git_sha=_git_sha(),
            stage_timings_ms=dict(self.stage_timings_ms),
            llm_calls=dict(self.llm_calls),
            cache_hits=dict(self.cache_hits),
            scrape_budget_used=self.scrape_budget_used,
            total_cost_usd=round(self.total_cost_usd, 4),
            warnings=list(self.warnings),
        )

    def write(
        self,
        path: Path,
        product_description: str,
        seed: int | None = None,
    ) -> RunManifest:
        manifest = self.finalize(product_description, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)
        return manifest


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None

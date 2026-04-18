"""Thin in-house pipeline orchestrator.

Rationale — see plan §0 (Framework Decision): this pipeline is 95%
linear and deterministic. LangGraph would be 85% unused. We keep the
existing synchronous LLMClient and wrap the pipeline with a tiny
registry/DAG runner that gives us:

  - typed Pydantic input/output on every stage boundary,
  - cache-aware invocation (keyed on stage name + input hash),
  - manifest recording of stage timings,
  - a validator-retry wrapper (used in Wave 2 for A4 ← A5),
  - parallel fork via ThreadPoolExecutor (used in Wave 4 for A2 ∥ A3).

Stages are synchronous — matching the current LLMClient — and the
orchestrator is a plain class, not an `async` framework. When a future
feature *actually* needs dynamic routing, switch to LangGraph (cf.
plan §0). Not before.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel

from mindsim.cache import DiskCache
from mindsim.llm.client import LLMClient
from mindsim.pipeline.manifest import ManifestRecorder

logger = logging.getLogger(__name__)


@runtime_checkable
class Stage(Protocol):
    """One pipeline node. Named, typed, synchronous."""

    name: str
    input_type: type[BaseModel]
    output_type: type[BaseModel]

    def run(self, inp: BaseModel, ctx: "StageContext") -> BaseModel: ...


@dataclass
class StageContext:
    """Everything a stage needs to do its work.

    Kept flat on purpose — stages reach for exactly what they need.
    """

    llm: LLMClient | None = None
    cache: DiskCache | None = None
    manifest: ManifestRecorder = field(default_factory=ManifestRecorder)
    # Arbitrary shared data — research results, scraped docs, etc.
    # Stages pass their primary output via return, but can stash
    # cross-stage context here when a sibling stage needs it.
    state: dict[str, Any] = field(default_factory=dict)


class StageError(RuntimeError):
    """Raised when a stage cannot produce a valid output."""


class Orchestrator:
    """Runs stages sequentially, in parallel, or with validator retry.

    This is deliberately not a full graph executor. There is no dynamic
    routing, no conditional edges based on LLM output, no agent loops.
    Three primitives cover the v2-middle pipeline:

      * run_sequence(["A1", "A2", ...], initial)  — linear chain
      * run_parallel(["A2", "A3"], common_input)  — fork
      * run_with_retry("A4", "A5", initial)       — validator-retry
    """

    def __init__(self, stages: list[Stage], ctx: StageContext):
        self.ctx = ctx
        self.stages: dict[str, Stage] = {}
        for s in stages:
            if s.name in self.stages:
                raise ValueError(f"duplicate stage name: {s.name}")
            self.stages[s.name] = s

    def register(self, stage: Stage) -> None:
        if stage.name in self.stages:
            raise ValueError(f"duplicate stage name: {stage.name}")
        self.stages[stage.name] = stage

    # ────────────────────────── core primitives ──────────────────────────

    def run_sequence(self, names: list[str], initial: BaseModel) -> BaseModel:
        current: BaseModel = initial
        for name in names:
            current = self._invoke(name, current)
        return current

    def run_parallel(
        self,
        names: list[str],
        input_val: BaseModel,
    ) -> dict[str, BaseModel]:
        """Run the named stages concurrently on the same input.

        Uses a thread pool since our LLM calls release the GIL on I/O.
        Returns a dict mapping stage name → output.
        """
        if not names:
            return {}
        results: dict[str, BaseModel] = {}
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {pool.submit(self._invoke, n, input_val): n for n in names}
            for fut in futures:
                name = futures[fut]
                results[name] = fut.result()
        return results

    def run_with_retry(
        self,
        stage_name: str,
        validator_name: str,
        input_val: BaseModel,
        max_retries: int = 1,
        combine_issues: Callable[[BaseModel, list[str]], BaseModel] | None = None,
    ) -> BaseModel:
        """Run a stage, validate, re-prompt on failure up to `max_retries`.

        `combine_issues` is an optional callable that folds validator
        issue messages back into the stage input for the retry round.
        If omitted, the retry re-runs with the original input.
        """
        current_input = input_val
        last_output: BaseModel | None = None
        for attempt in range(max_retries + 1):
            last_output = self._invoke(stage_name, current_input)
            report = self._invoke(validator_name, last_output)
            is_valid = getattr(report, "is_valid", True)
            if is_valid:
                return last_output
            issues = getattr(report, "issues", [])
            issue_msgs = [
                getattr(i, "message", str(i)) for i in issues
            ]
            if attempt < max_retries:
                logger.warning(
                    "stage %s failed validation on attempt %d: %s",
                    stage_name,
                    attempt + 1,
                    issue_msgs,
                )
                if combine_issues is not None:
                    current_input = combine_issues(current_input, issue_msgs)
                continue
            self.ctx.manifest.warn(
                f"{stage_name} failed validation after {max_retries + 1} attempts; "
                f"proceeding with best-effort output"
            )
        assert last_output is not None
        return last_output

    # ────────────────────────── plumbing ──────────────────────────

    def _invoke(self, name: str, inp: BaseModel) -> BaseModel:
        stage = self.stages.get(name)
        if stage is None:
            raise KeyError(f"stage {name!r} not registered")

        # Cache probe before running. Key on stage name + input hash.
        cache_hit = False
        if self.ctx.cache is not None:
            cache_key = {"stage": name, "input": inp.model_dump(mode="json")}
            hit = self.ctx.cache.get(cache_key)
            if hit is not None:
                try:
                    out = stage.output_type.model_validate(hit)
                    self.ctx.manifest.record_cache_hit(name)
                    cache_hit = True
                    return out
                except Exception as exc:  # noqa: BLE001
                    logger.debug("cache deserialisation failed for %s: %s", name, exc)

        self.ctx.manifest.start_stage(name)
        try:
            out = stage.run(inp, self.ctx)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(f"stage {name!r} raised: {exc}") from exc
        finally:
            self.ctx.manifest.end_stage(name)

        if not isinstance(out, BaseModel):
            raise StageError(
                f"stage {name!r} returned {type(out).__name__}, expected BaseModel"
            )

        # Best-effort cache write.
        if self.ctx.cache is not None and not cache_hit:
            try:
                self.ctx.cache.set(
                    {"stage": name, "input": inp.model_dump(mode="json")},
                    out.model_dump(mode="json"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("cache write failed for %s: %s", name, exc)

        return out

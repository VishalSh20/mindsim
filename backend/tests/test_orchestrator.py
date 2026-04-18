"""Unit tests for the thin in-house orchestrator.

No LLM calls; every stage is a deterministic pure function wrapped in
the Stage protocol.
"""
import threading
import time

import pytest
from pydantic import BaseModel

from mindsim.cache import DiskCache
from mindsim.pipeline.manifest import ManifestRecorder
from mindsim.pipeline.orchestrator import (
    Orchestrator,
    StageContext,
    StageError,
)


class In(BaseModel):
    value: int


class Out(BaseModel):
    result: int


class Double:
    name = "double"
    input_type = In
    output_type = Out

    def run(self, inp: In, ctx) -> Out:
        return Out(result=inp.value * 2)


class AddOne:
    name = "add_one"
    input_type = Out
    output_type = Out

    def run(self, inp: Out, ctx) -> Out:
        return Out(result=inp.result + 1)


class SlowIdentity:
    name = "slow_identity"
    input_type = In
    output_type = Out

    def __init__(self, delay: float = 0.05):
        self.delay = delay

    def run(self, inp: In, ctx) -> Out:
        time.sleep(self.delay)
        return Out(result=inp.value)


class Rejector:
    """Stage that always raises — tests error wrapping."""

    name = "rejector"
    input_type = In
    output_type = Out

    def run(self, inp: In, ctx) -> Out:
        raise RuntimeError("nope")


class _Report(BaseModel):
    is_valid: bool
    issues: list[str] = []


class FailThenPass:
    """Validator that returns invalid N times then valid."""

    name = "picky_validator"
    input_type = Out
    output_type = _Report

    def __init__(self, fail_n: int):
        self.fail_n = fail_n
        self.calls = 0

    def run(self, inp, ctx):
        self.calls += 1
        if self.calls <= self.fail_n:
            return _Report(is_valid=False, issues=["try again"])
        return _Report(is_valid=True)


@pytest.fixture
def ctx() -> StageContext:
    return StageContext()


def test_run_sequence_produces_expected_chain(ctx):
    orch = Orchestrator([Double(), AddOne()], ctx)
    result = orch.run_sequence(["double", "add_one"], In(value=5))
    assert result.result == 11  # 5 * 2 + 1


def test_register_rejects_duplicates(ctx):
    with pytest.raises(ValueError):
        Orchestrator([Double(), Double()], ctx)


def test_unknown_stage_name(ctx):
    orch = Orchestrator([Double()], ctx)
    with pytest.raises(KeyError):
        orch.run_sequence(["missing"], In(value=1))


def test_stage_exception_wrapped_as_stage_error(ctx):
    orch = Orchestrator([Rejector()], ctx)
    with pytest.raises(StageError):
        orch.run_sequence(["rejector"], In(value=0))


def test_run_parallel_returns_all_outputs(ctx):
    stages = [SlowIdentity(delay=0.02) for _ in range(3)]
    # Give them unique names.
    for i, s in enumerate(stages):
        s.name = f"slow_{i}"
    orch = Orchestrator(stages, ctx)
    start = time.monotonic()
    results = orch.run_parallel(["slow_0", "slow_1", "slow_2"], In(value=7))
    elapsed = time.monotonic() - start
    assert set(results.keys()) == {"slow_0", "slow_1", "slow_2"}
    assert all(r.result == 7 for r in results.values())
    # Sequential would be ≥0.06s; parallel should be meaningfully less.
    assert elapsed < 0.06


def test_manifest_records_stage_timings(ctx):
    orch = Orchestrator([Double(), AddOne()], ctx)
    orch.run_sequence(["double", "add_one"], In(value=5))
    assert "double" in ctx.manifest.stage_timings_ms
    assert "add_one" in ctx.manifest.stage_timings_ms


def test_run_with_retry_succeeds_on_first_pass(ctx):
    validator = FailThenPass(fail_n=0)
    orch = Orchestrator([Double(), validator], ctx)
    result = orch.run_with_retry("double", "picky_validator", In(value=3))
    assert result.result == 6
    assert validator.calls == 1


def test_run_with_retry_retries_once_then_succeeds(ctx):
    validator = FailThenPass(fail_n=1)
    orch = Orchestrator([Double(), validator], ctx)
    result = orch.run_with_retry(
        "double", "picky_validator", In(value=3), max_retries=1
    )
    assert result.result == 6
    assert validator.calls == 2  # one fail, one pass


def test_run_with_retry_exhausts_and_returns_last_output(ctx):
    validator = FailThenPass(fail_n=10)  # never passes
    orch = Orchestrator([Double(), validator], ctx)
    result = orch.run_with_retry(
        "double", "picky_validator", In(value=3), max_retries=1
    )
    # After exhaustion, returns the last output with a manifest warning.
    assert result.result == 6
    assert any("failed validation" in w for w in ctx.manifest.warnings)


def test_cache_hit_skips_stage_invocation(tmp_path):
    cache = DiskCache("orch_test", ttl_seconds=60, root=tmp_path)
    ctx = StageContext(cache=cache)

    class Counter:
        name = "counter"
        input_type = In
        output_type = Out
        calls = 0

        def run(self, inp, c):
            Counter.calls += 1
            return Out(result=inp.value * 10)

    orch = Orchestrator([Counter()], ctx)
    orch.run_sequence(["counter"], In(value=4))
    assert Counter.calls == 1
    # Second run with same input → cache hit, counter unchanged.
    orch.run_sequence(["counter"], In(value=4))
    assert Counter.calls == 1
    assert ctx.manifest.cache_hits.get("counter", 0) == 1

    # Different input → new invocation.
    orch.run_sequence(["counter"], In(value=5))
    assert Counter.calls == 2


def test_stage_must_return_basemodel(ctx):
    class BadStage:
        name = "bad"
        input_type = In
        output_type = Out

        def run(self, inp, c):
            return {"result": 99}  # not a BaseModel

    orch = Orchestrator([BadStage()], ctx)
    with pytest.raises(StageError):
        orch.run_sequence(["bad"], In(value=1))


def test_parallel_fork_thread_safe_manifest(ctx):
    """Concurrent stages must not clobber each other's manifest entries."""
    stages = []
    for i in range(5):

        class S:
            def __init__(self, n):
                self.name = f"p_{n}"
                self.input_type = In
                self.output_type = Out

            def run(self, inp, c):
                time.sleep(0.01)
                return Out(result=inp.value + 1)

        stages.append(S(i))
    # Give distinct names via closure.
    for i, s in enumerate(stages):
        s.name = f"p_{i}"
    orch = Orchestrator(stages, ctx)
    results = orch.run_parallel([s.name for s in stages], In(value=0))
    assert len(results) == 5
    # All stages should have timing entries.
    for name in [s.name for s in stages]:
        assert name in ctx.manifest.stage_timings_ms

import json
import time

from mindsim.pipeline.manifest import ManifestRecorder, RunManifest


def test_manifest_records_stage_timing():
    rec = ManifestRecorder()
    rec.start_stage("understand")
    time.sleep(0.01)
    rec.end_stage("understand")
    manifest = rec.finalize(product_description="a product", seed=42)
    assert "understand" in manifest.stage_timings_ms
    assert manifest.stage_timings_ms["understand"] >= 5.0


def test_manifest_counters():
    rec = ManifestRecorder()
    rec.record_llm_call("A1")
    rec.record_llm_call("A1")
    rec.record_llm_call("A2")
    rec.record_cache_hit("llm")
    rec.record_scrape(3)
    rec.record_cost(0.02)
    rec.warn("bias note")

    manifest = rec.finalize(product_description="x", seed=1)
    assert manifest.llm_calls == {"A1": 2, "A2": 1}
    assert manifest.cache_hits == {"llm": 1}
    assert manifest.scrape_budget_used == 3
    assert manifest.total_cost_usd == 0.02
    assert manifest.warnings == ["bias note"]


def test_manifest_product_hash_deterministic():
    a = ManifestRecorder().finalize("same product", seed=1)
    b = ManifestRecorder().finalize("same product", seed=2)
    assert a.product_hash == b.product_hash


def test_manifest_product_hash_differs_by_text():
    a = ManifestRecorder().finalize("product A", seed=1)
    b = ManifestRecorder().finalize("product B", seed=1)
    assert a.product_hash != b.product_hash


def test_manifest_run_id_unique():
    a = ManifestRecorder().finalize("x", seed=1)
    b = ManifestRecorder().finalize("x", seed=1)
    assert a.run_id != b.run_id


def test_manifest_end_without_start_warns(caplog):
    rec = ManifestRecorder()
    rec.end_stage("nonexistent")
    assert "nonexistent" not in rec.stage_timings_ms


def test_manifest_write_roundtrip(tmp_path):
    rec = ManifestRecorder()
    rec.record_llm_call("A1")
    path = tmp_path / "nested" / "manifest.json"
    rec.write(path, product_description="my product", seed=123)
    assert path.exists()
    data = json.loads(path.read_text())
    restored = RunManifest.model_validate(data)
    assert restored.llm_calls == {"A1": 1}
    assert restored.seed == 123

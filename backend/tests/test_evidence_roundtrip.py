"""Wave 8.5 — Evidence + EvidenceStore round-trip + compression tests."""
from __future__ import annotations

import json

import pytest

from mindsim.models.evidence import (
    Evidence,
    EvidenceStore,
    build_evidence_store_from_market,
    default_evidence,
    evidence_from_research_snippet,
    evidence_from_voc_quote,
)
from mindsim.models.market import MarketContext
from mindsim.models.voc import VoCQuote, VoCReport


# ─────────────────── basic add / lookup ───────────────────


class TestEvidenceStore:
    def test_add_and_get(self):
        store = EvidenceStore()
        ev = Evidence(id="voc:123", text="hi", source_type="voc")
        store.add(ev)
        assert "voc:123" in store
        assert store.get("voc:123") is ev
        assert store.get("voc:nope") is None
        assert len(store) == 1

    def test_resolve_skips_unknown_ids(self):
        store = EvidenceStore()
        store.add(Evidence(id="src:a", text="x", source_type="research"))
        store.add(Evidence(id="src:b", text="y", source_type="research"))
        out = store.resolve(["src:a", "src:nope", "src:b"])
        assert [e.id for e in out] == ["src:a", "src:b"]


class TestCompression:
    def test_compressed_drops_unreferenced_records(self):
        store = EvidenceStore()
        store.add(Evidence(id="voc:1", text="kept", source_type="voc"))
        store.add(Evidence(id="voc:2", text="dropped", source_type="voc"))
        store.add(Evidence(id="src:9", text="kept too", source_type="research"))

        small = store.compressed({"voc:1", "src:9"})
        assert set(small.by_id.keys()) == {"voc:1", "src:9"}
        # Original store unchanged.
        assert len(store) == 3

    def test_compressed_with_empty_referenced_returns_empty_store(self):
        store = EvidenceStore()
        store.add(Evidence(id="voc:1", text="x", source_type="voc"))
        small = store.compressed(set())
        assert len(small) == 0


# ─────────────────── builders ───────────────────


class TestBuilders:
    def test_evidence_from_voc_quote(self):
        q = VoCQuote(
            quote_id="voc:abc",
            text="users love it",
            polarity="delight",
            archetype_hint="early_adopter",
            source="reddit",
            url="https://reddit.com/r/x/1",
        )
        ev = evidence_from_voc_quote(q)
        assert ev.id == "voc:abc"
        assert ev.source_type == "voc"
        assert ev.archetype_attribution == "early_adopter"
        assert ev.source_url == "https://reddit.com/r/x/1"

    def test_evidence_from_research_snippet_clamps_text(self):
        long = "x" * 500
        ev = evidence_from_research_snippet("src:1", "https://example.com", long)
        assert len(ev.text) <= 200

    def test_default_evidence(self):
        ev = default_evidence("present_bias_beta", "Augenblick 2015 default")
        assert ev.id == "default:present_bias_beta"
        assert ev.source_type == "default"
        assert ev.confidence < 0.5  # defaults are lower confidence than VoC


class TestBuildFromMarket:
    def test_builds_from_voc_quotes_and_tavily_and_scraped(self):
        from mindsim.scrape.base import (
            ScrapedDocument,
            make_quote_id,
            make_source_id,
        )
        from datetime import datetime, timezone

        # VoC quote.
        q = VoCQuote(
            quote_id=make_quote_id("https://r.com", "great pricing"),
            text="great pricing",
            polarity="delight",
            source="reddit",
            url="https://r.com",
        )
        voc = VoCReport(quotes=[q], pain_points=[], delight_points=[])

        # Market with a Tavily snippet AND a scraped doc.
        market = MarketContext()
        market.tavily_results_cache.append({
            "url": "https://example.com/p",
            "content": "some research snippet about pricing",
        })
        market.scraped_docs_cache.append(ScrapedDocument(
            source="reddit",
            url="https://reddit.com/x",
            retrieved_at=datetime.now(timezone.utc),
            title="t",
            body="another body about onboarding",
            quote_id="voc:ignored_for_research",
        ))

        store = build_evidence_store_from_market(market, voc)
        # Has voc + 1 tavily + 1 scraped → ≥ 3 entries.
        assert len(store) >= 3
        # VoC quote round-trips by ID.
        assert q.quote_id in store
        # Tavily entry's id matches make_source_id.
        sid_tavily = make_source_id(
            "https://example.com/p",
            "some research snippet about pricing"[:240],
        )
        assert sid_tavily in store

    def test_handles_missing_market_gracefully(self):
        store = build_evidence_store_from_market(None, voc=None)
        assert len(store) == 0


# ─────────────────── JSON round-trip ───────────────────


class TestJSONRoundTrip:
    def test_store_serialises_through_pydantic_json(self):
        store = EvidenceStore()
        store.add(Evidence(
            id="voc:1", text="text", source_type="voc",
            source_url="https://x", confidence=0.7,
        ))
        store.add(Evidence(
            id="src:1", text="snippet", source_type="research", confidence=0.5,
        ))
        as_json = store.model_dump_json()
        restored = EvidenceStore.model_validate_json(as_json)
        assert len(restored) == 2
        assert restored.get("voc:1").text == "text"
        assert restored.get("src:1").source_type == "research"

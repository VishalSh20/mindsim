"""A2 Research Synthesizer tests (Wave 5).

Stubbed LLM client — verifies:
  - competitors with invented source_ids are dropped (provenance invariant).
  - feature_scores clamped to [0,1].
  - existing verified_competitors preserved; new ones appended.
  - no-evidence path marks low_confidence rather than crashing.
  - category_penetration overridden only when A2 is more confident.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mindsim.models.market import MarketContext, VerifiedCompetitor
from mindsim.models.product import ProductProfile, ResearchPlan
from mindsim.models.voc import VoCReport
from mindsim.pipeline.research_synthesizer import synthesize_market
from mindsim.scrape.base import ScrapedDocument, make_quote_id, make_source_id


class _StubLLM:
    def __init__(self, response, raise_on_call=False):
        self.response = response
        self.raise_on_call = raise_on_call

    def complete_json(self, system_prompt, user_message, temperature=0.15):
        if self.raise_on_call:
            raise RuntimeError("boom")
        return self.response


def _profile():
    return ProductProfile(
        name="MyTool",
        category="devtools",
        value_proposition="ship faster",
        research_plan=ResearchPlan(),
    )


def _tavily(url: str, content: str) -> dict:
    return {"url": url, "content": content, "title": content[:40]}


def _pricing_doc(url: str, body: str) -> ScrapedDocument:
    return ScrapedDocument(
        source="pricing_page",
        url=url,
        retrieved_at=datetime.now(timezone.utc),
        title="pricing",
        body=body,
        quote_id=make_quote_id(url, body),
    )


class TestHappyPath:
    def test_parses_and_attaches_competitors(self):
        t1 = _tavily("https://compA.com/pricing", "CompA pricing $20/mo")
        t2 = _tavily("https://compB.com", "CompB is the market leader")
        sid1 = make_source_id(t1["url"], t1["content"][:240])
        sid2 = make_source_id(t2["url"], t2["content"][:240])

        response = {
            "competitors": [
                {
                    "name": "CompA",
                    "confirmed_price": 20.0,
                    "price_model": "subscription",
                    "has_free_tier": True,
                    "market_position": "challenger",
                    "market_share": 0.12,
                    "source_url": "https://compA.com/pricing",
                    "feature_scores": {
                        "core_value": 0.7,
                        "social_signal": 0.3,
                        "ongoing_cost": 0.4,
                        "switching_friction_reducer": 0.5,
                    },
                    "source_ids": [sid1],
                },
                {
                    "name": "CompB",
                    "confirmed_price": None,
                    "market_position": "leader",
                    "market_share": 0.60,
                    "source_ids": [sid2],
                    "feature_scores": {"core_value": 1.5, "ongoing_cost": -0.2},  # out of range
                },
            ],
            "category_penetration": 0.45,
            "category_penetration_confidence": 0.8,
        }
        llm = _StubLLM(response)
        context = MarketContext()
        context.category_penetration_confidence = 0.3  # weaker than synthesizer

        out = synthesize_market(
            context=context,
            tavily_results=[t1, t2],
            scraped_pricing_docs=[],
            voc=VoCReport.empty(),
            profile=_profile(),
            llm_client=llm,
        )

        names = {c.name for c in out.verified_competitors}
        assert names == {"CompA", "CompB"}
        compA = next(c for c in out.verified_competitors if c.name == "CompA")
        assert compA.confirmed_price == 20.0
        assert compA.market_share == 0.12
        assert compA.source_ids == [sid1]
        compB = next(c for c in out.verified_competitors if c.name == "CompB")
        # out-of-range feature scores clamped
        assert compB.feature_scores["core_value"] == pytest.approx(1.0)
        assert compB.feature_scores["ongoing_cost"] == pytest.approx(0.0)
        # Category penetration upgraded (higher confidence).
        assert out.category_penetration == pytest.approx(0.45)
        assert out.category_penetration_confidence == pytest.approx(0.8)


class TestFabricatedSidsDropped:
    def test_competitor_with_only_invented_sids_dropped(self):
        t1 = _tavily("https://real.com", "Real comp discussion")
        sid_real = make_source_id(t1["url"], t1["content"][:240])

        response = {
            "competitors": [
                {"name": "Fabricated", "source_ids": ["src:FAKE"], "feature_scores": {}},
                {"name": "Real", "source_ids": [sid_real, "src:FAKE"]},
            ]
        }
        llm = _StubLLM(response)
        context = MarketContext()
        out = synthesize_market(
            context=context,
            tavily_results=[t1],
            scraped_pricing_docs=[],
            voc=VoCReport.empty(),
            profile=_profile(),
            llm_client=llm,
        )
        names = [c.name for c in out.verified_competitors]
        assert "Fabricated" not in names
        assert "Real" in names
        real = next(c for c in out.verified_competitors if c.name == "Real")
        # Fabricated sid filtered out from the surviving competitor's refs.
        assert real.source_ids == [sid_real]


class TestExistingCompetitorsPreserved:
    def test_existing_verified_competitors_kept_and_new_appended(self):
        t1 = _tavily("https://new.com", "new comp")
        sid1 = make_source_id(t1["url"], t1["content"][:240])
        response = {
            "competitors": [
                {"name": "NewComp", "source_ids": [sid1], "feature_scores": {}},
                {"name": "ExistingComp", "source_ids": [sid1]},  # dedupe target
            ]
        }
        llm = _StubLLM(response)
        context = MarketContext(
            verified_competitors=[
                VerifiedCompetitor(name="ExistingComp", confirmed_price=10.0),
            ]
        )
        out = synthesize_market(
            context=context,
            tavily_results=[t1],
            scraped_pricing_docs=[],
            voc=VoCReport.empty(),
            profile=_profile(),
            llm_client=llm,
        )
        names = [c.name for c in out.verified_competitors]
        assert names.count("ExistingComp") == 1
        existing = next(c for c in out.verified_competitors if c.name == "ExistingComp")
        assert existing.confirmed_price == 10.0  # not overwritten
        assert "NewComp" in names


class TestNoEvidence:
    def test_no_evidence_marks_low_confidence(self):
        context = MarketContext()
        out = synthesize_market(
            context=context,
            tavily_results=[],
            scraped_pricing_docs=[],
            voc=VoCReport.empty(),
            profile=_profile(),
            llm_client=_StubLLM({}),
        )
        assert any("A2 synthesizer skipped" in f for f in out.low_confidence_flags)


class TestLLMFailure:
    def test_llm_failure_marks_flag_and_returns(self):
        t1 = _tavily("https://a", "content")
        llm = _StubLLM({}, raise_on_call=True)
        context = MarketContext()
        out = synthesize_market(
            context=context,
            tavily_results=[t1],
            scraped_pricing_docs=[],
            voc=VoCReport.empty(),
            profile=_profile(),
            llm_client=llm,
        )
        assert any("A2 synthesizer failed" in f for f in out.low_confidence_flags)
        assert out.verified_competitors == []


class TestPricingDocsContributeSourceIds:
    def test_pricing_doc_source_id_available_to_llm(self):
        doc = _pricing_doc("https://pricing.com/plans", "Pro $19/mo")
        sid = make_source_id(doc.url, doc.body[:240])
        response = {
            "competitors": [
                {"name": "PricingOnly", "source_ids": [sid], "confirmed_price": 19.0},
            ]
        }
        llm = _StubLLM(response)
        context = MarketContext()
        out = synthesize_market(
            context=context,
            tavily_results=[],
            scraped_pricing_docs=[doc],
            voc=VoCReport.empty(),
            profile=_profile(),
            llm_client=llm,
        )
        assert any(c.name == "PricingOnly" for c in out.verified_competitors)

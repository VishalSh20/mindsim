"""A3 VoC Analyst tests (Wave 5).

LLM client is stubbed — no network, no OpenAI. Tests verify:
  - empty corpus → VoCReport.empty() without invoking LLM.
  - happy path parses quotes, feature_sentiment, bias_note.
  - fabricated quote_ids are dropped (defence against hallucinated citations).
  - innovator-skewed corpus auto-sets bias_note when LLM omits it.
  - LLM failure gracefully degrades to empty report with corpus metadata.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mindsim.models.product import ProductProfile, ResearchPlan
from mindsim.pipeline.voc import analyze_voc
from mindsim.scrape.base import ScrapedDocument, make_quote_id


class _StubLLM:
    def __init__(self, response: dict | None = None, raise_on_call: bool = False):
        self.response = response or {}
        self.raise_on_call = raise_on_call
        self.calls = 0

    def complete_json(self, system_prompt: str, user_message: str, temperature: float = 0.15):
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("LLM down")
        return self.response


def _doc(source: str, url: str, body: str) -> ScrapedDocument:
    return ScrapedDocument(
        source=source,
        url=url,
        retrieved_at=datetime.now(timezone.utc),
        title=body[:40],
        body=body,
        quote_id=make_quote_id(url, body),
    )


def _profile() -> ProductProfile:
    return ProductProfile(
        name="Thing",
        category="AI tooling",
        value_proposition="Helps devs ship faster",
        research_plan=ResearchPlan(),
    )


class TestEmptyCorpus:
    def test_empty_returns_empty_report_without_llm(self):
        llm = _StubLLM()
        report = analyze_voc([], _profile(), llm_client=llm)
        assert llm.calls == 0
        assert report.corpus_size == 0
        assert report.bias_note and "No scraped VoC" in report.bias_note


class TestHappyPath:
    def test_parses_structured_response(self):
        d1 = _doc("reddit", "https://reddit.com/r/x/1", "The pricing is great but the UX is rough")
        d2 = _doc("hackernews", "https://news.ycombinator.com/item?id=2", "Free tier is generous")
        response = {
            "pain_points": ["UX is rough"],
            "delight_points": ["generous free tier"],
            "complaint_themes": ["learning curve"],
            "unmet_needs": ["enterprise SSO"],
            "feature_sentiment": [
                {
                    "feature": "pricing",
                    "positive_pct": 0.7,
                    "negative_pct": 0.1,
                    "net_sentiment": 0.6,
                    "n_mentions": 3,
                    "quote_ids": [d1.quote_id, d2.quote_id],
                }
            ],
            "quotes": [
                {
                    "quote_id": d1.quote_id,
                    "text": "pricing great but UX rough",
                    "polarity": "pain",
                    "archetype_hint": "early_adopter",
                    "source": "reddit",
                    "url": d1.url,
                },
                {
                    "quote_id": d2.quote_id,
                    "text": "Free tier is generous",
                    "polarity": "delight",
                    "archetype_hint": "innovator",
                    "source": "hackernews",
                    "url": d2.url,
                },
            ],
            "bias_note": None,
        }
        llm = _StubLLM(response)
        report = analyze_voc([d1, d2], _profile(), llm_client=llm)

        assert llm.calls == 1
        assert len(report.quotes) == 2
        assert report.feature_sentiment[0].net_sentiment == pytest.approx(0.6)
        assert report.corpus_size == 2
        assert report.source_breakdown == {"reddit": 1, "hackernews": 1}
        # Innovator-skewed corpus (100% reddit+HN) → auto bias_note.
        assert report.bias_note is not None
        assert "innovator" in report.bias_note.lower() or "reddit" in report.bias_note.lower()


class TestFabricatedIdsDropped:
    def test_invented_quote_ids_filtered(self):
        d1 = _doc("reddit", "https://u", "legitimate body text")
        response = {
            "pain_points": [],
            "delight_points": [],
            "quotes": [
                {
                    "quote_id": "voc:FABRICATED1",
                    "text": "hallucinated quote",
                    "polarity": "pain",
                    "source": "reddit",
                },
                {
                    "quote_id": d1.quote_id,
                    "text": "legitimate body text",
                    "polarity": "delight",
                    "source": "reddit",
                },
            ],
            "feature_sentiment": [
                {
                    "feature": "foo",
                    "positive_pct": 0.5,
                    "negative_pct": 0.1,
                    "n_mentions": 1,
                    "quote_ids": ["voc:FAKE", d1.quote_id],
                }
            ],
            "bias_note": "manual note",
        }
        llm = _StubLLM(response)
        report = analyze_voc([d1], _profile(), llm_client=llm)

        assert len(report.quotes) == 1
        assert report.quotes[0].quote_id == d1.quote_id
        assert report.feature_sentiment[0].quote_ids == [d1.quote_id]


class TestBiasNoteAutoInjection:
    def test_innovator_skew_forces_bias_note(self):
        d1 = _doc("reddit", "https://u1", "one")
        d2 = _doc("reddit", "https://u2", "two")
        d3 = _doc("hackernews", "https://u3", "three")
        # No bias_note in response; auto-inject because 100% innovator sources.
        response = {"pain_points": [], "delight_points": [], "quotes": [], "feature_sentiment": []}
        llm = _StubLLM(response)
        report = analyze_voc([d1, d2, d3], _profile(), llm_client=llm)
        assert report.bias_note is not None

    def test_balanced_corpus_no_auto_injection(self):
        d1 = _doc("reddit", "https://u1", "one")
        d2 = _doc("pricing_page", "https://u2", "two")
        d3 = _doc("pricing_page", "https://u3", "three")  # 1/3 reddit only
        response = {"pain_points": [], "delight_points": [], "quotes": [], "feature_sentiment": []}
        llm = _StubLLM(response)
        report = analyze_voc([d1, d2, d3], _profile(), llm_client=llm)
        # 33% innovator < 60% threshold → no auto bias_note.
        assert report.bias_note is None


class TestLLMFailureDegrades:
    def test_llm_exception_returns_empty_with_corpus_metadata(self):
        d1 = _doc("reddit", "https://u", "body")
        llm = _StubLLM(raise_on_call=True)
        report = analyze_voc([d1], _profile(), llm_client=llm)
        assert report.corpus_size == 1
        assert report.source_breakdown == {"reddit": 1}
        assert report.bias_note  # empty() populates a note
        assert len(report.quotes) == 0

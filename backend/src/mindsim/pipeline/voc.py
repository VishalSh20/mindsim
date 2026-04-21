"""Stage A3 — Voice-of-Customer Analyst.

Consumes scraped documents → emits a `VoCReport` with pain/delight
points, per-feature sentiment, representative quotes (each carrying its
stable `voc:<hash>` id), and a transparency `bias_note` when the corpus
is innovator-skewed.

One LLM call (`VOC_PROMPT`). Graceful degradation: on empty corpus or
parse failure, returns `VoCReport.empty()` rather than raising.
"""
from __future__ import annotations

import logging
from collections import Counter

from mindsim.llm.client import LLMClient
from mindsim.llm.prompts import VOC_PROMPT
from mindsim.models.product import ProductProfile
from mindsim.models.voc import FeatureSentiment, VoCQuote, VoCReport
from mindsim.scrape.base import ScrapedDocument

logger = logging.getLogger(__name__)


INNOVATOR_SOURCES = {"reddit", "hackernews"}
BIAS_THRESHOLD = 0.60  # >60% innovator-leaning → bias_note required.


def analyze_voc(
    docs: list[ScrapedDocument],
    profile: ProductProfile,
    llm_client: LLMClient | None = None,
) -> VoCReport:
    """Run the A3 VoC analyst over scraped documents.

    Args:
        docs: scraped corpus from research stage (may be empty).
        profile: product profile for context in the prompt.
        llm_client: optional injected client (tests pass a stubbed one).

    Returns:
        VoCReport. Never raises — on failure, emits VoCReport.empty().
    """
    if not docs:
        logger.info("A3 VoC: empty corpus, returning VoCReport.empty()")
        return VoCReport.empty()

    source_breakdown = dict(Counter(d.source for d in docs))
    corpus_size = len(docs)

    user_message = _build_user_message(docs, profile)

    client = llm_client or LLMClient()
    try:
        raw = client.complete_json(VOC_PROMPT, user_message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("A3 VoC LLM call failed: %s — returning empty report", exc)
        report = VoCReport.empty()
        report.source_count = corpus_size
        report.corpus_size = corpus_size
        report.source_breakdown = source_breakdown
        return report

    report = _parse_report(raw, docs)
    report.source_count = corpus_size
    report.corpus_size = corpus_size
    report.source_breakdown = source_breakdown

    # Enforce bias_note when corpus is innovator-skewed, even if the LLM
    # forgot to include one. Cheap deterministic guard.
    if _is_innovator_skewed(source_breakdown, corpus_size) and not report.bias_note:
        report.bias_note = (
            "Corpus is weighted toward Reddit / Hacker News — treat signal "
            "as innovator / early-adopter voice; mainstream sentiment may differ."
        )

    return report


def _is_innovator_skewed(breakdown: dict[str, int], total: int) -> bool:
    if total <= 0:
        return False
    innovator_count = sum(n for s, n in breakdown.items() if s in INNOVATOR_SOURCES)
    return (innovator_count / total) >= BIAS_THRESHOLD


def _build_user_message(docs: list[ScrapedDocument], profile: ProductProfile) -> str:
    header = (
        f"Product: {profile.name}\n"
        f"Category: {profile.category or 'unknown'}\n"
        f"Value proposition: {profile.value_proposition or 'n/a'}\n"
        f"Corpus size: {len(docs)}\n"
        f"\nDocuments:\n"
    )
    lines = [header]
    for d in docs:
        title = (d.title or "").strip().replace("\n", " ")
        body = (d.body or "").strip().replace("\n", " ")
        lines.append(
            f"[{d.source} | quote_id={d.quote_id}] {title}\n  {body[:400]}\n"
        )
    return "".join(lines)


def _parse_report(raw: dict, docs: list[ScrapedDocument]) -> VoCReport:
    """Shape LLM JSON into a VoCReport, dropping fabricated quote_ids.

    The only external validation: every `quote_id` in the response must
    match a document we supplied, else it's dropped. Defends the Wave 8.5
    EvidenceStore chain against fabricated citations.
    """
    valid_ids = {d.quote_id for d in docs}
    url_by_id = {d.quote_id: d.url for d in docs}

    def _filter_ids(ids: list[str]) -> list[str]:
        return [q for q in ids if isinstance(q, str) and q in valid_ids]

    quotes_raw = raw.get("quotes", []) or []
    quotes: list[VoCQuote] = []
    for q in quotes_raw:
        qid = q.get("quote_id")
        if qid not in valid_ids:
            continue
        try:
            quotes.append(
                VoCQuote(
                    quote_id=qid,
                    text=(q.get("text") or "")[:200],
                    polarity=_coerce_polarity(q.get("polarity")),
                    archetype_hint=_coerce_archetype(q.get("archetype_hint")),
                    source=_coerce_source(q.get("source")),
                    url=q.get("url") or url_by_id.get(qid),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("skipping malformed quote: %s", exc)

    fs_raw = raw.get("feature_sentiment", []) or []
    feature_sentiment: list[FeatureSentiment] = []
    for fs in fs_raw:
        try:
            pos = float(fs.get("positive_pct") or 0.0)
            neg = float(fs.get("negative_pct") or 0.0)
            net = float(
                fs.get("net_sentiment")
                if fs.get("net_sentiment") is not None
                else pos - neg
            )
            feature_sentiment.append(
                FeatureSentiment(
                    feature=str(fs.get("feature") or "unnamed"),
                    positive_pct=max(0.0, min(pos, 1.0)),
                    negative_pct=max(0.0, min(neg, 1.0)),
                    net_sentiment=max(-1.0, min(net, 1.0)),
                    n_mentions=int(fs.get("n_mentions") or 0),
                    quote_ids=_filter_ids(fs.get("quote_ids") or []),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("skipping malformed feature_sentiment: %s", exc)

    return VoCReport(
        pain_points=_coerce_str_list(raw.get("pain_points")),
        delight_points=_coerce_str_list(raw.get("delight_points")),
        complaint_themes=_coerce_str_list(raw.get("complaint_themes")),
        unmet_needs=_coerce_str_list(raw.get("unmet_needs")),
        feature_sentiment=feature_sentiment,
        quotes=quotes,
        bias_note=raw.get("bias_note") or None,
    )


def _coerce_str_list(x) -> list[str]:
    if not isinstance(x, list):
        return []
    return [str(v) for v in x if v]


def _coerce_polarity(x) -> str:
    return x if x in ("pain", "delight", "neutral") else "neutral"


def _coerce_archetype(x) -> str:
    allowed = {"innovator", "early_adopter", "early_majority",
               "late_majority", "laggard", "unknown"}
    return x if x in allowed else "unknown"


def _coerce_source(x) -> str:
    allowed = {"reddit", "hackernews", "pricing_page", "review", "other"}
    return x if x in allowed else "other"

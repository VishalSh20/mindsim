"""Wave 8.5 — unified evidence store.

`Evidence` is the canonical record for every quote / source / LLM
judgment that downstream artefacts cite. The store keys these by
stable IDs (`voc:<hash>` for VoC quotes, `src:<hash>` for research
snippets — both emitted by Wave 5's A2/A3 stages) so that
`Assumption.evidence_refs`, `CalibratedParam.evidence_refs`, and
`ProsConsItem.voc_evidence` can all resolve to the same record.

Compression rule: at serialisation time, the store keeps only the
records actually referenced by at least one downstream artefact (params,
features, ProsConsItems, assumptions). Unreferenced evidence is dropped
so the dump doesn't bloat with the full Tavily / scrape corpus.
"""
from __future__ import annotations

from typing import Iterable, Literal

from pydantic import BaseModel, Field


SourceType = Literal["voc", "research", "trends", "default", "llm_judgment"]


class Evidence(BaseModel):
    """One piece of evidence, addressable by `id`."""

    id: str  # "voc:<hash>" | "src:<hash>" | "default:<param>" | "llm:<hash>"
    text: str  # quote / snippet, clamped to 200 chars upstream of insertion
    source_type: SourceType
    source_url: str | None = None
    archetype_attribution: str | None = None  # voc.archetype_hint when known
    confidence: float = 0.5  # 0-1


class EvidenceStore(BaseModel):
    """Lookup table of Evidence records keyed by id."""

    by_id: dict[str, Evidence] = Field(default_factory=dict)

    def add(self, evidence: Evidence) -> None:
        self.by_id[evidence.id] = evidence

    def get(self, evidence_id: str) -> Evidence | None:
        return self.by_id.get(evidence_id)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self.by_id

    def __len__(self) -> int:
        return len(self.by_id)

    def resolve(self, ids: Iterable[str]) -> list[Evidence]:
        """Return the Evidence records matching `ids`. Unknown IDs are silently skipped."""
        out: list[Evidence] = []
        for eid in ids:
            ev = self.by_id.get(eid)
            if ev is not None:
                out.append(ev)
        return out

    def compressed(self, referenced_ids: Iterable[str]) -> "EvidenceStore":
        """Return a new EvidenceStore containing only IDs in `referenced_ids`.

        Compression is applied at serialisation boundary so the dump only
        carries evidence that's actually load-bearing for some artefact.
        """
        keep = set(referenced_ids)
        return EvidenceStore(
            by_id={k: v for k, v in self.by_id.items() if k in keep}
        )


def _strip_text(text: str, max_chars: int = 200) -> str:
    """Clamp text length and strip control characters that could confuse
    downstream prompts (defence-in-depth against prompt injection in
    scraped quotes — A3 already does some of this, but we re-clamp on
    insertion just in case)."""
    if not text:
        return ""
    cleaned = "".join(c for c in text if c.isprintable() or c in (" ", "\t"))
    return cleaned[:max_chars].strip()


def evidence_from_voc_quote(
    quote,
    confidence: float = 0.6,
) -> Evidence:
    """Build an Evidence record from a VoCQuote."""
    return Evidence(
        id=getattr(quote, "quote_id", ""),
        text=_strip_text(getattr(quote, "text", "")),
        source_type="voc",
        source_url=getattr(quote, "url", None),
        archetype_attribution=getattr(quote, "archetype_hint", None),
        confidence=confidence,
    )


def evidence_from_research_snippet(
    source_id: str,
    url: str,
    snippet: str,
    confidence: float = 0.5,
) -> Evidence:
    """Build an Evidence record from a Tavily / scraped-doc snippet."""
    return Evidence(
        id=source_id,
        text=_strip_text(snippet),
        source_type="research",
        source_url=url,
        confidence=confidence,
    )


def default_evidence(parameter: str, basis: str) -> Evidence:
    """Build an Evidence record for a default-sourced parameter.

    Used when --skip-research or no research corpus is available. The
    final report's limitations section must surface every default-
    sourced load-bearing number — this record makes them resolvable
    rather than evidence-less.
    """
    return Evidence(
        id=f"default:{parameter}",
        text=_strip_text(basis or "Behavioral-science default"),
        source_type="default",
        confidence=0.3,
    )


def build_evidence_store_from_market(market, voc) -> EvidenceStore:
    """Assemble a fresh EvidenceStore from a MarketContext + VoCReport.

    Walks three sources in priority order:
      1. VoC quotes — each carries its `voc:<hash>` id.
      2. Tavily snippets — recomputes `src:<hash>` via make_source_id
         to match what the synthesizer cited.
      3. Scraped docs — same recomputation.

    Compression happens later in `analyze()` via
    `EvidenceStore.compressed(referenced_ids)` once all artefacts have
    been built. This builder is intentionally inclusive.
    """
    # Local import — scrape.base depends on nothing in models, but we
    # avoid pulling it at module load time so models/ stays portable.
    from mindsim.scrape.base import make_source_id  # noqa: PLC0415

    store = EvidenceStore()

    if voc is not None:
        for q in getattr(voc, "quotes", []) or []:
            ev = evidence_from_voc_quote(q)
            if ev.id:
                store.add(ev)

    if market is None:
        return store

    for t in getattr(market, "tavily_results_cache", []) or []:
        url = (t.get("url") if isinstance(t, dict) else None) or ""
        if not url:
            continue
        snippet = (t.get("content") or t.get("title") or "")[:240]
        sid = make_source_id(url, snippet)
        store.add(evidence_from_research_snippet(sid, url, snippet))

    for d in getattr(market, "scraped_docs_cache", []) or []:
        url = getattr(d, "url", "") or ""
        body = (getattr(d, "body", "") or "")[:240]
        if not url or not body:
            continue
        sid = make_source_id(url, body)
        store.add(evidence_from_research_snippet(sid, url, body))

    return store

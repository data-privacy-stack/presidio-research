"""Entity mapping: Protocol and CanonicalMapper implementation."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from presidio_evaluator.entity_mapping.hierarchy import (
    ALL_CANONICAL_ENTITIES,
    EntityHierarchy,
    EntityNotMappedError,
)

logger = logging.getLogger("presidio_evaluator.entity_mapping")
# ---------------------------------------------------------------------------
# BIO/BIOES/BILOU/BILUO prefix/suffix stripping
# ---------------------------------------------------------------------------

_BIO_PREFIX_RE = re.compile(r"^[BIOELSU]-(.+)$", re.IGNORECASE)
_BIO_SUFFIX_RE = re.compile(r"^(.+)-[BIOELSU]$", re.IGNORECASE)


def _strip_bio(label: str) -> str:
    """Strip a single BIO/BIOES/BILOU/BILUO prefix or suffix (e.g. B-PERSON → PERSON)."""
    m = _BIO_PREFIX_RE.match(label)
    if m:
        return m.group(1)
    m = _BIO_SUFFIX_RE.match(label)
    if m:
        return m.group(1)
    return label


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class IncompleteMapping(RuntimeError):
    """Raised by get_mapping() when one or more labels are still pending."""

    def __init__(self, pending: list[str]) -> None:
        self.pending = list(pending)
        super().__init__(
            f"Mapping incomplete: {len(pending)} label(s) still pending: {pending}"
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EntityMapper(Protocol):
    """
    Protocol for entity label mappers.

    Implementations resolve a set of raw entity labels to canonical entities,
    maintain state about what is resolved vs. pending, and allow manual
    overrides via map().
    """

    @property
    def pending(self) -> list[str]:
        """Labels that have not yet been resolved to a canonical entity."""
        ...

    def map(self, mappings: dict[str, str | None]) -> "EntityMapper":
        """
        Manually assign canonical entities (or None) to one or more labels.

        :param mappings: dict of raw label → canonical entity (or None to suppress).
        :return: self, for chaining.
        """
        ...

    def resolve_interactively(self, *, prompt_fn=input) -> "EntityMapper":
        """
        Prompt the user for each pending label.

        :param prompt_fn: Callable used for input (injectable in tests).
        :return: self, for chaining.
        """
        ...

    def get_mapping(self) -> dict[str, str | None]:
        """
        Return the complete label → canonical dict.

        :raises IncompleteMapping: if any labels are still pending.
        """
        ...

    def render_html(self) -> None:
        """Render an audit table (HTML in Jupyter, plain text otherwise)."""
        ...


# ---------------------------------------------------------------------------
# Internal resolution record
# ---------------------------------------------------------------------------


@dataclass
class _Resolution:
    tier: str            # EXACT | FUZZY | COUNTRY | COUNTRY_FALLBACK | MANUAL | NONE
    canonical: str | None
    score: float | None  # similarity score for FUZZY; None otherwise


# ---------------------------------------------------------------------------
# CanonicalMapper
# ---------------------------------------------------------------------------

_INTERACTIVE_CUTOFF = 0.40


class CanonicalMapper:
    """
    Resolves a set of raw entity labels to canonical EntityHierarchy entities.

    Construction triggers an automatic resolution pass (exact alias, country-prefix,
    fuzzy). Labels that cannot be auto-resolved land in ``pending`` and must be
    handled via :meth:`map` or :meth:`resolve_interactively` before
    :meth:`get_mapping` will succeed.

    Example::

        mapper = CanonicalMapper(["EMAIL_ADDRESS", "EMAILADRES", "MY_LABEL"])
        mapper.render_html()              # inspect auto-resolved labels
        mapper.resolve_interactively()    # handle pending labels in the terminal
        mapping = mapper.get_mapping()    # raises IncompleteMapping if still pending
    """

    def __init__(
        self,
        labels: list[str] | set[str],
        *,
        hierarchy: EntityHierarchy | None = None,
        fuzzy_threshold: float = 0.80,
    ) -> None:
        self._hierarchy = hierarchy if hierarchy is not None else EntityHierarchy.default()
        self._fuzzy_threshold = fuzzy_threshold

        # Deduplicate, preserving order for lists
        seen: set[str] = set()
        self._labels: list[str] = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                self._labels.append(label)

        # BIO-stripped version of each label (original label remains the dict key)
        self._stripped: dict[str, str] = {label: _strip_bio(label) for label in self._labels}

        self._records: dict[str, _Resolution] = {}
        self._auto_resolve()

    @classmethod
    def from_dataset(cls, samples: list, **kwargs) -> "CanonicalMapper | dict":
        """
        Construct a CanonicalMapper from a list of InputSample objects.

        Extracts unique entity_type values from all spans, then delegates to
        :meth:`__init__` with those labels. ``**kwargs`` (e.g. ``hierarchy``,
        ``fuzzy_threshold``) are passed through.

        Returns the completed mapping dict when all labels are auto-resolved,
        or the CanonicalMapper instance when manual resolution is still needed.
        """
        from presidio_evaluator.data_objects import InputSample  # local import to avoid heavy dep at module level
        labels: set[str] = {
            span.entity_type
            for sample in samples
            if isinstance(sample, InputSample)
            for span in sample.spans
        }
        mapper = cls(labels, **kwargs)
        if not mapper.pending:
            return mapper.get_mapping()
        return mapper

    # ── Auto-resolve pass ────────────────────────────────────────────────────

    def _auto_resolve_one(self, label: str) -> _Resolution | None:
        """Return a _Resolution for *label* if auto-resolvable, else None."""
        h = self._hierarchy
        stripped = self._stripped[label]

        # Handle the outside token (O) — never enters the resolution pipeline
        if stripped.upper() == "O":
            logger.info("[NONE] O -> None  (outside token)")
            return _Resolution(tier="NONE", canonical=None, score=None)

        norm = h._normalize(stripped)

        # Tier 1: exact alias map
        if norm in h.raw_to_canonical:
            canonical = h.raw_to_canonical[norm]
            logger.info("[EXACT]   %s -> %s", label, canonical)
            return _Resolution(tier="EXACT", canonical=canonical, score=None)

        # Tier 2/3: country-prefix (exact remainder then NATIONAL_ID fallback)
        upper = stripped.upper()
        parts3 = upper.split("_", 2)
        parts2 = upper.split("_", 1)
        has_country = (
            (len(parts3) == 3 and f"{parts3[0]}_{parts3[1]}" in h.countries)
            or (len(parts2) >= 2 and parts2[0] in h.countries)
        )
        if has_country:
            result = h._country_prefix_canonical(stripped, threshold=1.0)
            if result is not None:
                if result == "NATIONAL_ID":
                    logger.warning(
                        "[COUNTRY-FALLBACK] %s -> NATIONAL_ID  ⚠ document type not recognised",
                        label,
                    )
                    return _Resolution(tier="COUNTRY_FALLBACK", canonical="NATIONAL_ID", score=None)
                logger.info("[COUNTRY] %s -> %s", label, result)
                return _Resolution(tier="COUNTRY", canonical=result, score=None)

        # Tier 4: fuzzy (alias or fuzzy country-prefix)
        try:
            canonical = h.canonicalize(stripped, threshold=self._fuzzy_threshold)
            nm = difflib.get_close_matches(
                norm, list(h.raw_to_canonical.keys()), n=1, cutoff=self._fuzzy_threshold
            )
            score = difflib.SequenceMatcher(None, norm, nm[0]).ratio() if nm else None
            logger.info("[FUZZY %s] %s -> %s", f"{score:.0%}" if score else "?", label, canonical)
            return _Resolution(tier="FUZZY", canonical=canonical, score=score)
        except EntityNotMappedError:
            return None

    def _auto_resolve(self) -> None:
        for label in self._labels:
            resolution = self._auto_resolve_one(label)
            if resolution is not None:
                self._records[label] = resolution

        n_total = len(self._labels)
        n_resolved = len(self._records)
        n_fuzzy = sum(1 for r in self._records.values() if r.tier == "FUZZY")
        n_fallback = sum(1 for r in self._records.values() if r.tier == "COUNTRY_FALLBACK")
        n_pending = n_total - n_resolved

        logger.info(
            "Resolved %d/%d labels automatically (%d fuzzy, %d country-fallback). "
            "%d require manual mapping.",
            n_resolved, n_total, n_fuzzy, n_fallback, n_pending,
        )
        for label in self._labels:
            if label not in self._records:
                logger.warning(
                    "[UNRESOLVED] %s  — no automatic match found", label
                )

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def pending(self) -> list[str]:
        """Labels not yet resolved, in alphabetical order."""
        return sorted(label for label in self._labels if label not in self._records)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def map(self, mappings: dict[str, str | None]) -> "CanonicalMapper":
        """
        Manually assign canonical entities (or None) to one or more labels.

        Validates all entries before applying any (atomic). Raises ValueError on
        labels not in the original input or invalid canonical values.
        """
        for label, canonical in mappings.items():
            if label not in self._labels:
                raise ValueError(
                    f"Label {label!r} was not in the original input. "
                    f"Create a new CanonicalMapper to add new labels."
                )
            if canonical is not None and canonical not in ALL_CANONICAL_ENTITIES:
                raise ValueError(
                    f"{canonical!r} is not a valid canonical entity. "
                    f"See EntityHierarchy.default().all_canonical_entities for valid values."
                )
        for label, canonical in mappings.items():
            if canonical is None:
                logger.info("[NONE]    %s -> None  (suppressed from evaluation)", label)
                self._records[label] = _Resolution(tier="NONE", canonical=None, score=None)
            else:
                logger.info("[MANUAL]  %s -> %s", label, canonical)
                self._records[label] = _Resolution(tier="MANUAL", canonical=canonical, score=None)
        return self

    # ── Interactive resolution ────────────────────────────────────────────────

    def resolve_interactively(self, *, prompt_fn=input) -> "CanonicalMapper":
        """
        Prompt the user for each pending label, showing ranked fuzzy suggestions.

        Returns self so calls can be chained. No-op when pending is empty.
        """
        to_resolve = list(self.pending)
        if not to_resolve:
            return self

        canonicals = self._hierarchy.all_canonical_entities
        norm_to_canonical = {self._hierarchy._normalize(c): c for c in canonicals}
        normalized_canonicals = list(norm_to_canonical.keys())

        for label in to_resolve:
            norm = self._hierarchy._normalize(label)
            close = difflib.get_close_matches(
                norm, normalized_canonicals, n=5, cutoff=_INTERACTIVE_CUTOFF
            )
            ranked = [
                (norm_to_canonical[s], difflib.SequenceMatcher(None, norm, s).ratio())
                for s in close
            ]

            print(f"\n⚠  No automatic match for: {label!r}")
            if ranked:
                print("   Suggestions:")
                for i, (c, score) in enumerate(ranked, 1):
                    print(f"     {i}. {c}  ({score:.0%})")
            else:
                print("   No close suggestions found.")
            print("   Enter a number, a canonical entity name, or NONE to suppress.")

            while True:
                raw = prompt_fn(f"   → {label}: ").strip()
                if not raw:
                    continue
                if raw.upper() == "NONE":
                    self.map({label: None})
                    break
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(ranked):
                        self.map({label: ranked[idx][0]})
                        break
                    print(f"   Invalid selection. Enter 1–{len(ranked)}.")
                    continue
                try:
                    self.map({label: raw.upper()})
                    break
                except ValueError as e:
                    print(f"   {e}")
        return self

    # ── Output ───────────────────────────────────────────────────────────────

    def get_mapping(self) -> dict[str, str | None]:
        """
        Return the complete label → canonical dict.

        :raises IncompleteMapping: if any labels are still pending.
        """
        if self.pending:
            raise IncompleteMapping(self.pending)
        return {label: self._records[label].canonical for label in self._labels}

    # ── HTML rendering ───────────────────────────────────────────────────────

    def render_html(self) -> None:
        """Render an audit table. Uses IPython.display.HTML in Jupyter; plain text fallback."""
        try:
            from IPython.display import HTML, display
            display(HTML(self._build_html()))
        except ImportError:
            self._print_text()

    def _tier_sort_key(self, label: str) -> int:
        rec = self._records.get(label)
        if rec is None:
            return 0  # pending first
        return {"COUNTRY_FALLBACK": 1, "FUZZY": 2, "EXACT": 3, "COUNTRY": 3, "MANUAL": 3, "NONE": 4}.get(rec.tier, 3)

    _TIER_COLORS: dict[str, tuple[str, str]] = {
        "EXACT":            ("#d1f4e0", "#2da44e"),
        "COUNTRY":          ("#d1f4e0", "#2da44e"),
        "MANUAL":           ("#ddf4ff", "#0969da"),
        "FUZZY":            ("#fff8c5", "#d4a72c"),
        "COUNTRY_FALLBACK": ("#fff8c5", "#b76e00"),
        "NONE":             ("#f6f8fa", "#57606a"),
        "PENDING":          ("#fff1f0", "#cf222e"),
    }

    def _build_html(self) -> str:
        sorted_labels = sorted(self._labels, key=self._tier_sort_key)

        counts: dict[str, int] = {}
        for r in self._records.values():
            counts[r.tier] = counts.get(r.tier, 0) + 1
        pending_count = len(self.pending)

        summary_parts = []
        for tier in ("EXACT", "FUZZY", "COUNTRY", "COUNTRY_FALLBACK", "MANUAL", "NONE"):
            n = counts.get(tier, 0)
            if n:
                _, color = self._TIER_COLORS[tier]
                summary_parts.append(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:10px;font-size:11px;margin:0 3px">'
                    f'{n} {tier.replace("_", "-").lower()}</span>'
                )
        if pending_count:
            _, color = self._TIER_COLORS["PENDING"]
            summary_parts.append(
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:10px;font-size:11px;margin:0 3px">'
                f'{pending_count} pending</span>'
            )

        rows = []
        for label in sorted_labels:
            rec = self._records.get(label)
            tier = rec.tier if rec else "PENDING"
            bg, badge_color = self._TIER_COLORS.get(tier, ("#ffffff", "#666"))
            badge = (
                f'<span style="background:{badge_color};color:white;padding:1px 6px;'
                f'border-radius:3px;font-size:11px;font-family:monospace">'
                f'{"⚠ " if tier == "COUNTRY_FALLBACK" else ""}{tier}</span>'
            )
            if rec is None:
                canonical_display = '<em style="color:#cf222e">pending</em>'
                score_str = "—"
            elif rec.canonical is None:
                canonical_display = '<em style="color:#57606a">None</em>'
                score_str = "—"
            else:
                canonical_display = f'<code>{rec.canonical}</code>'
                score_str = f"{rec.score:.0%}" if rec.score is not None else "—"

            rows.append(
                f'<tr style="background:{bg}">'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;font-family:monospace;font-weight:600">{label}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de">{badge}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;font-family:monospace">{canonical_display}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;text-align:center">{score_str}</td>'
                f'</tr>'
            )

        return (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">'
            '<h3 style="margin-top:0;color:#24292f">Entity Label Mapping</h3>'
            f'<div style="margin:10px 0">{"".join(summary_parts)}</div>'
            '<table style="width:100%;border-collapse:collapse;font-size:13px">'
            '<thead><tr style="background:#f6f8fa;border-bottom:2px solid #d0d7de">'
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:left">Raw Label</th>'
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:left">Tier</th>'
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:left">Canonical</th>'
            '<th style="padding:7px 10px;border:1px solid #d0d7de;text-align:center">Score</th>'
            '</tr></thead><tbody>'
            + "".join(rows)
            + '</tbody></table></div>'
        )

    def _print_text(self) -> None:
        sorted_labels = sorted(self._labels, key=self._tier_sort_key)
        n, p = len(self._labels), len(self.pending)
        print(f"Entity Label Mapping  ({n} total, {p} pending)\n")
        print(f"{'Label':<30} {'Tier':<18} {'Canonical':<30} Score")
        print("-" * 82)
        for label in sorted_labels:
            rec = self._records.get(label)
            tier = rec.tier if rec else "PENDING"
            canonical = str(rec.canonical) if rec else "—"
            score = f"{rec.score:.0%}" if rec and rec.score is not None else "—"
            print(f"{label:<30} {tier:<18} {canonical:<30} {score}")

    def __repr__(self) -> str:
        n, p = len(self._labels), len(self.pending)
        return f"CanonicalMapper({n - p} resolved, {p} pending)"

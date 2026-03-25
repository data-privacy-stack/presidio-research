"""Entity mapping: EntityHierarchy taxonomy and CanonicalMapper workflow."""

from __future__ import annotations

import copy
import difflib
import logging
import re
import warnings
from dataclasses import dataclass

import pandas as pd

from presidio_evaluator.entity_mapping.definitions import (
    COUNTRIES,
    HIERARCHY,
    EntityNotMappedError,
)

logger = logging.getLogger("presidio_evaluator.entity_mapping")
# ---------------------------------------------------------------------------
# BIO/BIOES/BILOU/BILUO prefix/suffix stripping
# ---------------------------------------------------------------------------

_BIO_PREFIX_RE = re.compile(r"^[BIOELSU]-(.+)$", re.IGNORECASE)
_BIO_SUFFIX_RE = re.compile(r"^(.+)-[BIOELSU]$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class IncompleteMapping(RuntimeError):  # noqa: N818
    """Raised by get_mapping() when one or more labels are still pending."""

    def __init__(self, pending: list[str]) -> None:
        self.pending = list(pending)
        super().__init__(
            f"Mapping incomplete: {len(pending)} label(s) still pending: {pending}",
        )


# ---------------------------------------------------------------------------
# EntityHierarchy
# ---------------------------------------------------------------------------


class EntityHierarchy:
    """
    PII entity taxonomy with canonicalization, branch lookup, and customization.

    Wraps a (deep-copied) taxonomy dict and exposes methods to resolve raw
    labels to canonical names, look up their branch in the tree, and add
    aliases to existing entities.

    Example — create a custom variant::

        h = EntityHierarchy()
        h.add_alias("EMAIL_ADDRESS", "ELECTRONIC_MAIL")
        h.canonicalize("ELECTRONIC_MAIL")   # -> 'EMAIL_ADDRESS'
    """

    def __init__(
        self,
        hierarchy: dict | None = None,
        canonical_depth: int = 3,
    ) -> None:
        self.hierarchy: dict = copy.deepcopy(
            hierarchy if hierarchy is not None else HIERARCHY,
        )
        self.countries: set[str] = COUNTRIES
        self.canonical_depth: int = canonical_depth
        self.country_prefixed_doc_types: dict[str, str] = {}
        self._rebuild()

    @staticmethod
    def normalize(label: str) -> str:
        """Normalize a label: strip BIO prefix/suffix, uppercase, remove underscores and dashes."""
        return (
            EntityHierarchy._strip_bio(label).upper().replace("_", "").replace("-", "")
        )

    def canonicalize(self, raw_label: str, threshold: float = 0.80) -> str:
        """
        Return the canonical entity name for a raw label.

        Resolution order: exact alias map → country-prefix → fuzzy fallback.
        Raises EntityNotMappedError if the label cannot be resolved.
        """
        norm = self.normalize(raw_label)
        if norm in self.raw_to_canonical:
            return self.raw_to_canonical[norm]
        country_match = self._country_prefix_canonical(raw_label, threshold=threshold)
        if country_match:
            return country_match
        fuzzy_match = self._fuzzy_resolve(raw_label, threshold)
        if fuzzy_match:
            return fuzzy_match
        raise EntityNotMappedError(f"Unknown entity label: {raw_label!r}")

    def get_branch(self, raw_label: str) -> list[str]:
        """
        Return the full ancestor path for a raw (or canonical) entity label.

        Example: ``get_branch("GERMANY_PASSPORT_NUMBER")`` → ``['PII', 'GOVERNMENT_ID', 'PASSPORT']``
        """
        canonical = self.canonicalize(raw_label)
        branch = self.canonical_to_branch.get(canonical)
        if branch is None:
            raise EntityNotMappedError(
                f"Canonical entity {canonical!r} has no branch in hierarchy",
            )
        return branch

    def add_alias(self, entity_name: str, alias: str) -> None:
        """Add a raw alias for an existing entity."""
        found = self._find_node(entity_name)
        if found is None:
            raise KeyError(f"Entity {entity_name!r} not found in hierarchy")
        parent_dict, key = found
        value = parent_dict[key]
        if isinstance(value, list):
            if alias not in value:
                value.append(alias)
        else:
            value[alias] = []
        self._rebuild()

    @staticmethod
    def _strip_bio(label: str) -> str:
        """Strip a single BIO/BIOES/BILOU/BILUO prefix or suffix (e.g. B-PERSON → PERSON)."""
        m = _BIO_PREFIX_RE.match(label)
        if m:
            return m.group(1)
        m = _BIO_SUFFIX_RE.match(label)
        if m:
            return m.group(1)
        return label

    @staticmethod
    def _collect_all_raw(value) -> list[str]:
        """Recursively collect all raw alias strings from a hierarchy value (list or nested dict)."""
        items: list[str] = []
        if isinstance(value, list):
            items.extend(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                items.append(k)
                items.extend(EntityHierarchy._collect_all_raw(v))
        return items

    @staticmethod
    def _build_alias_map(
        node: dict,
        canonical_depth: int = 3,
        depth: int = 1,
    ) -> dict[str, str]:
        """Build a normalized-label → canonical-name lookup dict by walking the hierarchy tree."""
        mapping: dict[str, str] = {}
        for key, value in node.items():
            if depth >= canonical_depth:
                mapping[EntityHierarchy.normalize(key)] = key
                for alias in EntityHierarchy._collect_all_raw(value):
                    mapping[EntityHierarchy.normalize(alias)] = key
            elif isinstance(value, list):
                mapping[EntityHierarchy.normalize(key)] = key
                for alias in value:
                    mapping[EntityHierarchy.normalize(alias)] = key
            elif isinstance(value, dict):
                mapping[EntityHierarchy.normalize(key)] = key
                mapping.update(
                    EntityHierarchy._build_alias_map(value, canonical_depth, depth + 1),
                )
        return mapping

    @staticmethod
    def _collect_canonical_nodes(
        node: dict,
        canonical_depth: int = 3,
        depth: int = 1,
    ) -> list[str]:
        """Collect the names of all canonical-depth leaf nodes from the hierarchy tree."""
        result: list[str] = []
        for key, value in node.items():
            if depth >= canonical_depth:
                result.append(key)
            elif isinstance(value, list):
                result.append(key)
            elif isinstance(value, dict):
                result.extend(
                    EntityHierarchy._collect_canonical_nodes(
                        value,
                        canonical_depth,
                        depth + 1,
                    ),
                )
        return result

    @staticmethod
    def _build_branch_map(
        node: dict,
        canonical_depth: int = 3,
        current_path: list[str] | None = None,
        depth: int = 1,
    ) -> dict[str, list[str]]:
        """Build a canonical-name → ancestor-path dict by walking the hierarchy tree."""
        if current_path is None:
            current_path = []
        result: dict[str, list[str]] = {}
        for key, value in node.items():
            path = current_path + [key]
            if depth >= canonical_depth:
                result[key] = path
            elif isinstance(value, list):
                result[key] = path
            elif isinstance(value, dict):
                result[key] = path
                result.update(
                    EntityHierarchy._build_branch_map(
                        value,
                        canonical_depth,
                        path,
                        depth + 1,
                    ),
                )
        return result

    def _rebuild(self) -> None:
        """Recompute raw_to_canonical, all_canonical_entities, and canonical_to_branch from the current hierarchy."""
        self.raw_to_canonical: dict[str, str] = self._build_alias_map(
            self.hierarchy,
            self.canonical_depth,
        )
        self.all_canonical_entities: list[str] = self._collect_canonical_nodes(
            self.hierarchy,
            self.canonical_depth,
        )
        self.canonical_to_branch: dict[str, list[str]] = self._build_branch_map(
            self.hierarchy,
            self.canonical_depth,
        )

    def _resolve_remainder(self, remainder: str, threshold: float) -> str:
        """Canonicalize the document-type portion of a country-prefixed label, falling back to NATIONAL_ID."""
        override = self.country_prefixed_doc_types.get(remainder.upper())
        if override:
            return override
        try:
            return self.canonicalize(remainder, threshold=threshold)
        except EntityNotMappedError:
            return "NATIONAL_ID"

    def _country_prefix_canonical(self, raw: str, threshold: float = 1.0) -> str | None:
        """Return the canonical entity for a country-prefixed label using exact country matching, or None."""
        upper = raw.upper()
        parts3 = upper.split("_", 2)
        if len(parts3) == 3 and f"{parts3[0]}_{parts3[1]}" in self.countries:
            return self._resolve_remainder(parts3[2], threshold)
        parts = upper.split("_", 1)
        if len(parts) < 2:
            return None
        country, remainder = parts
        if country not in self.countries:
            return None
        return self._resolve_remainder(remainder, threshold)

    def _fuzzy_country_prefix_canonical(self, raw: str, threshold: float) -> str | None:
        """Return the canonical entity for a country-prefixed label using fuzzy country matching, or None."""
        upper = raw.upper()
        parts3 = upper.split("_", 2)
        if len(parts3) == 3:
            two_token = f"{parts3[0]}_{parts3[1]}"
            two_token_cutoff = max(threshold, 0.90)
            if difflib.get_close_matches(
                two_token,
                self.countries,
                n=1,
                cutoff=two_token_cutoff,
            ):
                return self._resolve_remainder(parts3[2], threshold)
        parts = upper.split("_", 1)
        if len(parts) < 2:
            return None
        prefix, remainder = parts
        if difflib.get_close_matches(prefix, self.countries, n=1, cutoff=threshold):
            return self._resolve_remainder(remainder, threshold)
        return None

    def _fuzzy_resolve(self, raw_label: str, threshold: float) -> str | None:
        """Attempt fuzzy matching against the alias map and fuzzy country prefixes, returning a canonical name or None."""
        country_match = self._fuzzy_country_prefix_canonical(raw_label, threshold)
        if country_match:
            return country_match
        norm = self.normalize(raw_label)
        matches = difflib.get_close_matches(
            norm,
            self.raw_to_canonical,
            n=1,
            cutoff=threshold,
        )
        if matches:
            return self.raw_to_canonical[matches[0]]
        return None

    def _find_node(
        self,
        name: str,
        tree: dict | None = None,
    ) -> tuple[dict, str] | None:
        """Return (parent_dict, key) for the first node matching name in the hierarchy tree, or None."""
        if tree is None:
            tree = self.hierarchy
        for key, value in tree.items():
            if key == name:
                return (tree, key)
            if isinstance(value, dict):
                result = self._find_node(name, value)
                if result:
                    return result
        return None


# ---------------------------------------------------------------------------
# Internal resolution record
# ---------------------------------------------------------------------------


@dataclass
class _Resolution:
    tier: str  # EXACT | FUZZY | COUNTRY | COUNTRY_FALLBACK | MANUAL | NONE
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
        labels: list[str] | set[str] | None = None,
        *,
        hierarchy: dict | EntityHierarchy | None = None,
        canonical_depth: int = 3,
        fuzzy_threshold: float = 0.80,
    ) -> None:
        if isinstance(hierarchy, dict):
            self._hierarchy = EntityHierarchy(
                hierarchy=hierarchy,
                canonical_depth=canonical_depth,
            )
        elif isinstance(hierarchy, EntityHierarchy):
            self._hierarchy = hierarchy
        else:
            self._hierarchy = EntityHierarchy()

        self._fuzzy_threshold = fuzzy_threshold
        self._canonical_depth = canonical_depth

        # Deduplicate, preserving order for lists
        seen: set[str] = set()
        self._labels: list[str] = []
        for label in labels or []:
            if label not in seen:
                seen.add(label)
                self._labels.append(label)

        # BIO-stripped version of each label (original label remains the dict key)
        self._stripped: dict[str, str] = {
            label: EntityHierarchy._strip_bio(label) for label in self._labels
        }

        self._records: dict[str, _Resolution] = {}
        self._auto_resolve()

    def get_mapped_results_dataframe(
        self,
        results_df: pd.DataFrame,
        hierarchy: int | None = None,
    ) -> pd.DataFrame:
        """
        Apply entity mapping to annotation and prediction columns of results_df.

        Extracts unique labels from both columns, auto-resolves any that have
        not been seen before, then returns a new DataFrame with the annotation
        and prediction columns replaced by their canonical equivalents.

        Labels that map to None (suppressed) are passed through unchanged —
        they will appear as false positives or false negatives during evaluation.

        When annotation and prediction map to different canonical entities that
        are related by the hierarchy (e.g., model predicts PERSON but dataset
        annotates FIRSTNAME), a user-friendly warning is emitted.

        :param results_df: DataFrame with at least 'annotation' and 'prediction'
            columns (the 5-column schema returned by model.predict_dataset()).
        :param hierarchy: Canonical depth override. If None, uses the depth set
            at construction time (default 3).
        :return: New DataFrame with remapped annotation and prediction columns.
        """
        if hierarchy is not None and hierarchy != self._canonical_depth:
            # Rebuild hierarchy with the requested depth
            self._canonical_depth = hierarchy
            self._hierarchy = EntityHierarchy(canonical_depth=hierarchy)
            # Re-resolve all known labels with the new depth
            self._records.clear()
            self._auto_resolve()

        # Discover any new labels in the DataFrame, excluding the non-entity "O" tag
        all_labels = [
            label
            for label in (
                list(results_df["annotation"].dropna().unique())
                + list(results_df["prediction"].dropna().unique())
            )
            if label != "O"
        ]
        self._add_labels(all_labels)

        # Build the remapped columns
        mapped_df = results_df.copy()
        mapped_df["annotation"] = results_df["annotation"].map(self._map_tag)
        mapped_df["prediction"] = results_df["prediction"].map(self._map_tag)

        # Warn about mixed-granularity pairs
        self._warn_mixed_granularity(results_df)

        return mapped_df

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def pending(self) -> list[str]:
        """Labels not yet resolved, in alphabetical order."""
        return sorted(label for label in self._labels if label not in self._records)

    # ── Mutation ─────────────────────────────────────────────────────────────

    def map(self, mappings: dict[str, str | None]) -> CanonicalMapper:
        """
        Manually assign canonical entities (or None) to one or more labels.

        Validates all entries before applying any (atomic). Raises ValueError on
        labels not in the original input or invalid canonical values.
        """
        valid_canonicals = set(self._hierarchy.all_canonical_entities) | set(
            self._hierarchy.raw_to_canonical.values(),
        )
        for label, canonical in mappings.items():
            if label not in self._labels:
                raise ValueError(
                    f"Label {label!r} was not in the original input. "
                    f"Create a new CanonicalMapper to add new labels.",
                )
            if canonical is not None and canonical not in valid_canonicals:
                raise ValueError(
                    f"{canonical!r} is not a valid canonical entity. "
                    f"See CanonicalMapper._hierarchy.all_canonical_entities for valid values.",
                )
        for label, canonical in mappings.items():
            if canonical is None:
                logger.info("[NONE]    %s -> None  (suppressed from evaluation)", label)
                self._records[label] = _Resolution(
                    tier="NONE",
                    canonical=None,
                    score=None,
                )
            else:
                logger.info("[MANUAL]  %s -> %s", label, canonical)
                self._records[label] = _Resolution(
                    tier="MANUAL",
                    canonical=canonical,
                    score=None,
                )
        return self

    # ── Interactive resolution ────────────────────────────────────────────────

    def resolve_interactively(self, *, prompt_fn=input) -> CanonicalMapper:
        """
        Prompt the user for each pending label, showing ranked fuzzy suggestions.

        Returns self so calls can be chained. No-op when pending is empty.
        """
        to_resolve = list(self.pending)
        if not to_resolve:
            return self

        canonicals = self._hierarchy.all_canonical_entities
        norm_to_canonical = {self._hierarchy.normalize(c): c for c in canonicals}
        normalized_canonicals = list(norm_to_canonical.keys())

        for label in to_resolve:
            norm = self._hierarchy.normalize(label)
            close = difflib.get_close_matches(
                norm,
                normalized_canonicals,
                n=5,
                cutoff=_INTERACTIVE_CUTOFF,
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

    def get_mapping(
        self,
        mode: str | None = None,
    ) -> dict[str, str | None] | str:
        """
        Return the entity label mapping.

        Without arguments (default) returns a ``dict[str, str | None]`` mapping
        each label to its canonical equivalent (or ``None`` for suppressed labels).
        Raises :class:`IncompleteMapping` if any labels are still pending.

        :param mode: Optional rendering mode.
            * ``None`` (default) — return ``dict[str, str | None]``
            * ``"html"`` — return an HTML ``<table>`` string; pending labels are
              shown as ``(pending)`` without raising.
            * ``"text"`` — return a plain-text table string; pending labels are
              shown as ``(pending)`` without raising.
        :raises IncompleteMapping: if ``mode`` is ``None`` and any labels are pending.
        :return: Mapping dict, HTML string, or plain-text string depending on mode.
        """
        if mode is None:
            if self.pending:
                raise IncompleteMapping(self.pending)
            return {label: self._records[label].canonical for label in self._labels}
        if mode == "html":
            return self._build_html()
        if mode == "text":
            return self._build_text()
        raise ValueError(f"Unknown mode {mode!r}. Use None, 'html', or 'text'.")

    def render_html(self) -> None:
        """Render an audit table. Uses IPython.display.HTML in Jupyter; plain text fallback."""
        try:
            from IPython.display import HTML, display

            display(HTML(self._build_html()))
        except ImportError:
            self._print_text()

    def __repr__(self) -> str:
        n, p = len(self._labels), len(self.pending)
        return f"CanonicalMapper({n - p} resolved, {p} pending)"

    # ── Auto-resolve pass ────────────────────────────────────────────────────

    def _auto_resolve_one(self, label: str) -> _Resolution | None:
        """Return a _Resolution for *label* if auto-resolvable, else None."""
        h = self._hierarchy
        stripped = self._stripped[label]

        # Handle the outside token (O) — never enters the resolution pipeline
        if stripped.upper() == "O":
            logger.info("[NONE] O -> None  (outside token)")
            return _Resolution(tier="NONE", canonical=None, score=None)

        norm = h.normalize(stripped)

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
            len(parts3) == 3 and f"{parts3[0]}_{parts3[1]}" in h.countries
        ) or (len(parts2) >= 2 and parts2[0] in h.countries)
        if has_country:
            result = h._country_prefix_canonical(stripped, threshold=1.0)
            if result is not None:
                if result == "NATIONAL_ID":
                    logger.warning(
                        "[COUNTRY-FALLBACK] %s -> NATIONAL_ID  ⚠ document type not recognized",
                        label,
                    )
                    return _Resolution(
                        tier="COUNTRY_FALLBACK",
                        canonical="NATIONAL_ID",
                        score=None,
                    )
                logger.info("[COUNTRY] %s -> %s", label, result)
                return _Resolution(tier="COUNTRY", canonical=result, score=None)

        # Tier 4: fuzzy (alias or fuzzy country-prefix)
        try:
            canonical = h.canonicalize(stripped, threshold=self._fuzzy_threshold)
            nm = difflib.get_close_matches(
                norm,
                list(h.raw_to_canonical.keys()),
                n=1,
                cutoff=self._fuzzy_threshold,
            )
            score = difflib.SequenceMatcher(None, norm, nm[0]).ratio() if nm else None
            logger.info(
                "[FUZZY %s] %s -> %s",
                f"{score:.0%}" if score else "?",
                label,
                canonical,
            )
            return _Resolution(tier="FUZZY", canonical=canonical, score=score)
        except EntityNotMappedError:
            return None

    def _auto_resolve(self, new_labels: list[str] | None = None) -> None:
        """Run the auto-resolution pass over all labels (or only new_labels) and log a summary."""
        labels_to_process = new_labels if new_labels is not None else self._labels
        for label in labels_to_process:
            resolution = self._auto_resolve_one(label)
            if resolution is not None:
                self._records[label] = resolution

        if new_labels is None:
            # Initial pass — log summary
            n_total = len(self._labels)
            n_resolved = len(self._records)
            n_fuzzy = sum(1 for r in self._records.values() if r.tier == "FUZZY")
            n_fallback = sum(
                1 for r in self._records.values() if r.tier == "COUNTRY_FALLBACK"
            )
            n_pending = n_total - n_resolved

            logger.info(
                "Resolved %d/%d labels automatically (%d fuzzy, %d country-fallback). "
                "%d require manual mapping.",
                n_resolved,
                n_total,
                n_fuzzy,
                n_fallback,
                n_pending,
            )
            for label in self._labels:
                if label not in self._records:
                    logger.warning("[UNRESOLVED] %s  — no automatic match found", label)

    def _add_labels(self, new_labels: list[str]) -> None:
        """Incrementally add new labels that have not been seen before."""
        truly_new: list[str] = []
        seen = set(self._labels)
        for label in new_labels:
            if label not in seen:
                seen.add(label)
                self._labels.append(label)
                self._stripped[label] = EntityHierarchy._strip_bio(label)
                truly_new.append(label)
        if truly_new:
            self._auto_resolve(new_labels=truly_new)

    def _map_tag(self, tag: str) -> str:
        """Return the canonical form of a tag, or the original tag if not resolved."""
        if tag not in self._records:
            return tag  # label was never added or is pending
        canonical = self._records[tag].canonical
        return canonical if canonical is not None else tag

    def _warn_mixed_granularity(self, results_df: pd.DataFrame) -> None:
        """Emit a warning for annotation/prediction pairs at different hierarchy levels."""
        mismatched_pairs: set[tuple[str, str]] = set()
        for ann, pred in zip(
            results_df["annotation"], results_df["prediction"], strict=False
        ):
            if ann == "O" or pred == "O":
                continue
            mapped_ann = self._map_tag(ann)
            mapped_pred = self._map_tag(pred)
            if mapped_ann == mapped_pred:
                continue
            # Skip labels that were never resolved (still pending)
            if ann not in self._records or pred not in self._records:
                continue
            # Check if they share a common ancestor (related by hierarchy)
            try:
                branch_ann = self._hierarchy.get_branch(mapped_ann)
                branch_pred = self._hierarchy.get_branch(mapped_pred)
                if set(branch_ann) & set(branch_pred):
                    mismatched_pairs.add((ann, pred))
            except Exception:  # noqa: S110 — intentional: hierarchy lookup may fail for unknown labels
                pass

        if mismatched_pairs:
            pair_str = ", ".join(f"{a!r} vs {p!r}" for a, p in sorted(mismatched_pairs))
            warnings.warn(
                f"Some annotations and predictions resolve to different canonical entities: {pair_str}. "
                f"This will count as false negatives/positives. To fix this: "
                f"(a) call get_mapped_results_dataframe(results_df, hierarchy=2) to use a broader "
                f"mapping level where both resolve to the same entity, or "
                f"(b) call mapper.map({{'<label>': '<canonical>'}}) to manually specify the mapping.",
                UserWarning,
                stacklevel=3,
            )

    # ── HTML rendering ───────────────────────────────────────────────────────

    def _tier_sort_key(self, label: str) -> int:
        """Return a sort key that orders labels by tier: pending first, then fallback/fuzzy/resolved/suppressed."""
        rec = self._records.get(label)
        if rec is None:
            return 0  # pending first
        return {
            "COUNTRY_FALLBACK": 1,
            "FUZZY": 2,
            "EXACT": 3,
            "COUNTRY": 3,
            "MANUAL": 3,
            "NONE": 4,
        }.get(rec.tier, 3)

    _TIER_COLORS: dict[str, tuple[str, str]] = {
        "EXACT": ("#d1f4e0", "#2da44e"),
        "COUNTRY": ("#d1f4e0", "#2da44e"),
        "MANUAL": ("#ddf4ff", "#0969da"),
        "FUZZY": ("#fff8c5", "#d4a72c"),
        "COUNTRY_FALLBACK": ("#fff8c5", "#b76e00"),
        "NONE": ("#f6f8fa", "#57606a"),
        "PENDING": ("#fff1f0", "#cf222e"),
    }

    def _build_html(self) -> str:
        """Build and return a self-contained HTML table string showing all labels, tiers, and canonical mappings."""
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
                    f"{n} {tier.replace('_', '-').lower()}</span>",
                )
        if pending_count:
            _, color = self._TIER_COLORS["PENDING"]
            summary_parts.append(
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:10px;font-size:11px;margin:0 3px">'
                f"{pending_count} pending</span>",
            )

        rows = []
        for label in sorted_labels:
            rec = self._records.get(label)
            tier = rec.tier if rec else "PENDING"
            bg, badge_color = self._TIER_COLORS.get(tier, ("#ffffff", "#666"))
            badge = (
                f'<span style="background:{badge_color};color:white;padding:1px 6px;'
                f'border-radius:3px;font-size:11px;font-family:monospace">'
                f"{'⚠ ' if tier == 'COUNTRY_FALLBACK' else ''}{tier}</span>"
            )
            if rec is None:
                canonical_display = '<em style="color:#cf222e">pending</em>'
                score_str = "—"
            elif rec.canonical is None:
                canonical_display = '<em style="color:#57606a">None</em>'
                score_str = "—"
            else:
                canonical_display = f"<code>{rec.canonical}</code>"
                score_str = f"{rec.score:.0%}" if rec.score is not None else "—"

            rows.append(
                f'<tr style="background:{bg}">'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;font-family:monospace;font-weight:600">{label}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de">{badge}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;font-family:monospace">{canonical_display}</td>'
                f'<td style="padding:7px 10px;border:1px solid #d0d7de;text-align:center">{score_str}</td>'
                f"</tr>",
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
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        )

    def _print_text(self) -> None:
        """Print a plain-text audit table to stdout."""
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

    def _build_text(self) -> str:
        """Return a plain-text table string suitable for terminal output.

        Pending labels are shown as ``(pending)`` in the canonical column.
        """
        sorted_labels = sorted(self._labels, key=self._tier_sort_key)
        n, p = len(self._labels), len(self.pending)
        lines = [
            f"Entity Label Mapping  ({n} total, {p} pending)",
            "",
            f"{'Label':<30} {'Tier':<18} {'Canonical':<30} Score",
            "-" * 82,
        ]
        for label in sorted_labels:
            rec = self._records.get(label)
            tier = rec.tier if rec else "PENDING"
            canonical = (
                "(pending)"
                if rec is None
                else (str(rec.canonical) if rec.canonical is not None else "None")
            )
            score = f"{rec.score:.0%}" if rec and rec.score is not None else "—"
            lines.append(f"{label:<30} {tier:<18} {canonical:<30} {score}")
        return "\n".join(lines)

"""Entity mapping: two-phase (Identify → Project) CanonicalMapper."""

from __future__ import annotations

import difflib
import logging
import math
from dataclasses import dataclass

import pandas as pd

from presidio_evaluator.entity_mapping.data_objects import (
    IssueSeverity,
    IssueType,
    MappingIssue,
    ResolutionOption,
)
from presidio_evaluator.entity_mapping.definitions import EntityNotMappedError
from presidio_evaluator.entity_mapping.hierarchy import EntityHierarchy

logger = logging.getLogger("presidio_evaluator.entity_mapping")

# Full-depth hierarchy used for branch lookups and depth calculations.
# Constructed once at module load; never mutated.
_FULL_HIERARCHY = EntityHierarchy(canonical_depth=10)

_SEVERITY_ORDER = {
    IssueSeverity.ERROR: 0,
    IssueSeverity.WARNING: 1,
    IssueSeverity.INFO: 2,
}

_ISSUE_TYPE_ORDER = {
    IssueType.UNRESOLVED: 0,
    IssueType.COLLISION_AMBIGUOUS: 1,
    IssueType.COLLISION_CROSS_BRANCH: 2,
    IssueType.PREDICTION_ONLY: 3,
    IssueType.COLLISION_TRIVIAL: 4,
    IssueType.DATASET_ONLY: 5,
}


@dataclass
class _Resolution:
    tier: str  # EXACT | FUZZY | COUNTRY | COUNTRY_FALLBACK | MANUAL | NONE | UNRESOLVED
    canonical: str | None
    score: float | None  # similarity score for FUZZY; None otherwise
    projected: str | None = None  # final eval-surface entity after projection
    projection_type: str | None = (
        None  # EXACT | TRIVIAL | AMBIGUOUS | CROSS_BRANCH | UNRESOLVED | NONE
    )


class IncompleteMapping(RuntimeError):  # noqa: N818
    """Raised when blocking issues remain unresolved."""

    def __init__(self, issues: list[MappingIssue]) -> None:
        self.issues = list(issues)
        labels = [lbl for issue in issues for lbl in issue.labels]
        super().__init__(
            f"Mapping incomplete: {len(issues)} blocking issue(s) remain. "
            f"Affected labels: {labels}. "
            f"Call get_issues() for details.",
        )


class CanonicalMapper:
    """Two-phase (Identify -> Project) entity label mapper.

    Typical workflow::

        mapper = CanonicalMapper()
        mapper.analyze(results_df)
        mapper.render_html()
        for issue in mapper.get_issues():
            print(issue.message)
        mapper.map({"MY_LABEL": "PERSON"})
        mapped_df = mapper.get_mapped_results_dataframe()
    """

    def __init__(
        self,
        *,
        hierarchy: dict | EntityHierarchy | None = None,
        fuzzy_threshold: float = 0.80,
    ) -> None:
        if isinstance(hierarchy, dict):
            self._hierarchy = EntityHierarchy(hierarchy=hierarchy)
        elif isinstance(hierarchy, EntityHierarchy):
            self._hierarchy = hierarchy
        else:
            self._hierarchy = EntityHierarchy()

        self._fuzzy_threshold = fuzzy_threshold

        # Identification state
        self._stripped: dict[str, str] = {}
        self._records: dict[str, _Resolution] = {}

        # Per-label token counts (populated during analyze())
        self._label_annotation_counts: dict[str, int] = {}
        self._label_prediction_counts: dict[str, int] = {}

        # Eval surface (locked after first analyze())
        self._eval_surface: set[str] | None = None
        self._eval_depth: int | None = None

        # Issues and DataFrame (per analyze() call)
        self._issues: list[MappingIssue] = []
        self._results_df: pd.DataFrame | None = None

    # -- Properties -----------------------------------------------------------

    @property
    def eval_surface(self) -> set[str]:
        """The locked evaluation surface (empty set before first analyze())."""
        return set(self._eval_surface) if self._eval_surface is not None else set()

    @property
    def pending(self) -> list[str]:
        """Labels that failed identification (UNRESOLVED), in alphabetical order."""
        return sorted(
            lbl for lbl, rec in self._records.items() if rec.tier == "UNRESOLVED"
        )

    # -- Analysis -------------------------------------------------------------

    def analyze(self, results_df: pd.DataFrame) -> CanonicalMapper:
        """Run the full identify->project pipeline on results_df.

        Phase 1 - Identify: resolve every raw label via
        EXACT -> COUNTRY -> COUNTRY_FALLBACK -> FUZZY -> UNRESOLVED.

        Phase 2 - Project: compute majority-vote depth (first call only),
        lock eval surface, project all identified labels onto it.

        :param results_df: DataFrame with annotation and prediction columns.
        :return: self (for chaining).
        """
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError(
                f"results_df must be a pandas DataFrame, got {type(results_df).__name__}"
            )
        required = {"annotation", "prediction"}
        missing = required - set(results_df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        # Preserve manual mappings across calls
        manual_records: dict[str, _Resolution] = {
            lbl: rec
            for lbl, rec in self._records.items()
            if rec.tier in ("MANUAL", "NONE")
        }

        # Reset per-call state
        self._records = dict(manual_records)
        self._stripped = {
            lbl: EntityHierarchy._strip_bio(lbl) for lbl in manual_records
        }
        self._issues.clear()
        self._label_annotation_counts.clear()
        self._label_prediction_counts.clear()
        self._results_df = results_df

        # Count tokens per label
        for lbl, count in results_df["annotation"].value_counts().items():
            if lbl != "O":
                self._label_annotation_counts[str(lbl)] = int(count)
        for lbl, count in results_df["prediction"].value_counts().items():
            if lbl != "O":
                self._label_prediction_counts[str(lbl)] = int(count)

        # Discover all labels
        raw_labels = [
            lbl
            for lbl in (
                list(results_df["annotation"].dropna().unique())
                + list(results_df["prediction"].dropna().unique())
            )
            if lbl != "O"
        ]
        self._add_labels(raw_labels)

        # Phase 2a: compute eval depth (first analyze() only)
        if self._eval_surface is None:
            self._eval_depth = self._compute_eval_depth(results_df)
            h_eval = EntityHierarchy(
                hierarchy=self._hierarchy.hierarchy,
                canonical_depth=self._eval_depth,
            )
            self._eval_surface = set(h_eval.all_canonical_entities)
            logger.info(
                "Eval surface locked at depth %d: %d entities (%s...)",
                self._eval_depth,
                len(self._eval_surface),
                ", ".join(sorted(self._eval_surface)[:5]),
            )
        else:
            logger.info(
                "Eval surface reused (depth %d, %d entities).",
                self._eval_depth,
                len(self._eval_surface),
            )

        # Phase 2b: project all labels onto eval surface
        self._project_all()

        # Detect cross-surface issues
        self._detect_issues()

        # Summary log
        n_auto = sum(
            1 for r in self._records.values() if r.projection_type == "TRIVIAL"
        )
        n_warn = sum(
            1
            for i in self._issues
            if i.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING)
        )
        logger.info(
            "Analysis complete: depth=%d, surface=%d entities, %d labels, "
            "%d auto-fixes, %d blocking issue(s).",
            self._eval_depth,
            len(self._eval_surface),
            len(self._records),
            n_auto,
            n_warn,
        )

        return self

    # -- Majority-vote depth --------------------------------------------------

    def _compute_eval_depth(self, results_df: pd.DataFrame) -> int:
        """Compute weighted majority-vote depth from annotation entities.

        Formula: round(sum(min(depth, 3) * tokens) / sum(tokens))
        Tie at 0.5: prefer deeper (math.ceil).
        """
        total_tokens = 0
        weighted_sum = 0.0

        for lbl, count in self._label_annotation_counts.items():
            rec = self._records.get(lbl)
            if rec is None or rec.canonical is None:
                continue
            branch = _FULL_HIERARCHY.canonical_to_branch.get(rec.canonical)
            if branch is None:
                continue
            depth = len(branch)
            capped = min(depth, 3)
            weighted_sum += capped * count
            total_tokens += count

        if total_tokens == 0:
            return 3  # default

        raw = weighted_sum / total_tokens
        result = math.ceil(raw) if (raw - int(raw)) >= 0.5 else round(raw)
        return max(2, min(3, result))  # clamp to [2, 3]

    # -- Identification -------------------------------------------------------

    def _add_labels(self, labels: list[str]) -> None:
        """Add new labels and run identification for truly new ones."""
        seen = set(self._records)
        for lbl in labels:
            if lbl not in seen:
                seen.add(lbl)
                self._stripped[lbl] = EntityHierarchy._strip_bio(lbl)
                res = self._auto_resolve_one(lbl)
                if res is not None:
                    self._records[lbl] = res
                else:
                    self._records[lbl] = _Resolution(
                        tier="UNRESOLVED", canonical=None, score=None
                    )
                    logger.warning("[UNRESOLVED] %s  -- no automatic match found", lbl)

    def _auto_resolve_one(self, label: str) -> _Resolution | None:
        """Try to resolve label to a canonical entity. Returns None on failure."""
        h = self._hierarchy
        stripped = self._stripped.get(label, EntityHierarchy._strip_bio(label))

        if stripped.upper() == "O":
            logger.info("[NONE] O -> None  (outside token)")
            return _Resolution(tier="NONE", canonical=None, score=None)

        norm = h.normalize(stripped)

        # Tier 1: exact alias map
        if norm in h.raw_to_canonical:
            canonical = h.raw_to_canonical[norm]
            logger.info("[EXACT]   %s -> %s", label, canonical)
            return _Resolution(tier="EXACT", canonical=canonical, score=None)

        # Tier 2/3: country prefix
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
                    logger.info(
                        "[COUNTRY-FALLBACK] %s -> NATIONAL_ID  "
                        "document type not recognized",
                        label,
                    )
                    return _Resolution(
                        tier="COUNTRY_FALLBACK", canonical="NATIONAL_ID", score=None
                    )
                logger.info("[COUNTRY] %s -> %s", label, result)
                return _Resolution(tier="COUNTRY", canonical=result, score=None)

        # Tier 4: fuzzy
        try:
            canonical = h.canonicalize(stripped, threshold=self._fuzzy_threshold)
            nm = difflib.get_close_matches(
                norm, list(h.raw_to_canonical.keys()), n=1, cutoff=self._fuzzy_threshold
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

    # -- Projection -----------------------------------------------------------

    def _project_all(self) -> None:
        """Project all identified labels onto the eval surface."""
        if self._eval_surface is None:
            return
        eval_surface = self._eval_surface
        h_full = _FULL_HIERARCHY

        for lbl, rec in self._records.items():
            if rec.tier == "NONE":
                rec.projected = None
                rec.projection_type = "NONE"
                continue

            if rec.tier == "UNRESOLVED":
                rec.projected = None
                rec.projection_type = "UNRESOLVED"
                continue

            if rec.tier == "MANUAL":
                if rec.canonical is None:
                    rec.projected = None
                    rec.projection_type = "NONE"
                elif rec.canonical in eval_surface:
                    rec.projected = rec.canonical
                    rec.projection_type = "EXACT"
                else:
                    rec.projected = rec.canonical
                    rec.projection_type = "MANUAL"
                continue

            canonical = rec.canonical
            if canonical is None:
                rec.projected = None
                rec.projection_type = "NONE"
                continue

            # Exact match on eval surface
            if canonical in eval_surface:
                rec.projected = canonical
                rec.projection_type = "EXACT"
                continue

            # Check ancestry using full-depth hierarchy
            full_branch = h_full.canonical_to_branch.get(canonical, [])

            # Descendant of an eval-surface entity?
            found_ancestor = None
            for ancestor in reversed(full_branch[:-1]):
                if ancestor in eval_surface:
                    found_ancestor = ancestor
                    break

            if found_ancestor is not None:
                logger.info(
                    "[AUTO-FIX] %s -> %s  (descendant projected to eval-surface entity)",
                    lbl,
                    found_ancestor,
                )
                rec.projected = found_ancestor
                rec.projection_type = "TRIVIAL"
                continue

            # Ancestor of eval-surface entities?
            descendants_on_surface = [
                e
                for e in eval_surface
                if canonical in (h_full.canonical_to_branch.get(e) or [])
            ]
            if len(descendants_on_surface) == 1:
                target = descendants_on_surface[0]
                logger.info(
                    "[AUTO-FIX] %s -> %s  "
                    "(ancestor projected to sole eval-surface descendant)",
                    lbl,
                    target,
                )
                rec.projected = target
                rec.projection_type = "TRIVIAL"
            elif len(descendants_on_surface) > 1:
                rec.projected = None
                rec.projection_type = "AMBIGUOUS"
            else:
                # Different branch entirely
                rec.projected = None
                rec.projection_type = "CROSS_BRANCH"

    # -- Issue detection ------------------------------------------------------

    def _detect_issues(self) -> None:
        """Detect all issues after projection and sort them."""
        if self._eval_surface is None:
            return
        self._issues.clear()

        h_full = _FULL_HIERARCHY

        # UNRESOLVED (ERROR)
        for lbl, rec in self._records.items():
            if rec.projection_type != "UNRESOLVED":
                continue
            n_ann = self._label_annotation_counts.get(lbl, 0)
            n_pred = self._label_prediction_counts.get(lbl, 0)
            norm = self._hierarchy.normalize(EntityHierarchy._strip_bio(lbl))
            close = difflib.get_close_matches(
                norm, list(self._hierarchy.raw_to_canonical.keys()), n=3, cutoff=0.4
            )
            options = [
                ResolutionOption(
                    action="map_to_canonical",
                    description=f"Map {lbl!r} to {self._hierarchy.raw_to_canonical[m]!r}",
                    mapper_call={lbl: self._hierarchy.raw_to_canonical[m]},
                )
                for m in close
            ]
            options.append(
                ResolutionOption(
                    action="suppress",
                    description=f"Suppress {lbl!r} from evaluation",
                    mapper_call={lbl: None},
                )
            )
            self._issues.append(
                MappingIssue(
                    type=IssueType.UNRESOLVED,
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Label {lbl!r} could not be resolved to any canonical entity "
                        f"({n_ann + n_pred} tokens). "
                        f"Call mapper.map({{{lbl!r}: 'CANONICAL'}}) to fix."
                    ),
                    labels=[lbl],
                    annotation_count=n_ann,
                    prediction_count=n_pred,
                    resolution_options=options,
                )
            )

        # COLLISION_TRIVIAL (INFO)
        for lbl, rec in self._records.items():
            if rec.projection_type != "TRIVIAL":
                continue
            n_ann = self._label_annotation_counts.get(lbl, 0)
            n_pred = self._label_prediction_counts.get(lbl, 0)
            self._issues.append(
                MappingIssue(
                    type=IssueType.COLLISION_TRIVIAL,
                    severity=IssueSeverity.INFO,
                    message=(
                        f"[AUTO-FIX] {lbl!r} -> {rec.projected!r} "
                        f"(same-branch projection, {n_ann + n_pred} tokens)"
                    ),
                    labels=[lbl],
                    annotation_count=n_ann,
                    prediction_count=n_pred,
                )
            )

        # COLLISION_AMBIGUOUS (WARNING)
        for lbl, rec in self._records.items():
            if rec.projection_type != "AMBIGUOUS":
                continue
            canonical = rec.canonical
            n_ann = self._label_annotation_counts.get(lbl, 0)
            n_pred = self._label_prediction_counts.get(lbl, 0)
            descendants = [
                e
                for e in self._eval_surface
                if canonical in (h_full.canonical_to_branch.get(e) or [])
            ]
            overlap_counts = self._compute_overlap(lbl, descendants)
            sorted_candidates = sorted(
                descendants, key=lambda e: overlap_counts.get(e, 0), reverse=True
            )
            options = [
                ResolutionOption(
                    action="map_to_canonical",
                    description=f"Map {lbl!r} to {cand!r} ({overlap_counts.get(cand, 0)} token co-occurrences)",
                    mapper_call={lbl: cand},
                )
                for cand in sorted_candidates
            ]
            overlap_str = ", ".join(
                f"{e}: {overlap_counts.get(e, 0)}" for e in sorted_candidates
            )
            self._issues.append(
                MappingIssue(
                    type=IssueType.COLLISION_AMBIGUOUS,
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{lbl!r} maps to {canonical!r} which is an ancestor of multiple "
                        f"eval-surface entities: {sorted_candidates}. "
                        f"Token overlap: [{overlap_str}]. "
                        f"Call mapper.map({{{lbl!r}: 'TARGET'}}) to resolve."
                    ),
                    labels=[lbl],
                    annotation_count=n_ann,
                    prediction_count=n_pred,
                    resolution_options=options,
                    overlap_counts=overlap_counts,
                )
            )

        # COLLISION_CROSS_BRANCH (WARNING)
        for lbl, rec in self._records.items():
            if rec.projection_type != "CROSS_BRANCH":
                continue
            canonical = rec.canonical
            n_ann = self._label_annotation_counts.get(lbl, 0)
            n_pred = self._label_prediction_counts.get(lbl, 0)
            overlap_counts = self._compute_overlap(lbl, list(self._eval_surface))
            top_overlap = sorted(
                overlap_counts.items(), key=lambda x: x[1], reverse=True
            )[:3]
            options = [
                ResolutionOption(
                    action="map_to_canonical",
                    description=f"Remap {lbl!r} to {e!r} ({cnt} token co-occurrences)",
                    mapper_call={lbl: e},
                )
                for e, cnt in top_overlap
                if cnt > 0
            ]
            options.append(
                ResolutionOption(
                    action="suppress",
                    description=f"Suppress {lbl!r} from evaluation",
                    mapper_call={lbl: None},
                )
            )
            insight = ""
            if top_overlap and top_overlap[0][1] > 50:
                insight = (
                    f" Systematic misprediction: {lbl!r} predicted on "
                    f"{top_overlap[0][1]} {top_overlap[0][0]!r} tokens."
                )
            self._issues.append(
                MappingIssue(
                    type=IssueType.COLLISION_CROSS_BRANCH,
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{lbl!r} ({canonical!r}) is on a different hierarchy branch "
                        f"from all eval-surface entities.{insight} "
                        f"Call mapper.map({{{lbl!r}: 'TARGET'}}) to remap or suppress."
                    ),
                    labels=[lbl],
                    annotation_count=n_ann,
                    prediction_count=n_pred,
                    resolution_options=options,
                    overlap_counts=dict(top_overlap),
                )
            )

        # PREDICTION_ONLY (WARNING)
        predicted_projected = {
            rec.projected
            for lbl, rec in self._records.items()
            if self._label_prediction_counts.get(lbl, 0) > 0
            and rec.projected is not None
        }
        annotated_projected = {
            rec.projected
            for lbl, rec in self._records.items()
            if self._label_annotation_counts.get(lbl, 0) > 0
            and rec.projected is not None
        }
        prediction_only_entities = predicted_projected - annotated_projected

        for entity in sorted(prediction_only_entities):
            raw_labels = [
                lbl
                for lbl, rec in self._records.items()
                if rec.projected == entity
                and self._label_prediction_counts.get(lbl, 0) > 0
            ]
            n_pred = sum(
                self._label_prediction_counts.get(lbl, 0) for lbl in raw_labels
            )
            options = [
                ResolutionOption(
                    action="suppress",
                    description=f"Suppress {entity!r} from evaluation",
                    mapper_call=dict.fromkeys(raw_labels),
                ),
            ]
            self._issues.append(
                MappingIssue(
                    type=IssueType.PREDICTION_ONLY,
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{entity!r} appears in predictions but not in the eval surface "
                        f"({n_pred} prediction tokens). "
                        f"Suppress: mapper.map({{lbl: None}}), "
                        f"or remap: mapper.map({{lbl: 'TARGET'}})."
                    ),
                    labels=raw_labels,
                    annotation_count=0,
                    prediction_count=n_pred,
                    resolution_options=options,
                )
            )

        # DATASET_ONLY (INFO)
        # Entities that ARE annotated (in annotated_projected) but have NO predictions
        # projecting to them. Intersect with eval_surface to exclude manual off-surface
        # projections from the count.
        dataset_only_entities = (
            annotated_projected - predicted_projected
        ) & self._eval_surface
        for entity in sorted(dataset_only_entities):
            raw_labels = [
                lbl
                for lbl, rec in self._records.items()
                if rec.projected == entity
                and self._label_annotation_counts.get(lbl, 0) > 0
            ]
            n_ann = sum(self._label_annotation_counts.get(lbl, 0) for lbl in raw_labels)
            self._issues.append(
                MappingIssue(
                    type=IssueType.DATASET_ONLY,
                    severity=IssueSeverity.INFO,
                    message=(
                        f"{entity!r} is in the eval surface but no prediction maps to it. "
                        f"{n_ann} annotation tokens -- all will count as false negatives."
                    ),
                    labels=raw_labels or [entity],
                    annotation_count=n_ann,
                    prediction_count=0,
                )
            )

        # Sort: severity, then issue type order, then token count desc
        self._issues.sort(
            key=lambda i: (
                _SEVERITY_ORDER[i.severity],
                _ISSUE_TYPE_ORDER.get(i.type, 99),
                -((i.annotation_count or 0) + (i.prediction_count or 0)),
            )
        )

    def _compute_overlap(
        self, prediction_label: str, candidates: list[str]
    ) -> dict[str, int]:
        """Count how many tokens have prediction_label co-occurring with each candidate annotation."""
        if self._results_df is None or not candidates:
            return dict.fromkeys(candidates, 0)
        counts: dict[str, int] = dict.fromkeys(candidates, 0)
        df = self._results_df
        mask = df["prediction"] == prediction_label
        for ann_lbl in df.loc[mask, "annotation"].unique():
            if ann_lbl == "O":
                continue
            rec = self._records.get(ann_lbl)
            projected = rec.projected if rec else None
            if projected in counts:
                n = int((mask & (df["annotation"] == ann_lbl)).sum())
                counts[projected] += n
        return counts

    # -- Issues ---------------------------------------------------------------

    def get_issues(self) -> list[MappingIssue]:
        """Return issues from the last analyze() call, sorted by severity then token count."""
        return list(self._issues)

    # -- Mutation -------------------------------------------------------------

    def map(self, mappings: dict[str, str | None]) -> CanonicalMapper:
        """Assign projected entities (or None) to one or more labels.

        Validates all entries atomically before applying any.
        Returns self for chaining.
        """
        valid_canonicals = set(self._hierarchy.all_canonical_entities) | set(
            self._hierarchy.raw_to_canonical.values()
        )
        if self._eval_surface:
            valid_canonicals |= self._eval_surface

        for _lbl, canonical in mappings.items():
            if canonical is not None and canonical not in valid_canonicals:
                raise ValueError(f"{canonical!r} is not a valid canonical entity.")

        for lbl, canonical in mappings.items():
            if canonical is None:
                logger.info("[NONE]    %s -> None  (suppressed from evaluation)", lbl)
                self._records[lbl] = _Resolution(
                    tier="NONE",
                    canonical=None,
                    score=None,
                    projected=None,
                    projection_type="NONE",
                )
            else:
                logger.info("[MANUAL]  %s -> %s", lbl, canonical)
                proj_type = (
                    "EXACT"
                    if (self._eval_surface and canonical in self._eval_surface)
                    else "MANUAL"
                )
                self._records[lbl] = _Resolution(
                    tier="MANUAL",
                    canonical=canonical,
                    score=None,
                    projected=canonical,
                    projection_type=proj_type,
                )

        if self._results_df is not None and self._eval_surface is not None:
            self._detect_issues()

        return self

    def resolve_interactively(self, prompt_fn=input) -> CanonicalMapper:
        """Prompt the user to resolve all WARNING+ issues interactively.

        :param prompt_fn: callable(prompt_str) -> str. Defaults to built-in input.
        :return: self for chaining.
        """
        blocking = [
            i
            for i in self._issues
            if i.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING)
        ]
        if not blocking:
            return self

        for issue in list(blocking):
            print(f"\n[{issue.severity.value.upper()}] {issue.type.value}")
            print(f"  {issue.message}")
            tokens = (issue.annotation_count or 0) + (issue.prediction_count or 0)
            print(f"  Affected tokens: {tokens}")
            if issue.resolution_options:
                print("  Suggestions:")
                for i, opt in enumerate(issue.resolution_options, 1):
                    print(f"    {i}. {opt.description}")

            lbl = issue.labels[0] if issue.labels else None
            if lbl is None:
                continue

            while True:
                raw = prompt_fn(
                    f"  Enter suggestion number, entity name, or NONE [{lbl}]: "
                ).strip()
                if raw.upper() == "NONE":
                    self.map({lbl: None})
                    break
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(issue.resolution_options):
                        opt = issue.resolution_options[idx]
                        if opt.mapper_call:
                            self.map(opt.mapper_call)
                        break
                    print("  Invalid number. Try again.")
                    continue
                try:
                    self.map({lbl: raw})
                    break
                except ValueError as e:
                    print(f"  Invalid: {e}")

        return self

    # -- Output ---------------------------------------------------------------

    def get_mapping(
        self, entity: str | None = None
    ) -> dict[str, str | None] | str | None:
        """Return the current mapping.

        :param entity: A single raw label to look up (returns its projected entity).
        :return: dict of all label->projected mappings, or a single projected entity.
        """
        if entity is not None:
            rec = self._records.get(entity)
            if rec is not None:
                return rec.projected
            try:
                return self._hierarchy.canonicalize(entity)
            except EntityNotMappedError:
                raise ValueError(
                    f"Cannot resolve {entity!r}. "
                    f"Use mapper.map({{{entity!r}: 'CANONICAL'}}) to map it."
                ) from None

        return {lbl: rec.projected for lbl, rec in self._records.items()}

    def get_mapped_results_dataframe(self) -> pd.DataFrame:
        """Return the analyzed DataFrame with annotation/prediction remapped.

        :raises RuntimeError: if analyze() has not been called.
        :raises IncompleteMapping: if any ERROR or WARNING issues remain.
        """
        if self._results_df is None:
            raise RuntimeError(
                "No DataFrame available. Call analyze(results_df) first."
            )
        blocking = [
            i
            for i in self._issues
            if i.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING)
        ]
        if blocking:
            raise IncompleteMapping(blocking)

        mapped_df = self._results_df.copy()
        mapped_df["original_annotation"] = self._results_df["annotation"]
        mapped_df["original_prediction"] = self._results_df["prediction"]
        mapped_df["annotation"] = self._results_df["annotation"].map(self._map_tag)
        mapped_df["prediction"] = self._results_df["prediction"].map(self._map_tag)
        return mapped_df

    def _map_tag(self, tag: str) -> str:
        """Return the projected form of a tag, 'O' for suppressed/unresolved."""
        if tag == "O":
            return "O"
        rec = self._records.get(tag)
        if rec is None:
            return tag
        if rec.projected is None:
            return "O"
        return rec.projected

    def render_html(self) -> None:
        """Render an audit table via MapperRenderer.

        Convenience wrapper. For more control use
        ``MapperRenderer(mapper).render()`` directly.
        """
        from presidio_evaluator.entity_mapping.rendering import (  # noqa: PLC0415
            MapperRenderer,
        )

        MapperRenderer(self).render()

    def __repr__(self) -> str:
        n = len(self._records)
        n_blocking = sum(
            1
            for i in self._issues
            if i.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING)
        )
        return f"CanonicalMapper({n} labels, {n_blocking} blocking issues)"

"""Entity mapping: single-phase (Identify-only) CanonicalMapper."""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass

import pandas as pd

from presidio_evaluator.entity_mapping.data_objects import (
    IssueSeverity,
    IssueType,
    MappedResults,
    MappingIssue,
    ResolutionOption,
)
from presidio_evaluator.entity_mapping.definitions import EntityNotMappedError
from presidio_evaluator.entity_mapping.hierarchy import EntityHierarchy
from presidio_evaluator.entity_mapping.level_helpers import to_binary, to_branch

logger = logging.getLogger("presidio_evaluator.entity_mapping")


def _get_renderer_class():  # noqa: ANN201
    """Lazy import of MapperRenderer to avoid circular imports at module load time."""
    from presidio_evaluator.entity_mapping.rendering import MapperRenderer  # noqa: PLC0415, I001

    return MapperRenderer


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
    IssueType.COLLISION_CROSS_BRANCH: 1,
    IssueType.PREDICTION_ONLY: 2,
    IssueType.DATASET_ONLY: 3,
    IssueType.COLLISION_SAME_BRANCH: 4,
}


@dataclass
class _Resolution:
    tier: str  # EXACT | FUZZY | COUNTRY | COUNTRY_FALLBACK | MANUAL | NONE | UNRESOLVED
    resolved: str | None  # the canonical hierarchy node this label maps to
    score: float | None  # similarity score for FUZZY; None otherwise


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
    """Single-phase (Identify-only) entity label mapper.

    Typical workflow::

        mapper = CanonicalMapper()
        mapper.analyze(results_df)
        mapper.render_html()
        for issue in mapper.get_issues():
            print(issue.message)
        mapper.map({"MY_LABEL": "PERSON"})
        results = mapper.get_mapped_results_dataframe()
        # results.original, results.binary, results.branch, results.detailed
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

        # Issues and DataFrame (per analyze() call)
        self._issues: list[MappingIssue] = []
        self._results_df: pd.DataFrame | None = None
        self._min_severity: IssueSeverity = IssueSeverity.WARNING

    # -- Properties -----------------------------------------------------------

    @property
    def pending(self) -> list[str]:
        """Labels that failed identification (UNRESOLVED), in alphabetical order."""
        return sorted(
            lbl for lbl, rec in self._records.items() if rec.tier == "UNRESOLVED"
        )

    # -- Analysis -------------------------------------------------------------

    def analyze(
        self,
        results_df: pd.DataFrame,
        min_severity: str | IssueSeverity = "WARNING",
    ) -> CanonicalMapper:
        """Identify all labels in the hierarchy (single-phase, no projection).

        Resolves every raw label via
        EXACT -> COUNTRY -> COUNTRY_FALLBACK -> FUZZY -> UNRESOLVED.

        :param results_df: DataFrame with annotation and prediction columns.
        :param min_severity: Minimum severity to surface via get_issues() and
            render_html(). Accepts 'ERROR', 'WARNING', 'INFO' (or IssueSeverity
            enum values). Default is 'WARNING'. COLLISION_SAME_BRANCH (INFO)
            is only shown when min_severity='INFO'.
        :return: self (for chaining).
        :raises ValueError: If min_severity is an unrecognised string.
        """
        # Validate min_severity
        if isinstance(min_severity, str):
            try:
                self._min_severity = IssueSeverity(min_severity.lower())
            except ValueError:
                valid = [s.value for s in IssueSeverity]
                raise ValueError(
                    f"Unrecognised severity {min_severity!r}. Valid values: {valid}"
                ) from None
        elif isinstance(min_severity, IssueSeverity):
            self._min_severity = min_severity
        else:
            raise TypeError(
                f"min_severity must be str or IssueSeverity, got {type(min_severity).__name__}"
            )
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

        # Discover all labels and identify them
        raw_labels = [
            lbl
            for lbl in (
                list(results_df["annotation"].dropna().unique())
                + list(results_df["prediction"].dropna().unique())
            )
            if lbl != "O"
        ]
        self._add_labels(raw_labels)

        # Detect issues
        self._detect_issues()

        n_warn = sum(
            1
            for i in self._issues
            if i.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING)
        )
        logger.info(
            "Analysis complete: %d labels, %d issue(s) requiring attention.",
            len(self._records),
            n_warn,
        )

        return self

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
                        tier="UNRESOLVED", resolved=None, score=None
                    )
                    logger.warning("[UNRESOLVED] %s  -- no automatic match found", lbl)

    def _auto_resolve_one(self, label: str) -> _Resolution | None:
        """Try to resolve label to a canonical entity. Returns None on failure."""
        h = self._hierarchy
        stripped = self._stripped.get(label, EntityHierarchy._strip_bio(label))

        if stripped.upper() == "O":
            logger.info("[NONE] O -> None  (outside token)")
            return _Resolution(tier="NONE", resolved=None, score=None)

        norm = h.normalize(stripped)

        # Tier 1: exact alias map
        if norm in h.raw_to_canonical:
            resolved = h.raw_to_canonical[norm]
            logger.info("[EXACT]   %s -> %s", label, resolved)
            return _Resolution(tier="EXACT", resolved=resolved, score=None)

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
                        tier="COUNTRY_FALLBACK", resolved="NATIONAL_ID", score=None
                    )
                logger.info("[COUNTRY] %s -> %s", label, result)
                return _Resolution(tier="COUNTRY", resolved=result, score=None)

        # Tier 4: fuzzy
        try:
            resolved = h.canonicalize(stripped, threshold=self._fuzzy_threshold)
            nm = difflib.get_close_matches(
                norm, list(h.raw_to_canonical.keys()), n=1, cutoff=self._fuzzy_threshold
            )
            score = difflib.SequenceMatcher(None, norm, nm[0]).ratio() if nm else None
            logger.info(
                "[FUZZY %s] %s -> %s",
                f"{score:.0%}" if score else "?",
                label,
                resolved,
            )
            return _Resolution(tier="FUZZY", resolved=resolved, score=score)
        except EntityNotMappedError:
            return None

    # -- Issue detection ------------------------------------------------------

    def _detect_issues(self) -> None:
        """Detect all issues (single-phase identification) and sort them."""
        self._issues.clear()
        h_full = _FULL_HIERARCHY

        def _branch_key(resolved: str | None) -> str | None:
            if not resolved:
                return None
            path = h_full.canonical_to_branch.get(resolved, [])
            return path[1] if len(path) >= 2 else None

        # ── UNRESOLVED (ERROR) ──────────────────────────────────────────────
        for lbl, rec in self._records.items():
            if rec.tier != "UNRESOLVED":
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

        # ── Build branch sets for PREDICTION_ONLY / DATASET_ONLY ───────────
        annotation_branches: set[str] = set()
        for lbl, rec in self._records.items():
            if self._label_annotation_counts.get(lbl, 0) == 0:
                continue
            bk = _branch_key(rec.resolved)
            if bk:
                annotation_branches.add(bk)

        prediction_branches: set[str] = set()
        for lbl, rec in self._records.items():
            if self._label_prediction_counts.get(lbl, 0) == 0:
                continue
            bk = _branch_key(rec.resolved)
            if bk:
                prediction_branches.add(bk)

        # ── COLLISION_CROSS_BRANCH (WARNING) ────────────────────────────────
        # Prediction label co-occurs on the same tokens with annotation label(s)
        # from a different hierarchy branch — AND cross-branch co-occurrences
        # dominate over same-branch ones (otherwise it's a COLLISION_SAME_BRANCH).
        for pred_lbl, rec_pred in self._records.items():
            if self._label_prediction_counts.get(pred_lbl, 0) == 0:
                continue
            if rec_pred.resolved is None or rec_pred.tier in ("UNRESOLVED", "NONE"):
                continue
            pred_bk = _branch_key(rec_pred.resolved)
            if not pred_bk:
                continue

            cross_overlap: dict[str, int] = {}
            same_branch_count = 0
            for ann_lbl, rec_ann in self._records.items():
                if self._label_annotation_counts.get(ann_lbl, 0) == 0:
                    continue
                if rec_ann.resolved is None or rec_ann.tier in ("UNRESOLVED", "NONE"):
                    continue
                if self._results_df is not None:
                    mask = (self._results_df["prediction"] == pred_lbl) & (
                        self._results_df["annotation"] == ann_lbl
                    )
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    if _branch_key(rec_ann.resolved) == pred_bk:
                        same_branch_count += count  # same branch — tally but skip
                    else:
                        cross_overlap[ann_lbl] = count

            if not cross_overlap:
                continue

            # Skip if same-branch co-occurrences dominate: the label is mostly
            # correctly branched; the cross-branch tokens are incidental FPs.
            cross_total = sum(cross_overlap.values())
            if same_branch_count >= cross_total:
                continue

            n_pred = self._label_prediction_counts.get(pred_lbl, 0)
            top_overlap = sorted(
                cross_overlap.items(), key=lambda x: x[1], reverse=True
            )[:3]
            options = []
            for ann_lbl, cnt in top_overlap:
                rec_ann = self._records.get(ann_lbl)
                target = rec_ann.resolved if rec_ann and rec_ann.resolved else ann_lbl
                options.append(
                    ResolutionOption(
                        action="map_to_canonical",
                        description=(
                            f"Remap {pred_lbl!r} to {target!r} "
                            f"({cnt} token co-occurrences with {ann_lbl!r})"
                        ),
                        mapper_call={pred_lbl: target},
                    )
                )
            options.append(
                ResolutionOption(
                    action="suppress",
                    description=f"Suppress {pred_lbl!r} from evaluation",
                    mapper_call={pred_lbl: None},
                )
            )
            insight = ""
            if top_overlap and top_overlap[0][1] > 50:
                insight = (
                    f" Systematic misprediction: {pred_lbl!r} predicted on "
                    f"{top_overlap[0][1]} {top_overlap[0][0]!r} tokens."
                )
            self._issues.append(
                MappingIssue(
                    type=IssueType.COLLISION_CROSS_BRANCH,
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{pred_lbl!r} (→ {rec_pred.resolved!r}) co-occurs with "
                        f"annotation(s) on a different hierarchy branch.{insight} "
                        f"Call mapper.map({{{pred_lbl!r}: 'TARGET'}}) to remap or suppress."
                    ),
                    labels=[pred_lbl],
                    annotation_count=0,
                    prediction_count=n_pred,
                    resolution_options=options,
                    overlap_counts=dict(top_overlap),
                )
            )

        # ── PREDICTION_ONLY (WARNING) ────────────────────────────────────────
        # Prediction label's branch is entirely absent from all annotations.
        prediction_only_branches = prediction_branches - annotation_branches
        prediction_only_by_entity: dict[str, list[str]] = {}
        for lbl, rec in self._records.items():
            if self._label_prediction_counts.get(lbl, 0) == 0:
                continue
            if rec.resolved is None or rec.tier in ("UNRESOLVED", "NONE"):
                continue
            if _branch_key(rec.resolved) in prediction_only_branches:
                prediction_only_by_entity.setdefault(rec.resolved, []).append(lbl)

        for entity, raw_labels in sorted(prediction_only_by_entity.items()):
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
                        f"{entity!r} appears in predictions but not in annotations "
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

        # ── DATASET_ONLY (WARNING) ──────────────────────────────────────────
        # Annotation label's branch is entirely absent from all predictions.
        dataset_only_branches = annotation_branches - prediction_branches
        dataset_only_by_entity: dict[str, list[str]] = {}
        for lbl, rec in self._records.items():
            if self._label_annotation_counts.get(lbl, 0) == 0:
                continue
            if rec.resolved is None or rec.tier in ("UNRESOLVED", "NONE"):
                continue
            if _branch_key(rec.resolved) in dataset_only_branches:
                dataset_only_by_entity.setdefault(rec.resolved, []).append(lbl)

        for entity, raw_labels in sorted(dataset_only_by_entity.items()):
            n_ann = sum(self._label_annotation_counts.get(lbl, 0) for lbl in raw_labels)
            self._issues.append(
                MappingIssue(
                    type=IssueType.DATASET_ONLY,
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"{entity!r} has no prediction on its hierarchy branch. "
                        f"{n_ann} annotation tokens -- all will count as false negatives."
                    ),
                    labels=raw_labels or [entity],
                    annotation_count=n_ann,
                    prediction_count=0,
                )
            )

        # ── COLLISION_SAME_BRANCH (INFO) ─────────────────────────────────────
        # Prediction label co-occurs with annotation label(s) on same branch but
        # different depth (e.g. prediction=PERSON depth-2, annotation=NAME depth-3).
        for pred_lbl, rec_pred in self._records.items():
            if self._label_prediction_counts.get(pred_lbl, 0) == 0:
                continue
            if rec_pred.resolved is None or rec_pred.tier in ("UNRESOLVED", "NONE"):
                continue
            pred_branch = h_full.canonical_to_branch.get(rec_pred.resolved, [])
            if len(pred_branch) < 2:
                continue
            pred_branch_key = pred_branch[1]
            pred_depth = len(pred_branch)

            same_branch_overlap: dict[str, int] = {}
            for ann_lbl, rec_ann in self._records.items():
                if self._label_annotation_counts.get(ann_lbl, 0) == 0:
                    continue
                if rec_ann.resolved is None or rec_ann.tier in ("UNRESOLVED", "NONE"):
                    continue
                ann_branch = h_full.canonical_to_branch.get(rec_ann.resolved, [])
                if len(ann_branch) < 2 or ann_branch[1] != pred_branch_key:
                    continue
                if len(ann_branch) == pred_depth:
                    continue  # same depth — not a mismatch
                if self._results_df is not None:
                    mask = (self._results_df["prediction"] == pred_lbl) & (
                        self._results_df["annotation"] == ann_lbl
                    )
                    count = int(mask.sum())
                    if count > 0:
                        same_branch_overlap[ann_lbl] = count

            if not same_branch_overlap:
                continue

            n_pred = self._label_prediction_counts.get(pred_lbl, 0)
            ann_str = ", ".join(
                f"{a!r} ({c})"
                for a, c in sorted(same_branch_overlap.items(), key=lambda x: -x[1])
            )
            self._issues.append(
                MappingIssue(
                    type=IssueType.COLLISION_SAME_BRANCH,
                    severity=IssueSeverity.INFO,
                    message=(
                        f"{pred_lbl!r} (→ {rec_pred.resolved!r}, depth {pred_depth}) "
                        f"co-occurs with same-branch annotation(s) at different "
                        f"depth: {ann_str}. "
                        f"Handled automatically by hierarchical evaluation "
                        f"(branch/detailed projection)."
                    ),
                    labels=[pred_lbl],
                    annotation_count=0,
                    prediction_count=n_pred,
                    overlap_counts=same_branch_overlap,
                )
            )

        # ── Sort: severity order, issue type order, token count desc ────────
        self._issues.sort(
            key=lambda i: (
                _SEVERITY_ORDER[i.severity],
                _ISSUE_TYPE_ORDER.get(i.type, 99),
                -((i.annotation_count or 0) + (i.prediction_count or 0)),
            )
        )

    # -- Issues ---------------------------------------------------------------

    def get_issues(self) -> list[MappingIssue]:
        """Return issues from the last analyze() call, filtered by min_severity.

        Issues with severity below min_severity are excluded. COLLISION_SAME_BRANCH
        (INFO) is only returned when min_severity='INFO'.
        """
        min_order = _SEVERITY_ORDER[self._min_severity]
        return [i for i in self._issues if _SEVERITY_ORDER[i.severity] <= min_order]

    # -- Mutation -------------------------------------------------------------

    def map(self, mappings: dict[str, str | None]) -> CanonicalMapper:
        """Assign resolved entities (or None) to one or more labels.

        Validates all entries atomically before applying any.
        Returns self for chaining.
        """
        valid_canonicals = set(self._hierarchy.all_canonical_entities) | set(
            self._hierarchy.raw_to_canonical.values()
        )

        for _lbl, resolved in mappings.items():
            if resolved is not None and resolved not in valid_canonicals:
                raise ValueError(f"{resolved!r} is not a valid canonical entity.")

        for lbl, resolved in mappings.items():
            if resolved is None:
                logger.info("[NONE]    %s -> None  (suppressed from evaluation)", lbl)
                self._records[lbl] = _Resolution(
                    tier="NONE",
                    resolved=None,
                    score=None,
                )
            else:
                logger.info("[MANUAL]  %s -> %s", lbl, resolved)
                self._records[lbl] = _Resolution(
                    tier="MANUAL",
                    resolved=resolved,
                    score=None,
                )

        if self._results_df is not None:
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

        :param entity: A single raw label to look up (returns its resolved entity).
        :return: dict of all label->resolved mappings, or a single resolved entity.
        """
        if entity is not None:
            rec = self._records.get(entity)
            if rec is not None:
                return rec.resolved
            try:
                return self._hierarchy.canonicalize(entity)
            except EntityNotMappedError:
                raise ValueError(
                    f"Cannot resolve {entity!r}. "
                    f"Use mapper.map({{{entity!r}: 'CANONICAL'}}) to map it."
                ) from None

        return {
            lbl: rec.resolved
            for lbl, rec in self._records.items()
            if rec.tier != "UNRESOLVED"
        }

    def get_mapped_results_dataframe(self) -> MappedResults:
        """Return a MappedResults object with four pre-projected DataFrames.

        Each DataFrame has ``annotation`` and ``prediction`` columns
        (plus all original non-label columns) at the corresponding level:

        - ``.original`` — raw input labels, unmodified.
        - ``.binary``   — any non-O label → ``"PII"``; suppressed/O → ``"O"``.
        - ``.branch``   — depth-2 branch ancestor (e.g. ``NAME`` → ``PERSON``).
        - ``.detailed`` — hierarchy node at native depth (e.g. ``FIRST_NAME`` → ``NAME``).

        :raises RuntimeError: if analyze() has not been called.
        :raises IncompleteMapping: if any UNRESOLVED (ERROR) issues remain.
        """
        if self._results_df is None:
            raise RuntimeError(
                "No DataFrame available. Call analyze(results_df) first."
            )
        blocking = [i for i in self._issues if i.severity == IssueSeverity.ERROR]
        if blocking:
            raise IncompleteMapping(blocking)

        df = self._results_df

        def _resolve(label: str) -> str | None:
            """Return the resolved hierarchy node for a raw label, or None if suppressed."""
            if label == "O":
                return "O"
            rec = self._records.get(label)
            if rec is None:
                return label  # unknown label — pass through
            return rec.resolved  # None for suppressed labels

        def _level(label: str, level: str) -> str:
            resolved = _resolve(label)
            if resolved is None or resolved == "O":
                return "O"
            if level == "binary":
                return to_binary(resolved)
            if level == "branch":
                return to_branch(resolved, _FULL_HIERARCHY)
            # detailed — hierarchy node at native depth
            return resolved

        original = df.copy()

        binary = df.copy()
        binary["annotation"] = df["annotation"].map(lambda x: _level(x, "binary"))
        binary["prediction"] = df["prediction"].map(lambda x: _level(x, "binary"))

        branch = df.copy()
        branch["annotation"] = df["annotation"].map(lambda x: _level(x, "branch"))
        branch["prediction"] = df["prediction"].map(lambda x: _level(x, "branch"))

        detailed = df.copy()
        detailed["annotation"] = df["annotation"].map(lambda x: _level(x, "detailed"))
        detailed["prediction"] = df["prediction"].map(lambda x: _level(x, "detailed"))

        return MappedResults(
            original=original,
            binary=binary,
            branch=branch,
            detailed=detailed,
        )

    def render_html(self) -> None:
        """Render an audit table via MapperRenderer.

        Convenience wrapper. For more control use
        ``MapperRenderer(mapper).render()`` directly.
        """
        _get_renderer_class()(self).render()

    def render_summary(self) -> None:
        """Render a compact per-label summary table (counts + confidence).

        Shows one row per label with: canonical match, projected entity,
        annotation/prediction token counts, and match confidence.
        Convenience wrapper around ``MapperRenderer(mapper).render_summary()``.
        """
        _get_renderer_class()(self).render_summary()

    def __repr__(self) -> str:
        n = len(self._records)
        n_error = sum(1 for i in self._issues if i.severity == IssueSeverity.ERROR)
        return f"CanonicalMapper({n} labels, {n_error} unresolved)"

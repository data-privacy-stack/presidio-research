# PRD: CanonicalMapper Single-Phase Redesign and Hierarchical Multi-Level Evaluation

## Introduction

This PRD covers two related features defined in ADR-002 and ADR-003:

1. **CanonicalMapper redesign (ADR-002):** Replace the existing two-phase "Identify → Project"
   `CanonicalMapper` with a single-phase "Identify-only" mapper. Annotations are never projected
   onto a computed canonical surface — they keep their native hierarchy depth and serve as the
   evaluation contract.

2. **Hierarchical multi-level evaluation (ADR-003):** Run the **existing flat evaluator three
   times** at different granularities (binary/branch/detailed). This PRD introduces
   `MappedResults` — a dataclass emitted by the mapper with pre-projected DataFrames for each
   level — and updates `calculate_hierarchical_scores()` in `BaseEvaluator` to accept it
   directly, eliminating internal column projection.

### Background: What is changing and why

**Old behaviour (being removed):** The existing `CanonicalMapper` operates in two phases:

1. *Identify* — maps each raw label to a node in the entity hierarchy.
2. *Project* — computes a "canonical surface" by majority-voting the depth of annotation labels,
   then **rewrites both the annotation and prediction columns** so that every label sits at that
   majority depth. For example, if the dataset mostly uses `NAME` (depth 3), a model prediction
   of `PERSON` (depth 2) is projected up to `NAME`, and a dataset annotation of `FIRST_NAME`
   (depth 4) is projected down to `NAME`. Evaluation then runs once against these rewritten labels.

This projection is problematic because it modifies the evaluation contract: the scores reflect a
computed surface, not the labels the dataset was actually built with, and depth-granularity
information is silently discarded.

**New behaviour (this PRD):** The mapper is single-phase. After identifying where each label
lives in the hierarchy, it produces `MappedResults` — four DataFrames where both the annotation
and prediction columns are **independently projected** to each evaluation level (binary, branch,
detailed), without either side being rewritten to match the other. Evaluation runs three times:
once per level, always comparing annotation against prediction at that level. No majority-vote
canonical surface is computed. No label is ever changed to match its counterpart.

**Evaluator scope is intentionally lean:** No changes to `SpanEvaluator`, `TokenEvaluator`,
`EvaluationResult`, or `Plotter`. `get_mapped_results_dataframe()` returns a `MappedResults`
frozen dataclass with four DataFrames (`.original`, `.binary`, `.branch`, `.detailed`); each level DataFrame
already has `annotation`/`prediction` columns so it can be passed directly to
`calculate_score_on_df()` or `Plotter` without any column renaming. `calculate_hierarchical_scores()`
accepts the `MappedResults` object and returns scores for all three levels. No descendant-credit
rule, no new evaluator classes.

**Reference docs:**
- [ADR-002: Dataset-Anchored Entity Mapping](../docs/adr/ADR-002-entity-mapping.md)
- [ADR-003: Hierarchical Multi-Level Evaluation](../docs/adr/ADR-003-hierarchical-evaluation.md)

**Supersedes:** [prd-dataset-anchored-mapper.md](prd-dataset-anchored-mapper.md) (written
against an earlier ADR-002 design that still included a projection phase)

---

## Goals

- **Remove the projection phase from `CanonicalMapper`** — the mapper identifies labels in the
  hierarchy; it does not compute a canonical surface or rewrite either annotation or prediction
  labels to match the other. Both sides are preserved at their native hierarchy depth.
- **Simplify issue taxonomy** — five issue types: `UNRESOLVED` (ERROR),
  `COLLISION_CROSS_BRANCH` (WARNING), `PREDICTION_ONLY` (WARNING), `DATASET_ONLY` (WARNING),
  `COLLISION_SAME_BRANCH` (INFO).
- **Let users control issue verbosity** — `min_severity` parameter on `analyze()` controls
  which issues appear in `get_issues()` and `render_html()`.
- **Score at binary/branch/detailed by running the flat evaluator three times** — the mapper
  produces a `MappedResults` object with `.binary`, `.branch`, `.detailed` DataFrames; each is
  directly passable to
  `calculate_score_on_df()` or `Plotter` without column renaming. `calculate_hierarchical_scores()`
  accepts the `MappedResults` object and calls `calculate_score_on_df()` once per level.
- **Keep annotations unmodified** — `.original` preserves the raw input labels; each level
  DataFrame carries projected labels in its own `annotation`/`prediction` columns.

---

## User Stories

---

### US-001: Remove the projection phase from CanonicalMapper

**Description:** As a developer, I want the mapper to be single-phase (Identify only) so that
annotations are never modified and no majority-vote canonical surface is computed.

**Acceptance Criteria:**
- [ ] Remove `_compute_canonical_surface()` and the `_canonical_surface` attribute.
- [ ] Remove the `canonical_surface` property.
- [ ] Remove `canonical_depth` and `eval_entities` from `__init__` (if still present).
- [ ] `analyze()` no longer computes or locks a canonical surface.
- [ ] `get_mapped_results_dataframe()` returns a `MappedResults` frozen dataclass with four
  DataFrame attributes. In each, all non-label columns (`sentence_id`, `token`,
  `start_indices`) are preserved unchanged:
  - `.original` — `annotation` and `prediction` are the **raw input labels**, unmodified.
  - `.binary` — `annotation` and `prediction` resolved to binary: `"PII"` for any non-`O` label,
    `"O"` otherwise.
  - `.branch` — `annotation` and `prediction` resolved to the depth-2 branch ancestor
    (e.g. `FIRST_NAME` → `PERSON`, `STREET` → `LOCATION`). `"O"` passes through.
  - `.detailed` — `annotation` and `prediction` resolved to detailed: the hierarchy node at
    native depth (e.g. `FIRST_NAME` → `NAME`, `PER` → `PERSON`). `"O"` passes through.
    Suppressed labels (`None`) → `"O"`.
- [ ] All four DataFrames have only `annotation` and `prediction` as label columns — no
  `_binary`/`_branch`/`_detailed` suffixes. Each is directly passable to
  `calculate_score_on_df()` and `Plotter` without renaming.
- [ ] `MappedResults` is a frozen dataclass (or `NamedTuple`); fields are `.original`,
  `.binary`, `.branch`, `.detailed`.
- [ ] `get_mapped_results_dataframe()` still raises `IncompleteMapping` when any `UNRESOLVED`
  labels remain.
- [ ] All existing tests pass (update tests that reference `canonical_surface`, depth
  projection behavior, or `original_annotation`/`original_prediction` column names).
- [ ] Typecheck/lint passes.

---

### US-002: Update the issue type taxonomy

**Description:** As a user, I want a cleaner set of issue types that match the new single-phase
design so the audit table isn't cluttered with obsolete concepts.

**Acceptance Criteria:**
- [ ] Remove `COLLISION_TRIVIAL` and `COLLISION_AMBIGUOUS` from `IssueType` enum and all code
  paths that generate them.
- [ ] Add `COLLISION_SAME_BRANCH` to `IssueType` enum (severity: `INFO`). This fires when a
  prediction label and annotation label co-occur on the same tokens and both resolve to the
  **same hierarchy branch** at different depths (e.g., model predicts `PERSON`, dataset
  annotates `NAME`). It is purely informational — the hierarchical evaluator handles it
  automatically via column projection at each level.
- [ ] `COLLISION_CROSS_BRANCH` remains at severity `WARNING`. Fires when a prediction and
  annotation co-occur on the same tokens but resolve to **different branches** (potential
  vocabulary mismatch).
- [ ] `PREDICTION_ONLY` remains at severity `WARNING`. Fires when a prediction entity is in
  the hierarchy but never annotated by the dataset.
- [ ] `DATASET_ONLY` remains at severity `WARNING`. Fires when an annotation entity has no
  prediction on the same branch.
- [ ] `UNRESOLVED` remains at severity `ERROR` (only blocking issue type).
- [ ] `COUNTRY_FALLBACK` is **not** an issue type — logged internally at INFO level only, never
  surfaced in `get_issues()` or `render_html()` regardless of `min_severity`.
- [ ] `_ISSUE_TYPE_ORDER` dict in `mapper.py` updated: remove `COLLISION_TRIVIAL` and
  `COLLISION_AMBIGUOUS` entries; add `COLLISION_SAME_BRANCH`.
- [ ] Issue ordering: ERROR before WARNING before INFO; within same severity, descending token
  count.
- [ ] Typecheck/lint passes.

---

### US-003: Add `min_severity` parameter to `analyze()`

**Description:** As a user, I want to control which issues are surfaced so I can focus on
blocking issues only, or drill down to see all same-branch depth-mismatch information.

**Acceptance Criteria:**
- [ ] `analyze(results_df, min_severity="WARNING")` is the new signature (default: WARNING and
  above are shown).
- [ ] `min_severity="ERROR"` — `get_issues()` and `render_html()` show only `UNRESOLVED`.
- [ ] `min_severity="WARNING"` — show `UNRESOLVED` + `COLLISION_CROSS_BRANCH` +
  `PREDICTION_ONLY` + `DATASET_ONLY`.
- [ ] `min_severity="INFO"` — show all of the above plus `COLLISION_SAME_BRANCH`.
- [ ] `min_severity` accepts both string (`"WARNING"`) and `IssueSeverity` enum values.
- [ ] Raises `ValueError` for unrecognised severity strings.
- [ ] `COLLISION_SAME_BRANCH` issues are **never** shown when `min_severity` is `"WARNING"` or
  `"ERROR"`.
- [ ] Typecheck/lint passes.

---

### US-004: Add `get_mapping()` method

**Description:** As a pipeline operator, I want a serialisable view of the current mapping
state so I can version-control or share it across runs.

**Acceptance Criteria:**
- [ ] `mapper.get_mapping()` returns `dict[str, str | None]` where keys are raw labels and
  values are the resolved hierarchy entity (or `None` for suppressed labels).
- [ ] Only resolved labels are included; `UNRESOLVED` labels are excluded.
- [ ] The returned dict is a plain copy — mutating it does not affect the mapper.
- [ ] Typecheck/lint passes.

---

### US-005: Update the HTML audit table for the new issue taxonomy

**Description:** As a user, I want the HTML audit table to accurately reflect the new issue
types and severities so I understand what action (if any) to take.

**Acceptance Criteria:**
- [ ] Remove all references to `COLLISION_TRIVIAL`, `COLLISION_AMBIGUOUS`, and "eval surface"
  / "canonical surface" from the rendered HTML.
- [ ] Add a badge colour for `COLLISION_SAME_BRANCH` (e.g. blue).
- [ ] Summary bar shows counts per new issue type: UNRESOLVED (red), COLLISION_CROSS_BRANCH
  (orange), PREDICTION_ONLY (grey), DATASET_ONLY (amber), COLLISION_SAME_BRANCH (blue).
- [ ] Detail table respects `min_severity`: `COLLISION_SAME_BRANCH` rows only appear when
  `min_severity="INFO"`.
- [ ] For `COLLISION_SAME_BRANCH` rows, display: raw label, resolved hierarchy entity,
  co-occurring annotation label, token count, and the note: "Same-branch depth mismatch —
  handled automatically by hierarchical evaluation (branch/detailed projection)."
- [ ] For `COLLISION_CROSS_BRANCH` rows, retain overlap counts with annotation labels.
- [ ] `render_html()` still works before `analyze()` is called (shows empty state).
- [ ] Typecheck/lint passes.

---

### US-006: Update `calculate_hierarchical_scores()` to accept `MappedResults`

**Description:** As a user, I want a single call to score all three levels, and be able to
pass any individual level DataFrame to the flat evaluator or `Plotter` directly — no column
renaming needed anywhere in the flow.

**Background:** `MappedResults` makes per-level scoring trivial — each attribute is already a
valid input to `calculate_score_on_df()` and `Plotter`:

```python
results = mapper.get_mapped_results_dataframe()

# All three levels at once
scores = evaluator.calculate_hierarchical_scores(results)
# scores["binary"], scores["branch"], scores["detailed"] — each an EvaluationResult

# Or pick a single level for error analysis
result_detailed = evaluator.calculate_score_on_df(results.detailed)
Plotter(result_detailed, model_name="MyModel-detailed").plot_scores()
```

`calculate_hierarchical_scores()` is updated to call `calculate_score_on_df()` on the three
level DataFrames directly, eliminating the internal `_to_binary`/`_to_branch` projection.

**Acceptance Criteria:**
- [ ] `calculate_hierarchical_scores(results: MappedResults)` calls `calculate_score_on_df()`
  on `results.binary`, `results.branch`, and `results.detailed` directly — no internal
  column projection.
- [ ] The method returns `dict[str, EvaluationResult]` with keys `"binary"`, `"branch"`,
  `"detailed"`.
- [ ] Internal `_to_binary`/`_to_branch` projection helpers are removed from this method
  (projection is now done once by the mapper).
- [ ] A unit test verifies that for a dataset with mixed-depth annotations (e.g. both `PERSON`
  depth-2 and `NAME` depth-3), `calculate_hierarchical_scores()` produces correct branch-level
  scores: `NAME` → `PERSON` at branch level and `PERSON` prediction stays `PERSON` → TP.
- [ ] A unit test verifies that calling `calculate_score_on_df(results.branch)` directly
  produces the same result as `scores["branch"]` from `calculate_hierarchical_scores()`.
- [ ] No changes to `SpanEvaluator`, `TokenEvaluator`, `EvaluationResult`, or `Plotter`.
- [ ] Typecheck/lint passes.

---

### US-007: Update Notebook 4 (Evaluate Presidio Analyzer)

**Description:** As a user following the evaluation tutorial, I want Notebook 4 to demonstrate
the new single-phase mapping workflow, three-level scoring, and level-specific error analysis
using the existing Plotter.

**Acceptance Criteria:**
- [ ] Replace any `CanonicalMapper(canonical_depth=N)` with `CanonicalMapper()`.
- [ ] `analyze(results_df)` is called without any depth or hierarchy parameter.
- [ ] Notebook includes explanatory markdown covering the five issue types and their severities.
- [ ] Notebook shows resolving `UNRESOLVED` labels with `map()` before calling
  `get_mapped_results_dataframe()`.
- [ ] Notebook demonstrates calling `calculate_hierarchical_scores(results)` and printing
  binary, branch, detailed `EvaluationResult` objects.
- [ ] Notebook includes a cell showing per-level `Plotter` use: call
  `calculate_score_on_df(results.<level>)` and pass the result to `Plotter`. Example shown
  for detailed (native depth) and branch (entity-type level).
- [ ] Remove any references to `canonical_surface`, `canonical_depth`, `COLLISION_TRIVIAL`,
  `COLLISION_AMBIGUOUS`, or the projection phase.
- [ ] Notebook runs end-to-end without error.

---

### US-008: Update Notebook 6 (Interactive Entity Mapping)

**Description:** As a user learning entity mapping in depth, I want Notebook 6 to cover the
full single-phase workflow and updated issue types.

**Acceptance Criteria:**
- [ ] Rewrite intro cells to explain the single-phase model: Identify (5 tiers) only — no
  projection.
- [ ] Add a section covering all five issue types with concrete examples and when they occur.
- [ ] Explain `min_severity`: show `analyze(results_df, min_severity="INFO")` to surface
  `COLLISION_SAME_BRANCH` for awareness.
- [ ] Show `resolve_interactively()` for guided resolution of `UNRESOLVED` labels (no changes
  to `resolve_interactively()` — it already only prompts for `UNRESOLVED`).
- [ ] Show `map()` for programmatic batch resolution.
- [ ] Show calling `calculate_hierarchical_scores(results)` and explain what binary/branch/
  detailed mean.
- [ ] Show per-level `Plotter` use: call `calculate_score_on_df(results.<level>)` and pass
  the result to `Plotter` (same approach as Notebook 4).
- [ ] Remove all references to `canonical_depth`, `eval_entities`, `canonical_surface`,
  `COLLISION_TRIVIAL`, `COLLISION_AMBIGUOUS`, or the projection phase.
- [ ] Notebook runs end-to-end without error.

---

### US-009: Update Notebook 5 (Evaluate Custom Presidio Analyzer)

**Description:** As a user evaluating a custom Presidio Analyzer, I want Notebook 5 to use
the same single-phase mapping workflow and three-level evaluation as Notebook 4.

**Acceptance Criteria:**
- [ ] Replace any `CanonicalMapper(canonical_depth=N)` with `CanonicalMapper()`.
- [ ] `analyze(results_df)` is called without any depth or hierarchy parameter.
- [ ] Notebook shows resolving `UNRESOLVED` labels with `map()` before calling
  `get_mapped_results_dataframe()`.
- [ ] Notebook demonstrates calling `calculate_hierarchical_scores(results)` and printing
  binary, branch, detailed `EvaluationResult` objects.
- [ ] Notebook includes a cell showing per-level `Plotter` use: call
  `calculate_score_on_df(results.<level>)` and pass the result to `Plotter`. Example shown
  for detailed (native depth) and branch (entity-type level).
- [ ] Remove any references to `canonical_surface`, `canonical_depth`, `COLLISION_TRIVIAL`,
  `COLLISION_AMBIGUOUS`, or the projection phase.
- [ ] Notebook runs end-to-end without error.

---

## Functional Requirements

- **FR-1:** `CanonicalMapper.analyze(results_df, min_severity="WARNING")` — identification-only.
  `min_severity` controls which issue types appear in `get_issues()` and `render_html()`.
- **FR-2:** Issue types: `UNRESOLVED` (ERROR), `COLLISION_CROSS_BRANCH` (WARNING),
  `PREDICTION_ONLY` (WARNING), `DATASET_ONLY` (WARNING), `COLLISION_SAME_BRANCH` (INFO). No
  others.
- **FR-3:** `COLLISION_SAME_BRANCH` fires when a prediction label and annotation label co-occur
  on the same tokens and both resolve to the same hierarchy branch at different depths.
- **FR-4:** `get_mapped_results_dataframe()` raises `IncompleteMapping` if UNRESOLVED issues
  remain; otherwise returns a `MappedResults` frozen dataclass with four DataFrames. In each,
  all non-label columns (`sentence_id`, `token`, `start_indices`) are preserved:
  - `.original` — `annotation`/`prediction` = raw input labels, unchanged.
  - `.binary` — `annotation`/`prediction` resolved to `"PII"` / `"O"`. Suppressed labels → `"O"`.
  - `.branch` — `annotation`/`prediction` resolved to depth-2 branch ancestor (`"PERSON"`,
    `"LOCATION"`, …) or `"O"`. Suppressed labels → `"O"`.
  - `.detailed` — `annotation`/`prediction` resolved to hierarchy node at native depth
    (`"NAME"`, `"PERSON"`, `"ADDRESS"`, …) or `"O"`. Suppressed labels → `"O"`.
  Each DataFrame is directly passable to `calculate_score_on_df()` and `Plotter`.
- **FR-5:** `get_mapping()` returns `dict[str, str | None]` of resolved labels (UNRESOLVED
  excluded).
- **FR-6:** `calculate_hierarchical_scores(results: MappedResults) → dict[str, EvaluationResult]`
  calls `calculate_score_on_df()` on `results.binary`, `results.branch`, `results.detailed`
  directly, returning keys `"binary"`, `"branch"`, `"detailed"`. Internal `_to_binary`/
  `_to_branch` projection helpers are removed from this method. No changes to
  `SpanEvaluator`, `TokenEvaluator`, `EvaluationResult`, or `Plotter`.
- **FR-7:** `COUNTRY_FALLBACK` resolution is logged at INFO level internally and never surfaced
  in `get_issues()` or `render_html()`.

---

## Non-Goals (Out of Scope)

- No semantic similarity (embedding-based) label matching.
- No descendant-credit rule or partial-credit F1 in the evaluator.
- No LCA projection at evaluation time.
- No new `Plotter` methods, no changes to error analysis — existing `plot_scores()`,
  `plot_most_common_tokens()`, and `plot_confusion_matrix()` are used as-is. Level-specific
  analysis is done by passing `results.<level>` directly to `calculate_score_on_df()` and `Plotter`.
- No changes to `SpanEvaluator`, `TokenEvaluator`, `EvaluationResult`, or `Plotter`.
- No `HierarchicalEvaluationResult` type alias.
- No `get_hierarchical_mapped_dataframe()` method.
- No changes to `EntityHierarchy` structure or alias definitions.

---

## Technical Considerations

- **Same-branch detection for `COLLISION_SAME_BRANCH`:** Two labels are on the same branch if
  their `canonical_to_branch` paths share at least the second element (depth-2 ancestor). Use
  `EntityHierarchy.canonical_to_branch`.
- **Projection removal side-effects:** Code paths that currently reference `rec.projected`
  should be updated to use `rec.resolved` (see the "canonical vocabulary" note below for
  the rename of `rec.canonical` → `rec.resolved`). Code that only existed to support
  projection logic should be deleted.
- **`MappedResults` construction:** `get_mapped_results_dataframe()` builds the object by
  applying label-resolution four times (original = passthrough, binary, branch, detailed).
  Define `MappedResults` as a frozen dataclass in
  `presidio_evaluator/entity_mapping/data_objects.py` (alongside `IssueType`, `IssueSeverity`).
  The `_to_binary` and `_to_branch` helpers (currently `_to_l0`/`_to_l1` in
  `base_evaluator.py`) should be moved to a shared utility module (e.g.
  `presidio_evaluator/entity_mapping/level_helpers.py`) so `mapper.py` can import them without
  a circular dependency. `.detailed` resolution uses the mapper's own `_records` (the hierarchy
  node each raw label resolved to).
- **`calculate_hierarchical_scores()` simplified:** Replace internal projection with direct
  `calculate_score_on_df()` calls on `results.binary`, `results.branch`, `results.detailed`.
  Remove the `_to_binary`/`_to_branch` calls from this method.
- **`results.original` replaces the old flat `annotation`/`prediction` columns:** Code that
  previously relied on `annotation`/`prediction` in the mapped DataFrame (e.g. tests,
  notebooks) must use `results.original` (for raw labels) or `results.detailed` (for
  native-depth evaluation).
- **"canonical" vocabulary after this PRD:** The word "canonical" will no longer appear in the
  public API. Internally, `rec.canonical` on the `_Resolution` dataclass holds the hierarchy
  node a raw label resolved to — this is what `.detailed` exposes publicly. To avoid
  implementer confusion, rename `rec.canonical` to `rec.resolved` (or `rec.hierarchy_node`)
  as part of this work. `canonical_to_branch` on `EntityHierarchy` is an existing method name
  and is **not** renamed by this PRD.

---

## Success Metrics

- `COLLISION_TRIVIAL` and `COLLISION_AMBIGUOUS` no longer appear anywhere in the codebase.
- `get_mapped_results_dataframe()` returns a `MappedResults` object whose `.original`
  DataFrame preserves raw labels and whose `.binary`/`.branch`/`.detailed` DataFrames
  contain correctly projected labels — no column renaming needed at any call site.
- A unit test confirms that `calculate_hierarchical_scores()` on the new mapper output
  produces correct branch/detailed scores for a mixed-depth dataset.
- All existing tests pass without modification (or with minimal updates where old behavior was
  incorrect per the new design).
- Notebook 4 runs end-to-end without error with the new API.
- Notebook 5 runs end-to-end without error with the new API.

---

## Open Questions (Resolved)

- **`COLLISION_SAME_BRANCH` threshold:** Surface **all** same-branch depth co-occurrences,
  with no token-count threshold. Issues are sorted by descending token count so low-count
  noise appears at the bottom of the audit table.
- **Notebook 5 scope:** Notebook 5 receives the same API updates as Notebook 4 (see US-009).

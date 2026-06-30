# PRD: Dataset-Anchored Entity Mapping via CanonicalMapper

## Introduction

Users of Presidio Evaluator bring their own entity label vocabularies — from custom datasets,
fine-tuned models, or third-party tools. Before evaluation can happen, every user-defined label
must be resolved to a common vocabulary so that dataset annotations and model predictions can be
compared on equal footing.

The core use case is **comparing multiple models against the same dataset**. The dataset defines
the evaluation contract — its entity vocabulary is the ground truth. Models are the variable;
the dataset is the constant.

This PRD defines a complete rewrite of `CanonicalMapper` as a single-phase (Identify) mapper.
Annotations are never projected onto a computed surface — they keep their native depth and serve
as the evaluation contract. Predictions are compared against annotations at every depth level via
the descendant-credit rule: a prediction equal to or more specific than the annotation is TP;
a coarser prediction is FP+FN at the annotation's depth level. All same-branch depth mismatches
are handled automatically by the hierarchical evaluator with no user intervention required. Only
labels entirely absent from the hierarchy require user action.

The workflow is a pure Python API (no UI). Interactive prompting is done in the terminal (usable
in notebooks or CLI), and can be bypassed entirely by calling `map()` with pre-built assignments.

**Reference:** [ADR-002: Dataset-Anchored Entity Mapping](../docs/adr/ADR-002-entity-mapping.md)\
**See also:** [ADR-003: Hierarchical Multi-Level Evaluation](../docs/adr/ADR-003-hierarchical-evaluation.md) — layers L0/L1/L2 scoring on top of the mapped output from this pipeline.

**Replaces:** [prd-entity-mapping-workflow.md](prd-entity-mapping-workflow.md) (obsolete)

---

## Goals

- **Dataset-anchored mapping** — annotations define the evaluation contract at their native depth. They are never projected or modified.
- **No depth parameter, no majority vote** — the hierarchy position of each label is determined by the hierarchy alone, not by a vote over the dataset.
- **Finer predictions get credit** — a prediction that is a descendant of the annotation on the same branch scores TP (descendant-credit rule). Coarser predictions are FP+FN at the annotation's depth.
- **Force explicit decisions on unresolvable labels** — only `UNRESOLVED` (ERROR) labels block DataFrame extraction. All other issues are informational.
- **Frequency-based issue triage** — surface the most impactful problems first (by token count).
- **Preserve originals** — output DataFrame includes `original_annotation` and `original_prediction` columns alongside the mapped columns.
- **Log every decision** — every resolution is logged so users understand what happened.
- **Replace previous mapping entirely** — remove old issue types (`DEPTH_MISMATCH`, `SUPPRESSED_WITH_ANNOTATIONS`, `HIERARCHY_DEPTH_CHANGED`, `COLLISION_TRIVIAL`, `COLLISION_AMBIGUOUS`), remove `sentence-transformers` dependency, remove `canonical_depth`/`eval_entities`/`labels` constructor parameters.

---

## Process Overview

Entity mapping resolves the vocabulary mismatch between dataset annotations and model
predictions. The process has two phases that run inside `analyze()`, followed by a resolution
phase driven by the user.

### Phase 1 — Identify

Every raw label (from both annotations and predictions) is resolved to a canonical entity
in the hierarchy using a tiered resolution strategy, attempted in order:

| Tier | Description | Example |
|------|-------------|---------|
| **EXACT** | Label matches a hierarchy alias exactly (case-insensitive, delimiter-normalized) | `email_address` → `EMAIL_ADDRESS` |
| **COUNTRY** | Recognized country prefix; remainder resolves to a known document type | `GERMANY_PASSPORT_NUMBER` → `PASSPORT` |
| **COUNTRY_FALLBACK** | Recognized country prefix; remainder is unknown → falls back to `NATIONAL_ID` | `NIGERIA_UNICORN_CARD` → `NATIONAL_ID` (INFO log) |
| **FUZZY** | Fuzzy string match above threshold (default 0.80) | `EMAILADRES` → `EMAIL_ADDRESS` (87%) |
| **UNRESOLVED** | No tier matched | Flagged as ERROR |

Before any tier is attempted, BIO/BIOES/BILOU prefixes and suffixes are stripped.
The special outside token `O` is mapped to `None`.

### Phase 2 — Hierarchical evaluation (in SpanEvaluator, not the mapper)

The mapper's job ends after identification. Depth-level comparison is handled by
`SpanEvaluator.calculate_hierarchical_scores()` using the descendant-credit rule:

| Situation | Scoring |
|---|---|
| Prediction = annotation or any descendant on the same branch | **TP** at the annotation's depth level |
| Prediction is an ancestor of annotation (too coarse) | **FP+FN** at annotation's depth; TP at levels above the prediction |
| Prediction is on a different branch | **FP+FN** at all levels — COLLISION_CROSS_BRANCH in audit table |
| Prediction = `O`, annotation ≠ `O` | **FN** |
| Prediction ≠ `O`, annotation = `O` | **FP** |

The mapper does not project annotations. It only provides the hierarchy so the evaluator can
walk ancestors for level-by-level comparison.

### Issue Types

Four issue types. Only `UNRESOLVED` (ERROR) blocks DataFrame extraction; all others are
informational and never block:

| Issue Type | Severity | Blocks? | What It Means | How to Resolve |
|------------|----------|---------|---------------|-----------------|
| `UNRESOLVED` | ERROR | Yes | A raw label could not be matched to any canonical entity in the hierarchy. | `map({"LABEL": "TARGET"})` to assign, or `map({"LABEL": None})` to suppress. |
| `COLLISION_CROSS_BRANCH` | INFO | No | A prediction entity sits on a different branch from the annotation. Scored as FP+FN. May indicate a systematic misprediction. | No action needed; surfaced in audit table with overlap counts. Optional: remap or suppress via `map()`. |
| `PREDICTION_ONLY` | INFO | No | A prediction entity exists in the hierarchy but the dataset never annotates it — all occurrences are FP. | No action needed. Optional: `map({"X": None})` suppress, `map({"X": "Y"})` remap. |
| `DATASET_ONLY` | INFO | No | An annotation entity has no prediction on the same branch. False negatives will be counted — model coverage gap. | `map({"ENTITY": None})` to suppress if desired. |

### Resolution Phase

After `analyze()`, the user inspects and resolves any blocking issues:

1. **Inspect** — `render_html()` shows a summary bar and detail table sorted by severity then
   token count. `get_issues()` returns the same data programmatically.
2. **Resolve** — `map({"LABEL": "TARGET"})` for programmatic batch resolution of UNRESOLVED
   labels, `resolve_interactively()` for guided prompts, or a combination of both. INFO-level
   issues can optionally be overridden via `map()` but never require action.
3. **Extract** — `get_mapped_results_dataframe()` succeeds only when all `UNRESOLVED` (ERROR)
   issues are resolved. Output includes `original_annotation` and `original_prediction` columns.

For multi-model comparison, the eval surface is **locked** after the first `analyze()` call.
Subsequent models are analyzed against the same entity vocabulary.

---

## User Stories

### US-001: Identify all labels in the hierarchy
**Description:** As a user, I want all labels to be automatically identified in the hierarchy
so I only need to handle the ones that truly can't be resolved.

**Acceptance Criteria:**
- [ ] After `analyze(results_df)`, every annotation and prediction label is located in `EntityHierarchy` using the tiered resolution strategy.
- [ ] Labels that resolve via exact alias-map lookup are identified (tier: EXACT).
- [ ] Labels with a recognized country prefix whose remainder resolves to a known document type are identified (tier: COUNTRY).
- [ ] Labels with a recognized country prefix whose remainder is unknown resolve to `NATIONAL_ID` with an `INFO`-level log (tier: COUNTRY_FALLBACK).
- [ ] Labels that resolve via fuzzy match (≥ `fuzzy_threshold`, default 0.80) are identified (tier: FUZZY), with the resolved node and score logged.
- [ ] Labels that fail all identification tiers are flagged as `UNRESOLVED` (ERROR severity).
- [ ] BIO/BIOES/BILOU prefixes and suffixes are stripped before any tier is attempted.
- [ ] The special outside token `O` is automatically mapped to `None`.
- [ ] There is no `canonical_depth`, `eval_entities`, or majority-vote step.

---

### US-002: Identify labels in the hierarchy
**Description:** As a user, I want all labels to be automatically identified in the hierarchy
so I only need to handle the ones that truly can't be resolved.

**Acceptance Criteria:**
- [ ] Labels that resolve via exact alias-map lookup are identified without user interaction (tier: EXACT).
- [ ] Labels with a recognized country prefix whose remainder resolves to a known document type are identified (tier: COUNTRY).
- [ ] Labels with a recognized country prefix whose remainder is unknown resolve to `NATIONAL_ID` with an `INFO`-level log (tier: COUNTRY_FALLBACK). This is not a separate issue type.
- [ ] Labels that resolve via fuzzy match (≥ `fuzzy_threshold`, default 0.80) are identified (tier: FUZZY), with the resolved node and score logged.
- [ ] Labels that fail all identification tiers are flagged as `UNRESOLVED` (ERROR severity).
- [ ] Before any tier is attempted, BIO/BIOES/BILOU prefixes (`B-`, `I-`, etc.) and suffixes (`-B`, `-I`, etc.) are stripped. The original label remains the dict key.
- [ ] The special outside token `O` is automatically mapped to `None`.

---

### US-003: Hierarchical scoring with descendant-credit rule
**Description:** As a user, I want predictions that are more specific than the annotation to
be credited as correct, not penalised for over-specificity.

**Acceptance Criteria:**
- [ ] `calculate_hierarchical_scores(mapped_df)` produces L0, L1, L2 scores.
- [ ] At each level, a prediction equal to the annotation's ancestor at that level OR any descendant of it scores **TP**.
- [ ] A prediction that is an ancestor of the annotation at its level (too coarse) scores **FP+FN**.
- [ ] A prediction on a different branch scores **FP+FN** at all levels.
- [ ] Annotations are never modified or projected — they stay at their native depth.
- [ ] A dataset annotating both `PERSON` (depth 2) and `TITLE` (depth 3) on the same branch requires no user intervention; each token is scored at its own depth.

---

### US-004: Frequency-based issue triage
**Description:** As a user with a large label vocabulary, I want the most impactful issues
surfaced first so I fix the biggest problems quickly.

**Acceptance Criteria:**
- [ ] Each issue includes the number of affected tokens from the results DataFrame.
- [ ] `get_issues()` returns issues sorted by severity (ERROR > WARNING > INFO), then by token count descending.
- [ ] The audit table (`render_html()`) reflects the same sort order.

---

### US-005: DATASET_ONLY entity detection
**Description:** As a user, I want to know which eval-surface entities have no corresponding
prediction mapping so I understand my model's coverage gaps.

**Acceptance Criteria:**
- [ ] After projection, entities in the eval surface with no prediction mapping are flagged as `DATASET_ONLY` (INFO).
- [ ] Default: keep in eval surface — false negatives are counted (shows model gaps).
- [ ] The user can suppress via `mapper.map({"ENTITY": None})`.
- [ ] `DATASET_ONLY` does not block `get_mapped_results_dataframe()` (INFO severity).
- [ ] The issue includes the annotation token count.

---

### US-006: PREDICTION_ONLY entity detection
**Description:** As a user, I want to know about prediction entities with no eval-surface
counterpart so I can understand my model's prediction coverage.

**Acceptance Criteria:**
- [ ] After projection, entities in predictions with no eval-surface counterpart are flagged as `PREDICTION_ONLY` (INFO).
- [ ] `PREDICTION_ONLY` does **not** block `get_mapped_results_dataframe()` — every occurrence counts as FP by default.
- [ ] Optional resolution via `map()`:
  - `map({"EMAIL": None})` — suppress from evaluation (occurrences no longer counted).
  - `map({"EMAIL": "CONTACT"})` — remap to an eval-surface entity.
  - `map({"EMAIL": "EMAIL"})` — explicit keep-as-FP (documents intent; same as default behavior).
- [ ] The issue includes the prediction token count.

---

### US-007: Cross-branch overlap counts in audit table
**Description:** As a user, I want to see which annotation entities my model's cross-branch
predictions overlap with so I can diagnose systematic mispredictions.

**Acceptance Criteria:**
- [ ] For `COLLISION_CROSS_BRANCH` issues, the mapper counts token co-occurrence with each annotation label in the dataset (how many tokens were predicted as this entity while the dataset annotated something else).
- [ ] Overlap counts are stored in `issue.overlap_counts` and sorted by count descending.
- [ ] Zero-count entries are filtered from the display.
- [ ] Counts are displayed in `render_html()` and returned in `get_issues()` as informational context (not a resolution prompt).

---

### US-008: Programmatic (batch) mapping
**Description:** As a pipeline operator, I want to supply all mappings programmatically so
the workflow runs without interactive prompts.

**Acceptance Criteria:**
- [ ] `mapper.map({"LABEL": "TARGET", "OTHER": None})` assigns mappings without prompting.
- [ ] `map()` validates all values atomically before applying any: raises `ValueError` on invalid values.
- [ ] `map()` can override an already-resolved label (corrects auto-fix decisions).
- [ ] `map()` returns `self` for chaining: `mapper.map({...}).map({...})`.
- [ ] Supports three resolution modes: remap (`{"X": "Y"}`), suppress (`{"X": None}`), keep-as-FP (`{"X": "X"}`).
- [ ] After applying, the mapper re-evaluates which issues are resolved (removes resolved issues from the list).

---

### US-009: Interactive mapping for unresolved issues
**Description:** As a user, I want guided prompts when issues can't be auto-resolved so I
can make decisions quickly.

**Acceptance Criteria:**
- [ ] `resolve_interactively(prompt_fn=input)` prompts only for `UNRESOLVED` (ERROR) issues. All INFO issues are skipped.
- [ ] For each issue, shows: issue type, affected token count, and a resolution prompt.
- [ ] Prompt accepts: a free-text canonical entity name or `NONE` to suppress.
- [ ] Re-prompts on invalid input until a valid choice is made.
- [ ] `resolve_interactively()` is a no-op when there are no UNRESOLVED issues.
- [ ] `prompt_fn` parameter allows injection for testing (default: `input`).
- [ ] Returns `self` for chaining.

---

### US-010: Logging of resolution decisions
**Description:** As a user, I want every mapping decision logged so I can audit what happened.

**Acceptance Criteria:**
- [ ] Every identified label emits a log at `INFO`. Examples:
  - `[EXACT]   EMAIL_ADDRESS → EMAIL_ADDRESS`
  - `[FUZZY 87%] EMAILADRES → EMAIL_ADDRESS`
  - `[COUNTRY] GERMAN_PASSPORT_NUMBER → PASSPORT`
  - `[COUNTRY-FALLBACK] NIGERIA_UNICORN_CARD → NATIONAL_ID  ⚠ document type not recognized`
- [ ] Every auto-fix emits at `INFO`:
  - `[AUTO-FIX] FIRST_NAME → NAME  (descendant projected to eval-surface entity)`
  - `[AUTO-FIX] PERSON → NAME  (ancestor projected to sole eval-surface descendant)`
- [ ] Manual overrides logged: `[MANUAL] MY_LABEL → PERSON`
- [ ] Suppressions logged: `[NONE] EXTRA_LABEL → None  (suppressed from evaluation)`
- [ ] Unresolvable labels emit at `WARNING`: `[UNRESOLVED] {label}  — no automatic match found`
- [ ] After analysis, a summary `INFO` line reports: auto-discovered depth, eval-surface entities, resolution counts, auto-fixes applied, issues requiring attention.
- [ ] Logger name is `presidio_evaluator.entity_mapping`.

---

### US-011: HTML audit table
**Description:** As a user, I want a visual audit table showing all labels, their status,
and issues sorted by impact.

**Acceptance Criteria:**
- [ ] `render_html()` shows two sections:
  1. **Summary bar** — auto-discovered depth, counts per issue type, total auto-fixes applied.
  2. **Detail table** — one row per raw label, sorted by severity then token count.
- [ ] Badge colours: exact match = green, COLLISION_TRIVIAL = blue, UNRESOLVED = red, PREDICTION_ONLY = grey, DATASET_ONLY = amber, COLLISION_CROSS_BRANCH = orange.
- [ ] For cross-branch overlaps with high token counts, surfaces systematic misprediction insights (e.g., "LOCATION predicted on 200 PERSON tokens").
- [ ] Uses `IPython.display.HTML` in Jupyter; falls back to plain text in non-Jupyter environments.
- [ ] Never raises an exception.
- [ ] Fully self-contained HTML (inline styles, no external CSS/JS).
- [ ] Can be called at any point — before, during, or after resolution — reflecting current state.

---

### US-012: Multi-model comparison against same dataset
**Description:** As a user, I want to evaluate multiple models against the same dataset using
the same evaluation surface so their scores are directly comparable.

**Acceptance Criteria:**
- [ ] The eval surface is locked after the first `analyze()` call.
- [ ] Subsequent `analyze()` calls for different models reuse the locked surface.
- [ ] Issues are per-model: cleared and recomputed on each `analyze()` call.
- [ ] Each `get_mapped_results_dataframe()` call returns a DataFrame mapped to the same entity vocabulary.
- [ ] To start fresh with a different dataset, the user creates a new `CanonicalMapper` instance.

---

### US-013: Blocking on unresolved issues
**Description:** As a user, I want the mapper to prevent me from getting incorrect results
by blocking when important issues haven't been addressed.

**Acceptance Criteria:**
- [ ] `get_mapped_results_dataframe()` raises `IncompleteMapping` if any `UNRESOLVED` (ERROR) issues remain.
- [ ] All INFO issues (COLLISION_TRIVIAL, COLLISION_CROSS_BRANCH, PREDICTION_ONLY, DATASET_ONLY) never block.
- [ ] The exception message lists the unresolved issues.
- [ ] When no blocking issues remain, returns a DataFrame with:
  - `annotation` and `prediction` columns rewritten to eval-surface entities (suppressed labels → `"O"`).
  - `original_annotation` and `original_prediction` columns preserving the raw labels.
  - All other columns (`sentence_id`, `token`, `start_indices`) unchanged.

---

### US-014: Systematic misprediction insights
**Description:** As a user, I want to understand when my model systematically predicts the
wrong entity type so I can diagnose model weaknesses.

**Acceptance Criteria:**
- [ ] Cross-branch token overlaps with high token counts are surfaced in `render_html()` as insights.
- [ ] Example: "LOCATION predicted on 200 PERSON tokens" when the model systematically confuses branches.
- [ ] This is not a separate issue type — it is additional context on `COLLISION_CROSS_BRANCH` issues.

---

### US-015: Update Notebook 4 (Evaluate Presidio Analyzer)
**Description:** As a user following the evaluation tutorial, I want Notebook 4 to demonstrate
the new two-phase mapping workflow so I can learn the current API.

**Acceptance Criteria:**
- [ ] Replace `CanonicalMapper(canonical_depth=3)` with `CanonicalMapper()` (no constructor params).
- [ ] `analyze(results_df)` called without any `hierarchy` or `autofix` parameter.
- [ ] The notebook demonstrates inspecting issues with `render_html()` and `get_issues()`.
- [ ] Shows the issue types (COLLISION_TRIVIAL auto-fixes, PREDICTION_ONLY, COLLISION_CROSS_BRANCH) with explanatory markdown cells, noting that only UNRESOLVED blocks.
- [ ] Shows resolving `UNRESOLVED` labels with `map()` before calling `get_mapped_results_dataframe()`.
- [ ] Shows that `get_mapped_results_dataframe()` raises `IncompleteMapping` only when UNRESOLVED labels remain (brief demo cell).
- [ ] Output DataFrame uses `original_annotation` / `original_prediction` columns — notebook references these where applicable.
- [ ] Remove any references to `canonical_depth`, `eval_entities`, `hierarchy` parameter on `analyze()`, or old issue types.
- [ ] Notebook runs end-to-end without error.

---

### US-016: Update Notebook 5 (Evaluate Custom Presidio Analyzer)
**Description:** As a user evaluating a custom or third-party model, I want Notebook 5 to
show the new API for handling models with different label vocabularies.

**Acceptance Criteria:**
- [ ] Replace `mapper.analyze(results_df, hierarchy=2)` with `mapper.analyze(results_df)` (depth auto-discovered).
- [ ] Remove any pre-mapping of labels to `None` before `analyze()` — instead show that PREDICTION_ONLY is INFO and non-blocking; optionally demonstrate `map({"LABEL": None})` to suppress one.
- [ ] Show iterating `get_issues()` to identify any UNRESOLVED (ERROR) labels and resolve them programmatically.
- [ ] Demonstrate that DATASET_ONLY and PREDICTION_ONLY entities are both INFO (non-blocking) and explain their default behavior (FP and FN counted respectively).
- [ ] Use the new issue type names in markdown explanations.
- [ ] Remove references to `canonical_depth`, `hierarchy` parameter on `analyze()`, or old issue types (`DEPTH_MISMATCH`, `SUPPRESSED_WITH_ANNOTATIONS`).
- [ ] Notebook runs end-to-end without error.

---

### US-017: Update Notebook 6 (Interactive Entity Mapping Tutorial)
**Description:** As a user learning entity mapping in depth, I want Notebook 6 to be a
comprehensive tutorial covering the full two-phase workflow, all issue types, and interactive
resolution.

**Acceptance Criteria:**
- [ ] Rewrite intro cells to explain the two-phase model: Identify (5 tiers) → Project (majority-vote depth + projection rules).
- [ ] Add a section explaining each of the 5 issue types with concrete examples and when they occur (table or visual).
- [ ] Demonstrate per-branch majority-vote depth auto-discovery: show what depth is computed for each branch in the sample dataset and why.
- [ ] Demonstrate each projection rule with at least one example: exact match, descendant auto-fix, ancestor auto-fix, cross-branch (informational), unresolved.
- [ ] Show `resolve_interactively()` for guided resolution of `UNRESOLVED` labels.
- [ ] Show `map()` for programmatic batch resolution.
- [ ] Demonstrate eval-surface locking: `analyze()` a second model and verify `eval_surface` is unchanged.
- [ ] Remove the old granularity comparison (`hierarchy=3 vs 2`) — replace with an explanation that depth is now data-driven.
- [ ] Remove references to `canonical_depth`, `eval_entities`, `hierarchy` parameter on `analyze()`, or old issue types.
- [ ] Include a section on `EntityHierarchy` customization (custom hierarchy dict) — updated for new API.
- [ ] Notebook runs end-to-end without error.

---

### US-018: Update docs/entity_hierarchy.md
**Description:** As a documentation reader, I want the entity mapping guide to reflect the
new two-phase workflow and data-driven depth so the docs match the code.

**Acceptance Criteria:**
- [ ] Update "How it works" section to mention the two-phase process (Identify → Project).
- [ ] Update "The canonical vocabulary" section: replace fixed-depth description with explanation that eval depth is auto-discovered from the dataset via majority vote.
- [ ] Update "Typical workflow" code snippet:
  - Remove `hierarchy=2` from `analyze()` call.
  - Show `get_issues()` and `map()` for resolution.
  - Show that `get_mapped_results_dataframe()` blocks until `UNRESOLVED` labels are resolved.
  - Remove the "evaluate at a broader level" snippet that uses `analyzer.analyze(results_df, hierarchy=2)`.
- [ ] Add a brief note that the eval surface is locked after first `analyze()` for multi-model comparison.
- [ ] Remove references to `canonical_depth`, `eval_entities`, or `hierarchy` parameter on `analyze()`.
- [ ] All code examples are syntactically valid.

---

### US-019: Update docs/mapping_scenarios.md
**Description:** As a documentation reader, I want the mapping scenarios document to use the
new terminology and reference the new issue types.

**Acceptance Criteria:**
- [ ] Replace references to `canonical_depth=2` / `canonical_depth=3` with explanation of data-driven depth.
- [ ] Map each scenario to the new issue type it would produce (e.g., Scenario 1 → `DATASET_ONLY`, Scenario 8 → `PREDICTION_ONLY`).
- [ ] Update Scenario 2 ("Massive Model Entity Set") to reference projection rules instead of depth parameter.
- [ ] Update Scenario 5 ("Country-Prefixed Labels") to mention the COUNTRY and COUNTRY_FALLBACK identification tiers.
- [ ] Update Scenario 7 ("Label Collision Across Branches") to reference `COLLISION_CROSS_BRANCH` (INFO; informational only, does not block). Explain that same-branch depth mismatches are auto-resolved as COLLISION_TRIVIAL and not surfaced as issues.
- [ ] Update Scenario 8 ("Prediction-Only") to reference `PREDICTION_ONLY` issue type and resolution options (suppress, remap, keep-as-FP).
- [ ] Remove references to old issue types or API parameters.

---

### US-020: Update docs/why_canonical_entity_mapping.md
**Description:** As a documentation reader, I want the design rationale document to reflect
the data-driven depth approach.

**Acceptance Criteria:**
- [ ] Update "Why depth 3 as the default?" section to explain that depth is now auto-discovered via per-branch majority vote, not a fixed default. Depth 3 is still the cap but the eval surface is data-driven.
- [ ] No code snippets in this doc reference old API parameters.

---

### US-021: Hierarchical multi-level evaluation convenience wrapper
**Description:** As a user who wants to evaluate model performance at multiple entity
granularity levels simultaneously, I want a convenience wrapper that produces L0/L1/L2 scores
in one call so I don't have to manually prepare the DataFrame for each level.

**Acceptance Criteria:**
- [ ] A `calculate_hierarchical_scores(mapped_df)` convenience function (or method) accepts the output of `get_mapped_results_dataframe()`.
- [ ] It produces evaluation scores at three levels:
  - **L0** — PII vs O (binary: is this token a PII entity at all?)
  - **L1** — branch-level (depth-2 nodes: PERSON, LOCATION, etc.)
  - **L2** — canonical surface (depth-3 nodes: NAME, STREET_ADDRESS, etc.)
- [ ] Internally, for each level, the function maps annotation/prediction values to the appropriate depth node and calls `SpanEvaluator` — no changes to `SpanEvaluator` required.
- [ ] Returns a `dict[str, EvaluationResult]` mapping level name (`"L0"`, `"L1"`, `"L2"`) to its result.
- [ ] Documented in [ADR-003: Hierarchical Multi-Level Evaluation](../docs/adr/ADR-003-hierarchical-evaluation.md).
- [ ] Covered by unit tests.

---

### US-022: Compact summary table (`render_summary()`)
**Description:** As a user, I want a compact one-line-per-label summary view so I can quickly
scan all resolved labels without reading the full audit table.

**Acceptance Criteria:**
- [ ] `render_summary()` renders a compact HTML table with columns: **Label** | **Mapped to** | **Annotations** | **Predictions** | **Confidence**.
- [ ] "Mapped to" shows the projected eval-surface entity; if the canonical differs from the projected entity, shows `(via CANONICAL)` inline.
- [ ] "Confidence" column shows a visual bar for FUZZY-resolved labels (with score); shows a text label for other tiers (EXACT, COUNTRY, etc.).
- [ ] Uses `IPython.display.HTML` in Jupyter; falls back to plain text. Never raises.
- [ ] Can be called at any point after `analyze()`.

---

## Functional Requirements

- **FR-01:** Constructor accepts only `hierarchy` (dict or `EntityHierarchy`, optional) and `fuzzy_threshold` (float, default 0.80). No `labels`, `canonical_depth`, `eval_entities`, or majority-vote parameter.
- **FR-02:** Before any resolution, normalize each label by stripping BIO/BIOES/BILOU/BILUO tagging scheme prefixes and suffixes. The original label remains the dict key.
- **FR-03:** The special outside token `O` (exactly, after stripping) is automatically mapped to `None`.
- **FR-04:** `analyze()` runs identification only (EXACT → COUNTRY → COUNTRY_FALLBACK → FUZZY → UNRESOLVED). No majority vote. No projection. Issues are detected and sorted after identification.
- **FR-05:** ~~Majority-vote depth~~ — removed. Annotations keep their native hierarchy depth.
- **FR-06:** Issues are per-model (cleared on each `analyze()` call). No surface locking.
- **FR-07:** Predictions are compared against annotations via the descendant-credit rule in `SpanEvaluator`. The mapper only provides hierarchy position; it does not project labels.
- **FR-08:** `COLLISION_TRIVIAL` issue type removed. Same-branch depth mismatches are not surfaced by the mapper — they are handled correctly by the hierarchical evaluator.
- **FR-09:** For COLLISION_CROSS_BRANCH issues, count token co-occurrence with each annotation label from the DataFrame and store in `issue.overlap_counts`. Displayed in the audit table as informational context.
- **FR-10:** PREDICTION_ONLY entities are INFO and non-blocking. By default, every prediction of that entity counts as FP. Optional resolution via `map()`: suppress (`None`) or remap to a surface entity.
- **FR-11:** DATASET_ONLY entities default to keep (FN counted). This is INFO, non-blocking.
- **FR-12:** `get_mapped_results_dataframe()` raises `IncompleteMapping` if any `UNRESOLVED` (ERROR) issues remain. Output includes `original_annotation` and `original_prediction` columns.
- **FR-13:** `map()` validates all entries atomically before applying. Returns `self`. After applying, re-evaluates which issues are resolved.
- **FR-14:** `resolve_interactively(prompt_fn=input)` prompts only for `UNRESOLVED` (ERROR) issues. Accepts a canonical entity name or `NONE` to suppress.
- **FR-15:** `render_html()` shows summary bar + detail table. Uses `IPython.display.HTML` when available; falls back to `print()`. Never raises.
- **FR-16:** Logging uses `presidio_evaluator.entity_mapping` logger. See US-010 for formats.
- **FR-17:** `analyze()` returns `self` for chaining.
- **FR-18:** `get_issues()` returns issues sorted by severity (ERROR > WARNING > INFO), then by token count descending.
- **FR-19:** ~~`eval_surface` property~~ — removed. There is no canonical surface.
- **FR-20:** Systematic misprediction insights surfaced in `render_html()` for cross-branch overlaps with high token counts.

---

## Non-Goals (Out of Scope)

- No GUI or web interface — terminal prompts only.
- No persistence of mappings to disk (callers can serialize `get_mapping()` themselves).
- No semantic/embedding-based similarity (`sentence-transformers` removed entirely).
- No model label extraction utilities (deferred to follow-up).
- No `canonical_depth` or `eval_entities` parameter — depth is purely data-driven.
- No `labels` parameter on the constructor — labels are discovered from the DataFrame.
- No `autofix` parameter on `analyze()` — auto-fix is always on for COLLISION_TRIVIAL.
- No automatic suppression of PREDICTION_ONLY entities — default behavior counts them as FP; suppression is optional via `map()`.
- `label_finder.py` remains a separate utility, not integrated into `CanonicalMapper`.

---

## Technical Considerations

### Codebase structure

**Files to modify:**
- `presidio_evaluator/entity_mapping/data_objects.py` — replace all `IssueType` values with: `UNRESOLVED`, `COLLISION_TRIVIAL`, `COLLISION_CROSS_BRANCH`, `PREDICTION_ONLY`, `DATASET_ONLY`. Remove `COLLISION_AMBIGUOUS` and old types: `DEPTH_MISMATCH`, `SUPPRESSED_WITH_ANNOTATIONS`, `HIERARCHY_DEPTH_CHANGED`. Only `UNRESOLVED` is ERROR severity; all others are INFO. Add `overlap_counts` field to `MappingIssue` for cross-branch token-overlap data.
- `presidio_evaluator/entity_mapping/mapper.py` — full rewrite of `CanonicalMapper`: two-phase logic (identify + project), majority-vote depth, eval-surface locking, projection rules, token-overlap ranking, blocking behavior, `original_` columns in output DataFrame.
- `presidio_evaluator/entity_mapping/__init__.py` — update exports (remove old types, add new ones).

**Files to keep as-is:**
- `presidio_evaluator/entity_mapping/hierarchy.py` — existing `EntityHierarchy` with `canonicalize()`, `get_branch()`, `canonical_to_branch`, `all_canonical_entities`. May need a `get_depth(entity)` helper method added.
- `presidio_evaluator/entity_mapping/definitions.py` — `HIERARCHY` dict and `COUNTRIES` set unchanged.
- `presidio_evaluator/entity_mapping/label_finder.py` — separate utility, unchanged.

**Notebooks to update:**
- `notebooks/4_Evaluate_Presidio_Analyzer.ipynb` — replace `CanonicalMapper(canonical_depth=3)` with `CanonicalMapper()`, remove old params, show new issue types and blocking workflow.
- `notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb` — replace `analyze(results_df, hierarchy=2)` with `analyze(results_df)`, remove pre-mapping, show new programmatic resolution pattern.
- `notebooks/6_Interactive_Entity_Mapping.ipynb` — full rewrite as two-phase workflow tutorial: identification tiers, projection rules, all 5 issue types with examples, interactive resolution of UNRESOLVED labels, eval-surface locking, multi-model comparison.

**Documentation to update:**
- `docs/entity_hierarchy.md` — update workflow code snippet, remove `hierarchy` param on `analyze()`, explain data-driven depth.
- `docs/mapping_scenarios.md` — map scenarios to new issue types, remove `canonical_depth` references.
- `docs/why_canonical_entity_mapping.md` — update depth rationale for data-driven approach.

**Tests to rewrite:**
- `tests/entity_mapping/test_canonical_mapper.py` — full rewrite for new two-phase logic.

**Dependencies to remove:**
- `sentence-transformers` from `pyproject.toml` (if still present).

### Public API

```python
class CanonicalMapper:
    def __init__(
        self,
        *,
        hierarchy: dict | EntityHierarchy | None = None,
        fuzzy_threshold: float = 0.80,
    ) -> None: ...

    def analyze(self, results_df: pd.DataFrame) -> CanonicalMapper: ...

    @property
    def pending(self) -> list[str]: ...
    @property
    def eval_surface(self) -> set[str]: ...

    def get_issues(self) -> list[MappingIssue]: ...
    def map(self, mappings: dict[str, str | None]) -> CanonicalMapper: ...
    def resolve_interactively(self, prompt_fn=input) -> CanonicalMapper: ...
    def get_mapping(self) -> dict[str, str | None]: ...
    def get_mapped_results_dataframe(self) -> pd.DataFrame: ...
    def render_html(self) -> None: ...
    def render_summary(self) -> None: ...
```

### Input DataFrame format

| Column | Type | Description |
|---|---|---|
| `sentence_id` | `int` | Index of the source sentence in the dataset |
| `token` | `str` | Token string |
| `annotation` | `str` | Ground-truth entity tag |
| `prediction` | `str` | Model-predicted entity tag |
| `start_indices` | `int` | Character start position of the token |

### Output DataFrame format

Same columns as input, plus:
- `annotation` and `prediction` rewritten to eval-surface entities (suppressed → `"O"`).
- `original_annotation` and `original_prediction` preserving the raw labels.

### Internal state

- `_eval_surface: set[str] | None` — locked after first `analyze()`.
- `_eval_depth: int | None` — the auto-discovered depth.
- `_records: dict[str, _Resolution]` — per-label resolution (tier, canonical, score).
- `_issues: list[MappingIssue]` — per-model, cleared on each `analyze()`.
- `_results_df: pd.DataFrame | None` — the most recent results DataFrame.

### Key helper needed on `EntityHierarchy`

A `get_depth(entity: str) -> int` method that returns the depth of a canonical entity in the
hierarchy tree (root `PII` = depth 1, `PERSON` = depth 2, `NAME` = depth 3, `FIRST_NAME` = depth 4).
The current `canonical_to_branch` dict already contains branch paths, so depth = `len(branch)`.

---

## Success Metrics

- All `UNRESOLVED` labels must be resolved before DataFrame extraction — zero silent evaluation errors.
- Auto-fix resolves the majority of collisions (COLLISION_TRIVIAL) without user intervention.
- Multi-model comparisons use identical eval surfaces — `eval_surface` property returns the same set across `analyze()` calls.
- `ruff check` passes with zero errors on all changed files.
- Full test suite passes with `pytest`.

---

## Open Questions

1. **Majority vote tie-breaking** — When `round()` lands at exactly 0.5 (e.g., 2.5), use the
   deeper (more specific) depth. Confirmed in design discussion.

2. **`map({"X": "X"})` for PREDICTION_ONLY** — When a user maps a prediction entity to itself,
   it stays in the eval as an entity not in the dataset. Every prediction of that type counts as
   FP. The `render_html()` should make this consequence explicit (now optional, not required).

3. **Cross-branch insight format** — For systematic mispredictions, `render_html()` could show
   a mini confusion matrix (top 3 cross-branch overlaps per entity). Currently surfaced as
   `overlap_counts` in the issue object; richer visualization is a potential follow-up.

4. **Hierarchical evaluation output format** — `calculate_hierarchical_scores()` returns
   `dict[str, EvaluationResult]`; consider whether a merged DataFrame or a pretty-print method
   is more useful for notebook workflows. See ADR-003 for full design.

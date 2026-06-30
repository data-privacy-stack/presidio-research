# ADR-003: Hierarchical Multi-Level Evaluation

## Status

Proposed

## Date

2026-04-29 (revised 2026-04-30)

## Context

The current evaluation pipeline produces a single F1 score per entity type. Before evaluation
can run, every model label must be projected to match the dataset's labels, otherwise the
 evaluation would result in errors. This forces a choice whenever a model and dataset operate
 at different granularity levels:

- A model predicting `LOCATION` cannot be fairly evaluated against a dataset that annotates
  `STREET_ADDRESS` and `GPE` separately — any single mapping corrupts the metric for one of
  them.
- Projecting annotations to a specific hierarchy silently modifies the ground truth and
  can penalize models that are more specific than required.

The underlying issue is not a mapping problem — it is a **granularity mismatch** between the
model and the dataset. No single flat projection resolves it without losing information.

## Decision

Introduce a **hierarchical multi-level evaluation** mode that scores models at multiple
depth levels simultaneously, rather than requiring a single flat canonical surface.

### Core idea

Evaluate precision, recall and F2 independently at each depth level of the entity hierarchy.
Annotations are **never modified** — they keep their native depth. At each level, the
annotation is mapped to its ancestor at that level, and the prediction is compared against it.

**The key scoring rule:** a prediction is a TP at level L if it equals the annotation's
ancestor at L *or is a descendant of it in the hierarchy* (i.e., the model was more specific
than required). A prediction coarser than the annotation at level L is FP+FN at L.

Consider a dataset that annotates `TITLE` and `PERSON` (PERSON branch), and a model that
predicts `PERSON` for all of them:

| token | annotation | prediction | L0 — PII/not PII | L1 — PERSON | L2 — more specific |
|---|---|---|---|---|---|
| "Dr." | `TITLE` (depth 3) | `PERSON` (depth 2) | **TP** | **TP** | **FP+FN** — model predicted coarser than annotation |
| "John" | `PERSON` (depth 2) | `PERSON` (depth 2) | **TP** | **TP** | **TP** |
| "Jane" | `PERSON` (depth 2) | `PERSON` (depth 2) | **TP** | **TP** | **TP** |
| "Prof." | `TITLE` (depth 3) | `PERSON` (depth 2) | **TP** | **TP** | **FP+FN** |

Now with Model B predicting `NAME` and `TITLE`:

| token | annotation | prediction | L0 | L1 | L2 |
|---|---|---|---|---|---|
| "Dr." | `TITLE` | `TITLE` | **TP** | **TP** | **TP** |
| "John" | `PERSON` | `NAME` | **TP** | **TP** | **TP** — `NAME` is a descendant of `PERSON`; more specific than required → credit given |
| "Jane" | `PERSON` | `NAME` | **TP** | **TP** | **TP** |
| "Prof." | `TITLE` | `TITLE` | **TP** | **TP** | **TP** |

Model B is perfect at all levels. Model A is penalised only at L2 for the `TITLE` tokens — it
found the right branch but was too coarse for the annotation's depth. No majority vote, no
COLLISION_AMBIGUOUS, no surface projection.

### Scoring rule

At each level L, for each token:

1. Map the **annotation** to its ancestor at level L (or keep as-is if already at or above L).
2. The prediction is a **TP** if it equals the mapped annotation *or is a descendant of it* in the hierarchy (the model was more specific than required — credit is given).
3. The prediction is **FP+FN** if it is an ancestor of the mapped annotation (too coarse), on a different branch, or `O` when the annotation is non-`O`.
4. The prediction is **FP** if it is non-`O` when the annotation is `O`.

This means:
- A token annotated at depth 2 (`PERSON`) is evaluated at L2 using the depth-2 label — any prediction of `PERSON` or a descendant (`NAME`, `TITLE`, etc.) counts as TP.
- A token annotated at depth 3 (`TITLE`) requires the prediction to be `TITLE` or a descendant to score TP at L2; a prediction of `PERSON` (ancestor) is FP+FN.

Standard precision / recall / F1 apply at each level without modification. No new metric
definitions are needed.

The levels correspond to the natural structure of `EntityHierarchy`:
- **L0**: PII vs. non-PII (binary)
- **L1**: depth-2 branch node (PERSON, LOCATION, CONTACT, …)
- **L2**: the annotation's own depth (depth-3 for depth-3 annotations; depth-2 for depth-2 annotations)

### Granularity bonus / penalty

The scoring rule makes granularity visible without any penalty parameter.
The L2 F1 gap between models directly quantifies how much granularity matters for a given dataset.

### Implementation sketch

**Step 1 — Mapping**

1. `CanonicalMapper.analyze()` identifies every raw label in `EntityHierarchy` (EXACT → FUZZY →
UNRESOLVED). No projection, no surface computation. Output: each label has a hierarchy node
or UNRESOLVED.

2. `get_hierarchical_mapped_dataframe()` emits one annotation/prediction column pair per depth
level. For each token, the annotation is mapped to its ancestor at that level; the prediction
is kept as-is (the scorer applies the descendant rule):

```
sentence_id | token | annotation | prediction | annotation_l0 | annotation_l1 | annotation_l2
```

- `annotation_l0`: `PII` or `O`
- `annotation_l1`: depth-2 branch node or `O`
- `annotation_l2`: the annotation's native hierarchy node (depth 2 or 3), or `O`

Predictions are left at their native labels; the evaluator applies the descendant-credit rule
when comparing.

**Step 2 — Per-level evaluation**

`BaseEvaluator.calculate_hierarchical_scores` applies the descendant-credit rule: 
a prediction equal to or a descendant of the target annotation
counts as TP. When not provided, it uses the default annotated entities hierarchy.

```python
results = evaluator.calculate_hierarchical_scores(mapped_df)
```

Or a convenience wrapper that returns a `HierarchicalEvaluationResult`:

```python
results = evaluator.calculate_hierarchical_scores(mapped_df)
results["L0"]   # PII detection EvaluationResult
results["L1"]   # branch-level EvaluationResult
results["L2"]   # annotation-native-depth EvaluationResult
```

**Step 3 — Reporting**

The `Plotter` should be updated to render grouped bar chart showing L0 / L1 / L2 F1 per entity branch.
It further can shows the confusion matrix and error analysis for each level.

## Consequences

### Positive

- **No forced mapping choice** — mixed-depth branches and same-branch depth mismatches are handled automatically at every level without user intervention.
- **Finer models are rewarded** — a model predicting `NAME` on a `PERSON`-annotated token scores TP, not FP+FN. Specificity is an asset.
- **Coarser models are correctly penalized** — a model predicting `PERSON` on a `TITLE`-annotated token is FP+FN at L2, TP at L1. The granularity gap is explicit and quantified.
- **Interpretable at every level** — L0 answers "did the model find PII?", L1 answers "did it get the category right?", L2 answers "did it match or exceed the annotation's specificity?".
- **No new metric definitions** — standard F1 at each level. Results are directly comparable across teams and papers.
- **L0 is privacy-critical** — detecting that a span is PII at all is the most safety-relevant question.

### Negative / Trade-offs

- **Three scores instead of one** — reporting becomes more complex. Users need guidance on which level to optimize for their use case.
- **Requires `EntityHierarchy` for every label** — labels that are `UNRESOLVED` cannot be placed at any level and must still be handled before multi-level evaluation can run.
- **L0 precision is trivially high for entity-rich models** — a model that fires on everything gets near-perfect L0 recall at the cost of precision; L0 alone is not enough. The multi-level view makes this visible rather than hiding it.
- **Descendant-credit rule requires hierarchy access in SpanEvaluator** — a minor change to the evaluator is needed to pass the hierarchy and apply the rule.

## Alternatives Considered

### 1. Partial-credit F1 (continuous hierarchical penalty)

Weight each TP/FP/FN by a distance function in the hierarchy (e.g. `1 / (1 + depth_distance)`).
Rejected because partial-credit scores are difficult to interpret — "F1 = 0.81 with α = 0.5"
requires context that makes cross-team comparison unreliable.

### 2. LCA collapse per pair (ADR-002 Alternative 2)

Map both sides to their Lowest Common Ancestor per pair. Rejected — same reasons as in ADR-002:
different pairs converge at different depths, and annotations are modified, undermining the
dataset-as-ground-truth principle.

### 3. Keep flat mapping with explicit coarse-level surface

Let the user specify `eval_depth=1` to evaluate at the branch level. Rejected because it
requires upfront knowledge of the right depth and doesn't give the multi-level view that
makes model weaknesses diagnosable.

# Why canonical entity mapping?

This document explains the design decisions behind the canonical entity mapping approach used in Presidio Evaluator. For the taxonomy structure and usage guide, see [Entity hierarchy](entity_hierarchy.md).

## Approaches to entity label mapping

Different tools and datasets use incompatible label vocabularies. Several strategies exist for reconciling them during
evaluation; each has a different cost/comparability trade-off.

**1. Score against own labels** — each model is evaluated only on the entity types it natively supports, with no
mapping. Requires nothing from the user but makes cross-model comparison impossible — models are evaluated on different
things.

**2. One model's schema as standard** — all models map to one model's native label set. Simple but arbitrary: it
privileges one model's worldview, penalizes others for not conforming to it, and creates a moving target if that
model's schema changes.

**3. User labels as standard** — model outputs map to whatever labels the user used in their dataset. Feels natural
but breaks down quickly — user labels are inconsistent across datasets, often ambiguous, and can't be reused across
evaluations.

**4. Multi-label annotation** — users tag the same span multiple times, anticipating each model's vocabulary (e.g. a
span tagged as both `US_SSN` and `ID`). Eliminates the mapping layer but requires every model's vocabulary to be known
and stable at annotation time; re-annotation is needed whenever a model is added or changed. It also introduces scoring
ambiguity: if a span carries two labels and a model detects only one, how many false negatives does it generate? A
model's recall on a given entity type becomes a function of how many co-labels its spans carry rather than purely of
its detection quality.

**5. Interactive per-model mapping** — users map their labels to each model's vocabulary interactively, one model at a
time. Gives full transparency but repeats the mapping effort per model per dataset, with no guarantee of consistency
across models.

**6. Canonical entities** (this approach) — a tool-owned, model-agnostic schema that every model and every user
dataset maps to once. The only approach that achieves full comparability, reusability across users, and stability over
time. The trade-off is a one-time mapping step; mapping decisions can silently affect scores, making transparency in
the mapping layer important.

| Approach | Comparability | User burden | Stability | Customizability |
|---|---|---|---|---|
| 1. Score against own labels | None | None | High | Low |
| 2. One model as standard | Biased | Low | Low | Low |
| 3. User labels as standard | Per dataset | Medium | Low | High |
| 4. Multi-label annotation | Inconsistent | High | Low | Medium |
| 5. Interactive per-model mapping | Inconsistent | High | Medium | High |
| **6. Canonical entities** | **Full** | **Once** | **High** | **Medium** |

## Why not a flat alias map?

A flat `{raw: canonical}` dict was the original approach. It breaks down as the number of participating tools grows —
every new model means manually adding dozens of aliases. The nested hierarchy makes the *relationship* between entities
explicit and lets the country-prefix engine generate thousands of aliases automatically.

## Why depth 3 as the default?

Depth 2 (the domain branches) is useful for coarse-grained comparison (e.g. "did the model find any government ID?").
Depth 3 provides enough specificity for meaningful evaluation without fragmenting into micro-types that no realistic
model distinguishes. Depth-4+ entities exist for completeness but are intentionally aggregated upward during evaluation.

**Granularity is now branch-specific and data-driven:** rather than accepting a fixed
`canonical_depth` parameter, `CanonicalMapper` uses the annotation depth on each hierarchy branch.
More-specific predictions are projected upward to that depth, while less-specific predictions are
not projected downward. Mixed annotation depths on one branch block evaluation until the caller makes
an explicit mapping decision.

This requires no manual tuning when each annotation branch uses one granularity. Multi-model
comparisons remain consistent when every model is analyzed against the same annotations and explicit
mapping policy.

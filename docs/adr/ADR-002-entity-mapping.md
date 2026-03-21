# ADR-002: Entity Mapping via CanonicalMapper

## Status

Proposed

## Date

2026-03-21

## Context

Users of Presidio Evaluator bring their own entity label vocabularies — from custom datasets,
fine-tuned models, or third-party tools. Before evaluation can happen, every user-defined label
must be resolved to a canonical entity so that dataset annotations and model predictions can be
compared on equal footing.

The current approach has two pain points:

1. **Mapping is required for meaningful cross-model comparison, but the current approach is too simplistic** — comparing different models against a shared dataset requires that every model's predictions and the dataset's annotations use a common label vocabulary. Without reliable mapping, results are biased: labels that should be considered equivalent are treated as different, inflating false-negative counts and depressing recall. The existing mapping code handles only the simplest cases; real-world label vocabularies include tagging-scheme prefixes, country-prefixed document types, and near-synonyms that the current logic silently drops or fails on, requiring significant manual intervention to get trustworthy numbers.

2. **No structured resolution pipeline** — the existing code does not distinguish between labels
   that are exact matches, fuzzy matches, or unknown. All unmapped labels are silently dropped or
   cause errors, giving users no visibility into what happened.

## Decision

Introduce a single, stateful class — `CanonicalMapper` — as the sole entry point for resolving
user-defined labels to **canonical entities**. A canonical entity is a normalised, taxonomy-defined
label drawn from the `EntityHierarchy` vocabulary (e.g. `PERSON`, `EMAIL_ADDRESS`,
`PASSPORT`). Mapping a raw label to a canonical entity means finding the single taxonomy entry that
best represents the concept the raw label describes. Once every label on both sides of an evaluation
— dataset annotations and model predictions — has been mapped to a canonical entity, scores can be
computed fairly: identical canonical entities count as matches regardless of how the two sides
originally spelled or tagged them.

The same class is used for both sides of an evaluation: the dataset's label vocabulary and the
model's output label vocabulary. Both are resolved independently; evaluation then compares
canonical-to-canonical.

### Resolution tiers

`CanonicalMapper` attempts each tier in order, stopping at the first match:

| Resolution Tier | Matching Condition | Output |
|---|---|---|
| **EXACT** | Label (after normalisation) matches a known alias | Canonical entity |
| **COUNTRY** | Label begins with a recognised country prefix; remainder resolves to a known document type | Canonical entity |
| **COUNTRY_FALLBACK** | Label begins with a recognised country prefix; remainder is not a known document type | `NATIONAL_ID` (with warning) |
| **FUZZY** | Approximate string match against the alias vocabulary meets the confidence threshold | Canonical entity |
| **PENDING** | None of the above | Requires user action |

After the user resolves pending labels (via `map()` or `resolve_interactively()`), resolved labels
are tagged **MANUAL** or **NONE** (suppressed from evaluation).

Before any tier is attempted, BIO/BIOES/BILOU tagging scheme prefixes and suffixes are stripped
transparently (e.g. `B-PERSON` is looked up as `PERSON`). The original label remains the key in
all outputs.

### Workflow

```
CanonicalMapper(labels)
  → auto-resolve pass (EXACT → COUNTRY → COUNTRY_FALLBACK → FUZZY)
  → inspect .pending          # labels that need attention
  → .map({...})               # programmatic assignment
  → .resolve_interactively()  # guided terminal/notebook prompts
  → .get_mapping()            # returns dict[str, str | None]
```

When all labels auto-resolve, `.get_mapping()` can be called immediately — no additional steps
are needed. When pending labels remain, `.get_mapping()` raises `IncompleteMapping` until every
label is accounted for.

### Typical usage

```python
# `from_dataset` is a convenience constructor: it extracts entity labels from the
# dataset's spans, runs the auto-resolve pass, and returns the mapping dict directly
# if every label resolves automatically. When labels are still pending it returns
# the CanonicalMapper instance so the caller can resolve them before calling
# .get_mapping().

# Everything auto-resolves — mapping dict returned immediately
mapping = CanonicalMapper.from_dataset(samples)

# Some labels are pending — resolve interactively, then retrieve
mapper = CanonicalMapper(["PERSON", "EMAIL_ADDRESS", "MY_CUSTOM_LABEL"])
mapper.resolve_interactively()
mapping = mapper.get_mapping()

# Programmatic (batch) assignment — no prompts
mapper.map({"MY_CUSTOM_LABEL": "PERSON", "INTERNAL_ID": None})
mapping = mapper.get_mapping()
```

### Logging

Every resolution decision is logged at `INFO` level, with country-prefix fallback at `WARNING`
and unresolvable labels at `WARNING`. A summary line after the auto-resolve pass reports how many
labels were resolved, how many were fuzzy, and how many are pending. This gives users a full audit
trail without inspecting internal state.

## Consequences

### Positive

- **Single resolution entry point** — all mapping logic lives in one place; scattered
  `align_entity_types` calls across `BaseModel` and `BaseEvaluator` are replaced.
- **Transparency** — every resolution decision is logged and visible in the HTML audit table
  (`render_html()`). Users always know how each label was handled.
- **Guided manual resolution** — `resolve_interactively()` shows ranked fuzzy suggestions for
  pending labels, making manual mapping fast even for large vocabularies.
- **Composable** — `get_mapping()` returns a plain `dict[str, str | None]` that can be passed
  to any downstream evaluation code, serialised, or version-controlled without additional tooling.
- **Atomic batch updates** — `map()` validates all entries before applying any, preventing
  partial state corruption.

### Negative / Trade-offs

- **`IncompleteMapping` is a hard stop** — pipelines that previously silently dropped unknown
  labels will now fail explicitly until all labels are resolved or suppressed. This is intentional
  but requires pipeline changes for existing users.

## Alternatives Considered

### 1. Score against own labels (no mapping)

Each model is evaluated only on the entity types it natively supports, with no mapping at all.
This requires nothing from the user but makes cross-model comparison impossible — models are
evaluated on different sets of entities, so their scores are not comparable. High customizability
since each model is unconstrained, but that freedom undermines any meaningful benchmark.

### 2. Manual mapping only

Require users to supply a complete `dict[str, str]` upfront and apply it verbatim, with no
auto-resolution. This is simple to implement and fully predictable, but it is burdensome for
large or evolving label vocabularies where most mappings are straightforward. Users must enumerate
every label manually, including trivially resolvable ones, before evaluation can start.

### 3. Semantic similarity (embedding-based matching)

Use a sentence-transformer model to embed both the raw label and all canonical entity names, then
pick the nearest neighbour as the resolved canonical. This can handle abbreviated or
domain-specific labels that string matching misses. It was rejected because it adds a heavy ML
dependency (PyTorch, transformers, tokenizers) for a task whose label vocabulary is finite and
known; the quality gain over fuzzy string matching does not justify the install-time and runtime
overhead. It also makes resolution non-deterministic across model versions.

### 4. A stateless mapping function

A pure function `resolve_labels(labels, hierarchy) → dict` with no state or interactive
capability was considered. This was rejected because it cannot handle the fallback path for labels
that cannot be auto-resolved — callers would have to implement their own retry loops and conflict
resolution, defeating the goal of a single reusable entry point.

### 5. Embed mapping inside the model prediction step

Passing an entity mapping directly to the model's `predict` call was considered. This was
rejected because mapping is a property of the *evaluation goal* (which canonical entities to
score), not of the model. Keeping mapping separate makes both components independently testable
and allows the same model output to be evaluated under different mapping configurations.

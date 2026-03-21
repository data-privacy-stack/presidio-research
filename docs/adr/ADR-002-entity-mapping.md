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

The current approach has five pain points:

1. **Mapping logic is scattered** — entity type remapping lives in `BaseModel.align_entity_types`,
   `BaseEvaluator.align_entity_types`, and `InputSample` processing. There is no single, reusable
   entry point for the transformation.

2. **No structured resolution pipeline** — the existing code does not distinguish between labels
   that are exact matches, fuzzy matches, or unknown. All unmapped labels are silently dropped or
   cause errors, giving users no visibility into what happened.

3. **Semantic-similarity dependency** — the current `SemanticEntityMapper` relies on
   `sentence-transformers` to find approximate matches. This adds a heavy ML dependency for a task
   that can be solved adequately with lightweight string matching against a known vocabulary.

4. **No interactive fallback** — when a label cannot be resolved automatically, there is no guided
   way for the user to supply the correct canonical entity. They must edit source code or
   configuration files directly.

5. **No audit trail** — users cannot tell how each label was resolved (exact match, approximate
   match, country-prefix logic, manual entry) without reading the source code.

## Decision

Introduce a single, stateful class — `CanonicalMapper` — as the sole entry point for resolving
user-defined labels to canonical entities. The same class is used for both sides of an evaluation:
the dataset's label vocabulary and the model's output label vocabulary. Both are resolved
independently; evaluation then compares canonical-to-canonical.

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

### Removing the semantic-similarity dependency

`sentence-transformers` is removed from the project dependencies entirely. All entity resolution
goes through the `EntityHierarchy` alias vocabulary and lightweight string matching — no
embeddings, no external ML models.

## Consequences

### Positive

- **Single resolution entry point** — all mapping logic lives in one place; scattered
  `align_entity_types` calls across `BaseModel` and `BaseEvaluator` are replaced.
- **Transparency** — every resolution decision is logged and visible in the HTML audit table
  (`render_html()`). Users always know how each label was handled.
- **Lighter dependency footprint** — removing `sentence-transformers` eliminates a large
  transitive dependency chain (PyTorch, transformers, tokenizers) from the default install.
- **Guided manual resolution** — `resolve_interactively()` shows ranked fuzzy suggestions for
  pending labels, making manual mapping fast even for large vocabularies.
- **Composable** — `get_mapping()` returns a plain `dict[str, str | None]` that can be passed
  to any downstream evaluation code, serialised, or version-controlled without additional tooling.
- **Atomic batch updates** — `map()` validates all entries before applying any, preventing
  partial state corruption.

### Negative / Trade-offs

- **Breaking change for `SemanticEntityMapper` users** — the class is removed; callers must
  migrate to `CanonicalMapper`. No automatic migration path is provided.
- **Resolution quality ceiling** — fuzzy string matching has lower recall on abbreviated or
  domain-specific labels than semantic similarity. Users with highly non-standard vocabularies
  may have more labels land in `pending`.
- **Stateful object** — `CanonicalMapper` holds mutable resolution state. Callers that share an
  instance across threads or processes must handle concurrency themselves.
- **`IncompleteMapping` is a hard stop** — pipelines that previously silently dropped unknown
  labels will now fail explicitly until all labels are resolved or suppressed. This is intentional
  but requires pipeline changes for existing users.

## Alternatives Considered

### Keep `SemanticEntityMapper` and patch it

Extending the existing semantic-similarity approach with better logging and an interactive
fallback was considered. This was rejected because the `sentence-transformers` dependency is
disproportionate to the task: resolving a finite, known vocabulary of entity labels does not
benefit from embeddings, and the dependency imposes significant install-time and runtime costs.

### A stateless mapping function

A pure function `resolve_labels(labels, hierarchy) → dict` with no interactive capability was
considered. This was rejected because it cannot handle the interactive fallback path needed when
labels cannot be auto-resolved. Callers would have to implement their own retry loops, defeating
the goal of a single reusable entry point.

### Embed mapping inside `BaseModel.predict_dataset`

Passing an entity mapping directly to `predict_dataset` was considered (see ADR-001). This was
rejected because mapping is a property of the *evaluation goal* (which canonical entities to
score), not of the model. Keeping it separate makes both components independently testable and
allows the same model output to be evaluated under different mapping configurations.

### A fully automatic mapping with no `pending` state

Resolving every unknown label to a default value (e.g. `None`) without surfacing failures was
considered. This was rejected because silent data loss is worse than an explicit error. Surfacing
`pending` labels ensures users make conscious decisions about every label in their vocabulary.

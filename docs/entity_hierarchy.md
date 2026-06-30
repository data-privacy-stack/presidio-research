# Entity Mapping

## The problem

When you evaluate a PII detection model, the model and the dataset almost never use the same labels. Your dataset might call something `ORGANIZATION`, while the model calls it `ORG`. Your dataset says `PHONE_NUMBER`, the model says `PHONE`. Some models output `B-PERSON` (with BIO tags), others just `PERSON`. Some emit `GERMANY_PASSPORT_NUMBER`, others just `PASSPORT`.

If you compare these labels directly, everything looks like a mismatch — even when the model found the right thing.

## How it works

Presidio Evaluator solves this with a **shared vocabulary** of canonical entity names. Every label — from any model or dataset — gets mapped to one of these canonical names before evaluation using a **two-phase process**:

**Phase 1 — Identify:** each label is matched to a canonical entity through five tiers (in priority order):
1. Exact match in the alias map
2. Country-prefix strip (e.g. `GERMANY_PASSPORT_NUMBER` → `PASSPORT`)
3. Country-prefix fallback (tries removing leading country code)
4. Fuzzy string match (≥0.80 similarity by default)
5. `UNRESOLVED` — flagged for manual resolution

Examples:
- `EMAIL`, `email_address`, `EMAILADDRESS` → all become `EMAIL_ADDRESS`
- `B-PERSON`, `PERSON-I` → BIO tags are stripped, both become `NAME`
- `GERMANY_PASSPORT_NUMBER` → country prefix is recognized, becomes `PASSPORT`
- `CREDITCARD`, `credit_card` → case and delimiters don't matter, becomes `FINANCIAL`

**Phase 2 — Project:** canonical entities are projected onto the *canonical surface* — a set of entities at a
depth computed by majority vote from the **dataset annotation labels**. Depth-2 ancestors that have
multiple depth-3 descendants trigger a `COLLISION_AMBIGUOUS` issue; descendants that have exactly
one matching ancestor on the canonical surface are auto-fixed as `COLLISION_TRIVIAL`.

Labels that can't be resolved automatically are flagged for you to handle manually.

## The canonical vocabulary

The canonical entities are organized in a hierarchy. At the default evaluation level (depth 3), there are entities like `EMAIL_ADDRESS`, `PASSPORT`, `NAME`, `STREET_ADDRESS`, etc. These sit under broader categories:

| Category | Entities it includes |
|----------|---------------------|
| `PERSON` | Names, usernames, aliases |
| `CONTACT` | Email, phone, fax, social handles |
| `LOCATION` | Addresses, cities, countries, geo-coordinates |
| `ORGANIZATION` | Companies, schools, government agencies |
| `GOVERNMENT_ID` | SSN, passport, driver license, tax ID |
| `FINANCIAL_PII` | Credit cards, bank accounts, crypto wallets |
| `PHI` | Patient IDs, health insurance, conditions, medications |
| `DATE_TIME` | Dates, times, durations |

(Full list: `DEMOGRAPHIC`, `EMPLOYMENT`, `DEVICE_IDENTIFIER`, `BIOMETRIC`, `NETWORK_IDENTIFIER`, `AUTHENTICATION`, `VEHICLE_PII`, `LEGAL_PII`, `TRAVEL_PII`, `EDUCATION`)

The **evaluation depth is data-driven**: `CanonicalMapper` computes a weighted majority vote across the
annotation labels in your results DataFrame and selects depth 2 or 3 (capped at 3). Depth 3 is the most
common outcome when a dataset uses fine-grained entity types like `EMAIL_ADDRESS`, `NAME`, or `SSN`.

For more on why this approach was chosen over alternatives, see [why_canonical_entity_mapping.md](why_canonical_entity_mapping.md).

## Typical workflow

```python
from presidio_evaluator.entity_mapping import CanonicalMapper, IncompleteMapping

# 1. Create a mapper (no constructor arguments needed)
mapper = CanonicalMapper()

# 2. Analyze your data — labels are auto-resolved, canonical depth is inferred from annotations
mapper.analyze(results_df)

# 3. Inspect issues
mapper.render_html()                               # visual overview in Jupyter
for issue in mapper.get_issues():
    print(f"[{issue.severity.value}] {issue.type.value}: {issue.labels}")

# 4. Fix WARNING/ERROR issues before extracting the DataFrame
#    map to a canonical entity, or None to suppress
mapper.map({"MY_CUSTOM_LABEL": "EMAIL_ADDRESS"})
mapper.map({"JUNK_LABEL": None})                    # None = exclude from evaluation

# 5. get_mapped_results_dataframe() raises IncompleteMapping if WARNING/ERROR issues remain
try:
    mapped_df = mapper.get_mapped_results_dataframe()
except IncompleteMapping:
    print("Resolve blocking issues first — call get_issues() to see them")

# 6. Look up where a specific label was mapped
print(mapper.get_mapping(entity="ORG"))
```

If you already know some labels need specific mappings, pre-map them **before** `analyze()`
so they don't appear as issues:

```python
mapper = CanonicalMapper()
mapper.map({"MY_CUSTOM_LABEL": "EMAIL_ADDRESS", "JUNK_LABEL": None})
mapper.analyze(results_df)
```

**Multi-model comparison:** the canonical surface (set of entities used for evaluation) is locked after the
first `analyze()` call. Subsequent `analyze()` calls for other models reuse the same surface, ensuring
all models are evaluated on the same entity set:

```python
mapper = CanonicalMapper()
mapper.analyze(model_a_df)   # locks canonical surface from dataset annotations
mapper.analyze(model_b_df)   # reuses the same locked canonical surface
```

---

## Common issues

When you call `mapper.analyze(df)`, the mapper checks for problems that could silently distort your scores. Call `mapper.get_issues()` to see them — sorted by severity (ERROR first), then by token count. There are six issue types:

| Type | Severity | Meaning |
|------|----------|---------|
| `UNRESOLVED` | ERROR | Label could not be matched — blocks `get_mapped_results_dataframe()` |
| `COLLISION_AMBIGUOUS` | WARNING | Depth-2 label maps to multiple depth-3 entities — blocks extraction |
| `COLLISION_CROSS_BRANCH` | WARNING | Label maps across hierarchy branches — blocks extraction |
| `PREDICTION_ONLY` | WARNING | Label only in predictions, not dataset — blocks extraction |
| `COLLISION_TRIVIAL` | INFO | Auto-fixed: descendant collapsed to single ancestor |
| `DATASET_ONLY` | INFO | Label only in dataset annotations (model never predicts it) |

> **Warning**: `get_mapped_results_dataframe()` raises `IncompleteMapping` if any WARNING or ERROR issues remain. INFO issues are non-blocking.

### 1. A label could not be resolved to any canonical entity

This happens when the mapper encounters a label it doesn't recognize — it's not in the alias map, doesn't fuzzy-match anything above the threshold, and doesn't have a recognizable country prefix.

**Example:** Your dataset has spans annotated as `MEDICAL_RECORD` but that label doesn't exist in the hierarchy and isn't close enough to any alias to fuzzy-match.

**The problem:** Unresolved labels pass through to evaluation unchanged. Since neither side recognizes them as a canonical entity, they won't match anything — silently inflating your false positive and false negative counts. This also blocks `get_mapping()` from returning a complete mapping.

**Fix 1. — map it to the right canonical entity:**
```python
mapper.map({"MEDICAL_RECORD": "MEDICAL_RECORD_NUMBER"})  # or whichever canonical fits
```

**Fix 2. — suppress it if it's irrelevant:**
```python
mapper.map({"MEDICAL_RECORD": None})  # exclude from evaluation
```

The issue's `resolution_options` will suggest the closest canonical match (if any) and always offer a suppress option.

### 2. Mixed granularity — labels resolve to parent/child categories

This happens when two labels that refer to related things end up at different levels in the hierarchy. It can show up in two ways:

**In the mapping:** Your model outputs `MY_LOCATION` which you mapped to `LOCATION`, and your dataset has `MY_CITY` which auto-resolved to `ADDRESS`. `ADDRESS` is a more specific type of `LOCATION` — they're the same concept family, but at different levels.

**In specific rows:** Your dataset annotates a span as `ADDRESS` and the model predicts `STREET_ADDRESS` for the same span. Both refer to the same real-world thing, but at different levels of specificity.

**The problem:** In both cases, the evaluator counts this as wrong — even though the model found the right span. Your scores will look worse than they actually are.

**Fix 1. — align them manually.** Pick the more specific or the broader one, and map both labels to the same level:
```python
mapper.map({"MY_LOCATION": "ADDRESS"})      # go specific
# or
mapper.map({"MY_CITY": "LOCATION"})          # go broad
```

**Fix 2. — align to the depth the canonical surface uses.** The canonical surface is computed automatically
from the dataset annotations. If your dataset uses depth-2 labels (e.g. `PERSON`), the canonical surface
will be at depth 2, and depth-3 model labels will auto-collapse. If the dataset uses depth-3 labels,
use `map()` to remap the model's depth-2 label to a specific depth-3 target:
```python
mapper.map({"PERSON": "NAME"})  # pick the right depth-3 entity
```

### 3. You marked a label that has real annotations in the dataset as not important (mapped to `None`)

There are two very different reasons you might do this, and only one of them is safe:

**Safe — suppressing a model's label:** If a model predicts `ORGANIZATION` but your dataset doesn't have it annotated, suppressing it is fine. You're choosing not to evaluate the model on that entity type. That's a deliberate decision you control.

**Problematic — suppressing an annotated label:** If your dataset has spans annotated as `ORGANIZATION`, suppressing that label discards ground truth that were deliberately tagged. Those spans vanish from the evaluation entirely — no false negatives, no false positives, no PII coverage. Your evaluation results will silently say nothing about that entity type, even though your dataset covers it.

**Example:** You called `mapper.map({"MY_ORG": None})`. The model's `MY_ORG` predictions are removed — fine. But your dataset also has 50 annotated `MY_ORG` spans. Those are removed too. Your evaluation now has a blind spot for `MY_ORG` with no indication in the scores.

**Fix — map it to a canonical instead of suppressing it:**
```python
mapper.map({"MY_ORG": "ORGANIZATION"})  # or whichever canonical fits
```

If you genuinely want to exclude it (e.g. it's a catch-all junk label with no real meaning), keep `None` — but do it consciously and note it in your experiment log.

### 4. The model predicts an entity type that the dataset doesn't cover

**Example:** Your dataset has annotations for `PERSON` and `ORGANIZATION`. The model predicts `STREET_NUMBER` for some spans. `STREET_NUMBER` maps to the canonical entity `ADDRESS` — but no label in your dataset maps to `ADDRESS`. There are no gold annotations to compare against.

**The problem:** Every `STREET_NUMBER` prediction counts as a false positive — not because the model is wrong, but because the dataset simply has no spans that map to the same canonical entity. This makes the model's precision look worse than it is.

**Fix 1. — suppress the entity if your dataset doesn't cover it:**

```python
mapper.map({"STREET_NUMBER": None})
```

**Fix 2. — add the missing annotations:** If the model is actually finding real PII that was missed during annotation, the right fix is to go back and annotate those spans.

### 5. The canonical surface is locked — a new model uses different entities

The canonical surface (set of entities used for evaluation) is **locked after the first `analyze()` call**.
This ensures all models are evaluated on the same entity set for fair comparison.

**Example:** You evaluated model A and locked the surface at depth 3. Model B predicts a label that
would have changed the canonical depth if analyzed alone. It's projected onto the existing locked surface.

**This is intentional.** The dataset annotations define the ground truth, so the canonical surface is
anchored to the first model's dataset. If model B has labels that don't fit the surface, they appear
as `COLLISION_AMBIGUOUS` or `PREDICTION_ONLY` issues — resolve them with `map()` before extracting
results.

# ADR-002: Dataset-Anchored Entity Mapping via CanonicalMapper

## Status

Proposed

## Date

2026-04-30

## Context

Users of Presidio Evaluator bring their own entity label vocabularies — from custom datasets,
fine-tuned models, or third-party tools. Before evaluation can happen, every label must be
resolved to a shared vocabulary so that dataset annotations and model predictions can be compared.

The core use case is **comparing multiple models against the same dataset**. The dataset defines
the evaluation contract; models are the variable.

A flat `dict[str, str]` mapping is insufficient because:
- Many labels are aliases for the same concept (`FIRST_NAME`, `NAME_GIVEN`, `GIVENNAME` → `NAME`). Maintaining a hand-crafted dict for hundreds of model vocabularies is burdensome.
- Labels exist in a hierarchy — `NAME` is a sub-type of `PERSON`. A model predicting `PERSON` on a `NAME`-annotated token is partially correct, not wrong. A flat dict cannot express this.
- Unresolved labels need to be surfaced and triaged before evaluation; a dict silently drops or mismaps them.

> **Relationship to ADR-003:** [Hierarchical evaluation](ADR-003-hierarchical-evaluation.md) builds on top of this mapping layer. Hierarchical scoring handles same-branch depth differences without any label projection — so this ADR only needs to surface labels that are **entirely absent** from the hierarchy.

## Decision

### Core principle

Introduce `CanonicalMapper` as the sole entry point for resolving labels to evaluation entities.
It locates each raw label in the `EntityHierarchy` taxonomy using ordered resolution tiers.
Annotations keep their native depth. How predictions are scored against the hierarchy is defined in ADR-003.
There is no canonical surface and no depth parameter.

The mapper operates on the **paired results DataFrame** (one row per token, with both `annotation` and `prediction` columns) rather than a bare label list for two reasons:

1. **Complete label discovery** — both annotation and prediction vocabularies must be resolved before evaluation. Inspecting only one side would leave the other side unmapped.
2. **Impact-ranked issue triage** — by counting how many tokens are affected by each issue, the mapper surfaces the most impactful problems first. Token counts require the paired data.

### Identification tiers

Each label is looked up in order, stopping at the first match:

| Tier | Matching Condition | Output |
|---|---|---|
| **EXACT** | Label (after normalization) matches a known alias | Hierarchy node |
| **COUNTRY** | Label has a recognized country prefix; remainder is a known document type | Hierarchy node |
| **COUNTRY_FALLBACK** | Label has a recognized country prefix; remainder is unknown | `NATIONAL_ID` (with warning) |
| **FUZZY** | Approximate match against alias vocabulary meets confidence threshold | Hierarchy node |
| **UNRESOLVED** | None of the above | Requires user action |

BIO/BIOES/BILOU prefixes are stripped transparently before lookup (`B-PERSON` → `PERSON`).

### Issue types

| Issue Type | Default Severity | Description | Default Action |
|---|---|---|---|
| **UNRESOLVED** | ERROR | Label absent from hierarchy entirely | User must map or suppress |
| **COLLISION_CROSS_BRANCH** | WARNING | A prediction label and annotation label co-occur on the same tokens but map to different hierarchy branches. This may be a **vocabulary mismatch** (e.g., the model calls it `ORG` while the dataset calls it `COMPANY`) that can be fixed with `map()`. Even when not remapped, it is surfaced so the user is aware of the mismatch when interpreting results. | Surfaced in audit table with token counts; does not block |
| **PREDICTION_ONLY** | WARNING | Prediction entity in hierarchy but never annotated by the dataset | Surfaced in audit table; does not block |
| **DATASET_ONLY** | WARNING | Annotation entity never predicted by the model (nor any descendant) | Surfaced in audit table; does not block |
| **COLLISION_SAME_BRANCH** | INFO | A prediction label and annotation label co-occur on the same tokens and map to the **same hierarchy branch** but at different depths (e.g., model predicts `PERSON`, dataset annotates `NAME`). This is not a mapping problem — it is handled correctly by the hierarchical evaluation step (see ADR-003). | Shown inline in audit table for awareness; does not block |

Issues are ordered by severity (ERROR > WARNING > INFO), then by affected token count (descending).

By default, issues at WARNING level and above are surfaced. The user can control the minimum severity shown in the audit table and returned by `get_issues()` via the `min_severity` parameter on `analyze()`:

- `min_severity="ERROR"` — show only blocking issues
- `min_severity="WARNING"` — show blocking issues and vocabulary/coverage gaps *(default)*
- `min_severity="INFO"` — show everything, including same-branch depth differences

`COUNTRY_FALLBACK` is part of the identification phase and resolves to a node — it is not a separate issue type. It is logged internally and never shown in the audit table regardless of `min_severity`.

### Blocking behavior

`get_mapped_results_dataframe()` raises `IncompleteMapping` only if `UNRESOLVED` issues remain.
To resolve:
- `mapper.map({"MY_LABEL": "CANONICAL"})` — map to a known hierarchy entity
- `mapper.map({"MY_LABEL": None})` — suppress from evaluation entirely

### Workflow

```python
mapper = CanonicalMapper()
mapper.analyze(results_df)                        # default: surface WARNING and above
# mapper.analyze(results_df, min_severity="INFO") # surface everything incl. same-branch depth diffs
# mapper.analyze(results_df, min_severity="ERROR") # surface only blocking issues

mapper.render_html()                 # audit table — blocking issues first
for issue in mapper.get_issues():
    print(issue)

# Resolve UNRESOLVED labels (the only blocking type)
mapper.map({"MY_UNKNOWN_LABEL": "CANONICAL_ENTITY"})
mapper.map({"UNWANTED_LABEL": None})

mapped_df = mapper.get_mapped_results_dataframe()
# pass mapped_df to the evaluator (see ADR-003 for scoring)
```

## Consequences

### Positive

- **Dataset-anchored** — the dataset's label vocabulary is the evaluation contract; no projection or depth parameter needed.
- **Mixed-granularity datasets work naturally** — a dataset with both `PERSON` (depth 2) and `STREET_ADDRESS` (depth 3) annotations needs no special handling; each label is placed at its own depth.
- **Minimal blocking** — only labels genuinely absent from the hierarchy block extraction.
- **Frequency-driven triage** — highest-impact problems surface first.
- **Composable** — `get_mapping()` returns a plain `dict[str, str | None]` that can be serialised or version-controlled.

### Negative / Trade-offs

- **Dataset must be present at mapping time** — the mapper requires a results DataFrame with both annotations and predictions; there is no label-only mode.

## Alternatives Considered

### 1. Manual dictionary only

Require users to supply a complete `dict[str, str]` before evaluation. Too burdensome for large
vocabularies (hundreds of model-specific aliases) where most resolutions are unambiguous.

### 2. Semantic similarity (embedding-based matching)

Use a sentence-transformer to embed labels and pick nearest neighbours. Rejected: heavy ML
dependency for a finite, well-structured label space, and non-deterministic results across model
versions.


## Example mappings

A sample of how raw labels from common NER models and datasets resolve to hierarchy nodes. The full alias list lives in [`presidio_evaluator/entity_mapping/definitions.py`](../../presidio_evaluator/entity_mapping/definitions.py).

| Raw Label | Canonical Entity | Notes |
|-----------|-----------------|-------|
| `FIRST_NAME`, `GIVENNAME`, `NAME_GIVEN` | `NAME` | Alias resolution |
| `PER`, `PERSON` | `PERSON` | Coarser node — same branch as `NAME` |
| `STREET`, `STREETADDRESS` | `ADDRESS` | Alias resolution |
| `LOCATION` | `LOCATION` | Branch node |
| `SOCIALNUM`, `US_SSN`, `UK_NINO` | `SSN` | Alias resolution |
| `GERMANY_DRIVER_LICENSE` | `DRIVER_LICENSE` | Country-prefix tier |
| `FRANCE_UNKNOWN_DOC` | `NATIONAL_ID` | Country-prefix fallback (warning) |
| `CREDITCARD`, `IBAN`, `BITCOIN` | `FINANCIAL` | Alias resolution |
| `MY_CUSTOM_LABEL` | *(UNRESOLVED)* | Requires `mapper.map()` |

## Entity hierarchy

The full hierarchy is defined in [`presidio_evaluator/entity_mapping/definitions.py`](../../presidio_evaluator/entity_mapping/definitions.py). A condensed excerpt illustrating the structure:

```python
HIERARCHY: dict = {
    "PII": {
        "PERSON": {
            "NAME": {
                "FIRST_NAME": ["FIRSTNAME", "NAME_GIVEN", "GIVENNAME", ...],
                "LAST_NAME":  ["LASTNAME", "SURNAME", "NAME_FAMILY", ...],
                "FULL_NAME":  ["FULLNAME", "DOCTOR", "PATIENT_NAME", ...],
            },
            "TITLE":    [],
            "USERNAME": ["USER_NAME", "DISPLAYNAME", ...],
        },
        "LOCATION": {
            "ADDRESS": {
                "STREET_ADDRESS": ["STREET", "STREETADDRESS", ...],
                "CITY":           ["LOCATION_CITY"],
                "POSTAL_CODE":    ["ZIPCODE", "ZIP", "POSTCODE", ...],
                "COUNTRY":        ["LOCATION_COUNTRY", ...],
            },
            "GEO_COORDINATES": ["GPSCOORDINATES", "GEOCOORD", ...],
        },
        "GOVERNMENT_ID": {
            "SSN":            ["SOCIALNUMBER", "US_SSN", ...],
            "PASSPORT":       ["PSP", "PASSPORT_NUMBER", ...],
            "DRIVER_LICENSE": ["DRIVERLICENSE", "DLN", ...],
            "NATIONAL_ID":    ["IDCARD", "IDNUM", "IN_AADHAAR", ...],
        },
        # ... (see definitions.py for the full hierarchy)
    }
}
```

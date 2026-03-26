# PII Entity Hierarchy

This document explains the design of the canonical PII entity taxonomy used throughout Presidio Evaluator,
and shows how to work with the `EntityHierarchy` API.

## Overview

Different PII detection tools and models
each define their own entity label vocabularies. A model trained for healthcare may emit `PATIENT_NAME` or `HCW`, while another
emits `FULLNAME` or `PassengerID`. Evaluation across these tools requires a shared canonical vocabulary that every raw label
can be normalized to.

`presidio_evaluator.entity_mapping` provides:

- **`HIERARCHY`** — a single nested Python dict that is the authoritative taxonomy.
- **`EntityHierarchy`** — a class that wraps the taxonomy and exposes canonicalization, branch lookup, BIO prefix stripping, and alias extension.
- **`CanonicalMapper`** — a workflow class that resolves a full set of raw model/dataset labels through auto-resolution, fuzzy matching, and manual override.

---

## Taxonomy structure

`HIERARCHY` is a nested `dict`. Each key is an entity name; the value is either:

- a **`dict`** → an intermediate (parent) node that has children, or
- a **`list`** → a leaf node whose list contains raw aliases that also resolve to this entity.

### Depth and canonicalization

The default canonical depth is **3** (passed as `canonical_depth=3` to `EntityHierarchy.__init__`). Nodes are counted from the root (`PII` = depth 1):

| Depth | Role | Behaviour |
|-------|------|-----------|
| 1 | Root (`PII`) | Self-maps — `h.canonicalize("PII")` → `"PII"` |
| 2 | Domain branch (e.g. `GOVERNMENT_ID`, `CONTACT`) | Self-maps — `h.canonicalize("CONTACT")` → `"CONTACT"` |
| 3 | **Canonical entity** (e.g. `EMAIL_ADDRESS`, `PASSPORT`) | The resolution target |
| 4+ | Fine-grained sub-type (e.g. `CARD_NUMBER` under `CREDIT_CARD` under `FINANCIAL`) | Rolls up to its depth-3 ancestor |

canonicalization is **case- and delimiter-agnostic**: `credit_card`, `CREDIT_CARD`, and `CreditCard` all resolve the
same way.

### Top-level branches

| Branch | What it covers |
|--------|----------------|
| `PERSON` | Names, usernames, aliases |
| `DEMOGRAPHIC` | Age, gender, ethnicity, nationality, physical descriptors |
| `CONTACT` | Email, phone, fax, social handles |
| `LOCATION` | Addresses (street → postal code), geo-coordinates |
| `ORGANIZATION` | Companies, schools, government agencies, medical facilities |
| `EMPLOYMENT` | Job titles, departments, employee/customer IDs |
| `GOVERNMENT_ID` | SSN, passport, driver license, tax ID, national ID, and similar |
| `FINANCIAL_PII` | Credit cards, bank accounts, crypto wallets, financial amounts |
| `DEVICE_IDENTIFIER` | Device IDs, IMEI, MAC address, user-agent |
| `BIOMETRIC` | Fingerprint, face, iris, DNA — GDPR Article 9 special-category data |
| `NETWORK_IDENTIFIER` | IP address, URL, domain, cookies |
| `AUTHENTICATION` | Passwords, PINs, API keys, tokens |
| `PHI` | Protected health information (patient ID, health insurance, conditions, medications, clinical research) |
| `VEHICLE_PII` | License plates, VIN |
| `LEGAL_PII` | Case numbers, court and arrest records, inmate IDs |
| `TRAVEL_PII` | Passenger name records, e-tickets, world-tracer numbers |
| `EDUCATION` | Student IDs, academic records, institution IDs |
| `DATE_TIME` | Dates, times, epochs, durations |

### Branches with extra depth

A few branches have a four-level structure where depth-2 (`*_PII`) wraps a depth-3 canonical (`FINANCIAL`,
`VEHICLE`, …) which in turn wraps depth-4 sub-types. The practical consequence is that fine-grained labels
roll up to the depth-3 canonical:

```
FINANCIAL_PII (depth 2, self-maps)
  └─ FINANCIAL (depth 3, canonical)
       ├─ CREDIT_CARD (depth 4, rolls up → "FINANCIAL")
       │    ├─ CARD_NUMBER, CVV, EXPIRATION, …  (depth 5)
       └─ BANK_ACCOUNT (depth 4, rolls up → "FINANCIAL")
            ├─ ACCOUNT_NUMBER, IBAN, SWIFT_BIC, …  (depth 5)
```

```python
h = EntityHierarchy()
h.canonicalize("CREDIT_CARD")   # → "FINANCIAL"
h.canonicalize("IBAN")          # → "FINANCIAL"
h.canonicalize("FINANCIAL_PII") # → "FINANCIAL_PII"  (depth-2 self-map)
```

### Country-prefix auto-mapping

Rather than listing every `URUGUAY_TAX_ID`, `AUSTRALIA_DRIVERS_LICENSE`, etc. explicitly, the module keeps two tables:

- **`COUNTRIES`** — all 249 ISO 3166-1 alpha-2 codes plus full English country name tokens, demonyms, and adjectival forms (e.g. `AUSTRALIA`, `GERMANY`, `BRITISH`, `FRENCH`).
- **`country_prefixed_doc_types`** — an instance attribute on `EntityHierarchy`; a suffix keyword → canonical entity mapping (e.g. `"DRIVER"` → `"DRIVER_LICENSE"`). Mutate directly: `h.country_prefixed_doc_types["MY_SUFFIX"] = "MY_CANONICAL"`.

Any `<COUNTRY>_<SUFFIX>` label is resolved automatically. An unrecognized suffix with a known country prefix defaults
to `"NATIONAL_ID"`.

```python
h = EntityHierarchy()
h.canonicalize("URUGUAY_TAX_ID")           # → "TAX_ID"
h.canonicalize("AUSTRALIA_DRIVERS_LICENSE") # → "DRIVER_LICENSE"
h.canonicalize("GERMANY_PASSPORT_NUMBER")   # → "PASSPORT"
h.canonicalize("BRAZIL_UNKNOWN_DOC")        # → "NATIONAL_ID"  (fallback)
```

---

## Design decisions

### Approaches to entity label mapping

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

### Why not a flat alias map?

A flat `{raw: canonical}` dict was the original approach. It breaks down as the number of participating tools grows —
every new model means manually adding dozens of aliases. The nested hierarchy makes the *relationship* between entities
explicit and lets the country-prefix engine generate thousands of aliases automatically.

### Why depth 3 as the default?

Depth 2 (the domain branches) is useful for coarse-grained comparison (e.g. "did the model find any government ID?").
Depth 3 provides enough specificity for meaningful evaluation without fragmenting into micro-types that no realistic
model distinguishes. Depth-4+ entities exist for completeness but are intentionally aggregated upward during evaluation.

### BIOMETRIC vs PHYSICAL_DESCRIPTOR

`BIOMETRIC` (under `PII`) covers **authentication-grade** physical identifiers — fingerprint, iris scan, DNA,
voice print — that are GDPR Article 9 special-category data. `PHYSICAL_DESCRIPTOR` (under `DEMOGRAPHIC`) covers
**soft observable attributes** (eye colour, height, skin tone) that describe a person but do not uniquely re-identify
them. `BLOOD_TYPE` sits under `PHI` because it surfaces in a clinical rather than demographic context.

### PHI vs DEMOGRAPHIC

Medical observations (health conditions, medications, injuries) are under `PHI`, not `DEMOGRAPHIC`, because they are
clinical data governed by health-data regulations (HIPAA, GDPR Article 9) rather than demographic attributes.

---

## Usage

### Basic lookup — `EntityHierarchy`

```python
from presidio_evaluator.entity_mapping import EntityHierarchy, EntityNotMappedError

h = EntityHierarchy()   # uses the built-in HIERARCHY at canonical_depth=3

h.canonicalize("EMAIL")           # → "EMAIL_ADDRESS"
h.canonicalize("date_of_birth")   # → "BIRTH_DATE"
h.canonicalize("CREDITCARD")      # → "FINANCIAL"

h.get_branch("PASSPORT")
# → ["PII", "GOVERNMENT_ID", "PASSPORT"]

h.get_branch("GERMANY_PASSPORT_NUMBER")
# → ["PII", "GOVERNMENT_ID", "PASSPORT"]
```

`EntityNotMappedError` (a subclass of `ValueError`) is raised for labels that cannot be resolved:

```python
try:
    h.canonicalize("TOTALLY_UNKNOWN")
except EntityNotMappedError as e:
    print(e)  # Unknown entity label: 'TOTALLY_UNKNOWN'
```

#### Fuzzy resolution

`canonicalize()` accepts an optional `threshold` (default `0.80`). Set `threshold=1.0` to force exact matching only:

```python
h.canonicalize("EMAIL_ADRES")               # → "EMAIL_ADDRESS"  (fuzzy match)
h.canonicalize("EMAIL_ADRES", threshold=1.0) # → EntityNotMappedError
```

#### BIO prefix stripping

Labels with BIO/BIOES/BILOU/BILUO prefixes or suffixes are automatically stripped before lookup:

```python
h.canonicalize("B-PERSON")   # → "NAME"
h.canonicalize("PERSON-I")   # → "NAME"
```

### Accessing instance lookup tables

The lookup tables built from the hierarchy are exposed as instance attributes:

```python
h = EntityHierarchy()

# dict[str, str] — normalized raw → canonical
h.raw_to_canonical["CREDITCARD"]   # → "FINANCIAL"

# list[str] — every canonical-depth node name
h.all_canonical_entities           # ["NAME", "EMAIL_ADDRESS", ...]

# dict[str, list[str]] — canonical → full ancestor path
h.canonical_to_branch["PASSPORT"]  # → ["PII", "GOVERNMENT_ID", "PASSPORT"]
```

### Normalizing a label

`EntityHierarchy.normalize()` is a static method that strips BIO prefixes/suffixes, uppercases, and removes `_` and `-`. It is the first step of every canonicalization:

```python
EntityHierarchy.normalize("b-email_address")  # → "EMAILADDRESS"
EntityHierarchy.normalize("PERSON-I")         # → "PERSON"
```

### Adding aliases

```python
h = EntityHierarchy()
h.add_alias("EMAIL_ADDRESS", "ELECTRONIC_MAIL")
h.canonicalize("ELECTRONIC_MAIL")  # → "EMAIL_ADDRESS"
```

`add_alias()` raises `KeyError` if `entity_name` is not found in the hierarchy. Each `add_alias()` call triggers a full rebuild of the internal lookup tables.

### Country-prefix customization

Add entries to `country_prefixed_doc_types` to teach the engine new suffix → canonical mappings:

```python
h = EntityHierarchy()
h.country_prefixed_doc_types["HEALTH_CARD"] = "HEALTH_INSURANCE_ID"
h.canonicalize("CANADA_HEALTH_CARD")  # → "HEALTH_INSURANCE_ID"

del h.country_prefixed_doc_types["HEALTH_CARD"]
```

### Custom taxonomy from scratch

Pass your own hierarchy dict to the constructor:

```python
from presidio_evaluator.entity_mapping import EntityHierarchy

custom = EntityHierarchy(
    hierarchy={"MY_ROOT": {"MY_BRANCH": {"MY_ENTITY": ["alias1", "alias2"]}}},
    canonical_depth=3,
)
custom.canonicalize("alias1")  # → "MY_ENTITY"
```

### Inspecting the tree

```python
h = EntityHierarchy()

print(h.all_canonical_entities)            # list of all canonical names
print(h.canonical_to_branch["PASSPORT"])   # ["PII", "GOVERNMENT_ID", "PASSPORT"]
```

### Mapping model output with `CanonicalMapper`

`CanonicalMapper` resolves a full set of raw entity labels — as used by models or evaluation datasets — to canonical entities. Auto-resolution runs at construction time (exact alias → country-prefix → fuzzy); unresolvable labels land in `pending` and must be handled manually before `get_mapping()` will succeed.

```python
from presidio_evaluator.entity_mapping import CanonicalMapper, IncompleteMapping

mapper = CanonicalMapper(["EMAIL_ADDRESS", "EMAILADRES", "MY_CUSTOM_LABEL"])
# repr: "CanonicalMapper(2 resolved, 1 pending)"

mapper.render_html()   # inspect in Jupyter; plain-text fallback in terminals
```

#### Handling pending labels

```python
# Option A: manually assign a canonical (or None to suppress from evaluation)
mapper.map({"MY_CUSTOM_LABEL": "EMAIL_ADDRESS"})

# Option B: resolve interactively in the terminal (shows ranked fuzzy suggestions)
mapper.resolve_interactively()

# Retrieve the final mapping dict
mapping = mapper.get_mapping()
# → {"EMAIL_ADDRESS": "EMAIL_ADDRESS", "EMAILADRES": "EMAIL_ADDRESS", "MY_CUSTOM_LABEL": "EMAIL_ADDRESS"}
```

`get_mapping()` raises `IncompleteMapping` (a `RuntimeError` subclass) if any labels remain pending.
`map()` is atomic — if any entry is invalid, no changes are applied.

#### Rendering the mapping for review

```python
# Return an HTML string for saving or embedding; pending labels shown, no exception raised
html = mapper.get_mapping(mode="html")

# Return a plain-text table string
text = mapper.get_mapping(mode="text")
```

#### Applying the mapping to evaluation results

```python
# results_df is the 5-column DataFrame returned by model.predict_dataset()
mapped_df = mapper.get_mapped_results_dataframe(results_df)

# Switch to a coarser hierarchy level
mapped_df = mapper.get_mapped_results_dataframe(results_df, hierarchy=2)
```

When annotation and prediction labels are related but resolve to different canonical entities at the current depth, a `UserWarning` is emitted. Passing `hierarchy=2` (or another depth) rebuilds the underlying `EntityHierarchy` and re-resolves all previously seen labels.

---

## Quick reference

### `EntityHierarchy`

| Symbol | Type | Description |
|--------|------|-------------|
| `EntityHierarchy()` | `EntityHierarchy` | Default instance at `canonical_depth=3` |
| `h.canonicalize(raw, threshold=0.80)` | `str` | Resolve raw label → canonical; fuzzy-enabled by default |
| `h.get_branch(raw)` | `list[str]` | Full ancestor path for a raw label |
| `EntityHierarchy.normalize(label)` | `str` (static) | Strip BIO prefix/suffix, uppercase, remove `_`/`-` |
| `h.add_alias(entity, alias)` | `None` | Add a raw alias to an existing entity |
| `h.raw_to_canonical` | `dict[str, str]` | Normalized raw → canonical |
| `h.all_canonical_entities` | `list[str]` | Every canonical-depth entity name |
| `h.canonical_to_branch` | `dict[str, list[str]]` | Canonical → full ancestor path |
| `h.country_prefixed_doc_types` | `dict[str, str]` | Suffix → canonical overrides for the country-prefix engine |

### `CanonicalMapper`

| Symbol | Type | Description |
|--------|------|-------------|
| `CanonicalMapper(labels)` | `CanonicalMapper` | Resolve a set of raw labels with auto + manual fallback |
| `mapper.pending` | `list[str]` | Unresolved labels, alphabetically sorted |
| `mapper.map(dict)` | `CanonicalMapper` | Manually assign canonicals (atomic, returns `self`) |
| `mapper.resolve_interactively()` | `CanonicalMapper` | Terminal prompt loop for pending labels |
| `mapper.get_mapping()` | `dict[str, str\|None]` | Final mapping (raises `IncompleteMapping` if pending) |
| `mapper.get_mapping(mode="html")` | `str` | HTML audit table (no exception on pending) |
| `mapper.get_mapping(mode="text")` | `str` | Plain-text audit table (no exception on pending) |
| `mapper.render_html()` | `None` | Display in Jupyter / print plain-text fallback |
| `mapper.get_mapped_results_dataframe(df)` | `pd.DataFrame` | Remap annotation/prediction columns |

### Module-level constants

| Symbol | Type | Description |
|--------|------|-------------|
| `HIERARCHY` | `dict` | The raw taxonomy dict |
| `COUNTRIES` | `set[str]` | All recognized country tokens |
| `EntityNotMappedError` | `ValueError` subclass | Raised for unresolvable labels |
| `IncompleteMapping` | `RuntimeError` subclass | Raised by `get_mapping()` when labels are still pending |

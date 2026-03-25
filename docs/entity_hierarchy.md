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
- **`EntityHierarchy`** — a class that wraps the taxonomy and exposes canonicalization, branch lookup, and a mutation API.
- **Module-level shortcuts** — `canonicalize()`, `get_branch()`, and `print_hierarchy()` delegate to a shared default
  instance so most callers never need to instantiate the class directly.

---

## Taxonomy structure

`HIERARCHY` is a nested `dict`. Each key is an entity name; the value is either:

- a **`dict`** → an intermediate (parent) node that has children, or
- a **`list`** → a leaf node whose list contains raw aliases that also resolve to this entity.

### Depth and canonicalization

The default canonical depth is **3** (passed as `canonical_depth=3` to `EntityHierarchy.__init__`). Nodes are counted from the root (`PII` = depth 1):

| Depth | Role | Behaviour |
|-------|------|-----------|
| 1 | Root (`PII`) | Self-maps — `canonicalize("PII")` → `"PII"` |
| 2 | Domain branch (e.g. `GOVERNMENT_ID`, `CONTACT`) | Self-maps — `canonicalize("CONTACT")` → `"CONTACT"` |
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
canonicalize("CREDIT_CARD")   # → "FINANCIAL"
canonicalize("IBAN")          # → "FINANCIAL"
canonicalize("FINANCIAL_PII") # → "FINANCIAL_PII"  (depth-2 self-map)
```

### Country-prefix auto-mapping

Rather than listing every `URUGUAY_TAX_ID`, `AUSTRALIA_DRIVERS_LICENSE`, etc. explicitly, the module keeps two tables:

- **`COUNTRIES`** — all 249 ISO 3166-1 alpha-2 codes plus full English country name tokens (e.g. `AUSTRALIA`, `GERMANY`).
- **`country_prefixed_doc_types`** — an instance attribute on `EntityHierarchy`; a suffix keyword → canonical entity mapping (e.g. `"DRIVER"` → `"DRIVER_LICENSE"`). Add entries via `h.add_country_doc_type()`.

Any `<COUNTRY>_<SUFFIX>` label is resolved automatically. An unrecognized suffix with a known country prefix defaults
to `"NATIONAL_ID"`.

```python
canonicalize("URUGUAY_TAX_ID")           # → "TAX_ID"
canonicalize("AUSTRALIA_DRIVERS_LICENSE") # → "DRIVER_LICENSE"
canonicalize("GERMANY_PASSPORT_NUMBER")   # → "PASSPORT"
canonicalize("BRAZIL_UNKNOWN_DOC")        # → "NATIONAL_ID"  (fallback)
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

### Quick lookup — module-level shortcuts

For most use cases, import the three module-level functions directly:

```python
from presidio_evaluator.entity_mapping import (
    canonicalize,
    get_branch,
    EntityNotMappedError,
)

canonicalize("EMAIL")           # → "EMAIL_ADDRESS"
canonicalize("date_of_birth")   # → "BIRTH_DATE"
canonicalize("CREDITCARD")      # → "FINANCIAL"

get_branch("PASSPORT")
# → ["PII", "GOVERNMENT_ID", "PASSPORT"]

get_branch("GERMANY_PASSPORT_NUMBER")
# → ["PII", "GOVERNMENT_ID", "PASSPORT"]
```

`EntityNotMappedError` (a subclass of `ValueError`) is raised for labels that cannot be resolved:

```python
try:
    canonicalize("TOTALLY_UNKNOWN")
except EntityNotMappedError as e:
    print(e)  # Unknown entity label: 'TOTALLY_UNKNOWN'
```

### Accessing the pre-built lookup tables

The module exposes read-only snapshots of the default instance's lookup tables — useful for bulk operations:

```python
from presidio_evaluator.entity_mapping import (
    RAW_TO_CANONICAL,       # dict[str, str]  — normalized raw → canonical
    ALL_CANONICAL_ENTITIES, # list[str]       — every depth-3 (or shallower-leaf) node
    CANONICAL_TO_BRANCH,    # dict[str, list] — canonical → ancestor path
)

# Map a list of model outputs in one comprehension
raw_labels = ["CREDITCARD", "EMAIL", "DATE_OF_BIRTH"]
canonical  = [RAW_TO_CANONICAL.get(lbl.upper().replace("_",""), lbl) for lbl in raw_labels]
```

### Custom hierarchy — mutation API

When you need to extend or restrict the default taxonomy for a specific project, create an independent copy and mutate
it. The original default instance is never modified.

```python
from presidio_evaluator.entity_mapping import EntityHierarchy

h = EntityHierarchy.default().copy()

# Add a new alias for an existing entity
h.add_alias("EMAIL_ADDRESS", "ELECTRONIC_MAIL")
h.canonicalize("ELECTRONIC_MAIL")  # → "EMAIL_ADDRESS"

# Add a new canonical entity under an existing branch
h.add_entity(["PII", "CONTACT"], "MESSAGING_APP", aliases=["WHATSAPP", "SIGNAL"])
h.canonicalize("WHATSAPP")  # → "MESSAGING_APP"

# Rename a node
h.rename_entity("MESSAGING_APP", "INSTANT_MESSAGE")
h.canonicalize("WHATSAPP")  # → "INSTANT_MESSAGE"

# Remove an entity (and all its children/aliases)
h.remove_entity("INSTANT_MESSAGE")

# Remove a single alias
h.remove_alias("EMAIL_ADDRESS", "ELECTRONIC_MAIL")
```

### Country-prefix Customization

```python
h = EntityHierarchy.default().copy()

# Teach the engine that a new suffix maps to an existing canonical
h.add_country_doc_type("HEALTH_CARD", "HEALTH_INSURANCE_ID")
h.canonicalize("CANADA_HEALTH_CARD")  # → "HEALTH_INSURANCE_ID"

h.remove_country_doc_type("HEALTH_CARD")
```

### Custom taxonomy from scratch

Pass your own hierarchy dict to the constructor:

```python
custom = EntityHierarchy(
    hierarchy={"MY_ROOT": {"MY_BRANCH": {"MY_ENTITY": ["alias1", "alias2"]}}},
    canonical_depth=3,
)
custom.canonicalize("alias1")  # → "MY_ENTITY"
```

### Inspecting the tree

```python
h = EntityHierarchy.default()
h.print_hierarchy()

print(h.all_canonical_entities)   # list of all canonical names
print(h.canonical_to_branch["PASSPORT"])  # ["PII", "GOVERNMENT_ID", "PASSPORT"]
```

---

## Quick reference

| Import | Type | Description |
|--------|------|-------------|
| `canonicalize(raw)` | `str` | Resolve a raw label to its canonical form |
| `get_branch(raw)` | `list[str]` | Full ancestor path for a raw label |
| `RAW_TO_CANONICAL` | `dict[str, str]` | Pre-built normalized-raw → canonical map |
| `ALL_CANONICAL_ENTITIES` | `list[str]` | Every canonical entity name |
| `CANONICAL_TO_BRANCH` | `dict[str, list[str]]` | Canonical → ancestor path map |
| `EntityHierarchy.default()` | `EntityHierarchy` | Shared read-only default instance |
| `EntityHierarchy.default().copy()` | `EntityHierarchy` | Mutable independent copy |
| `EntityNotMappedError` | `ValueError` subclass | Raised for unresolvable labels |
| `IncompleteMapping` | `RuntimeError` subclass | Raised by `get_mapping()` when labels are still pending |
| `HIERARCHY` | `dict` | The raw taxonomy dict |
| `COUNTRIES` | `set[str]` | All recognized country tokens |
| `h.country_prefixed_doc_types` | `dict[str, str]` | Instance attr: suffix → canonical mapping (mutate via `add_country_doc_type()`) |

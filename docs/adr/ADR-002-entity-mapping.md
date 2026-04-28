# ADR-002: Dataset-Anchored Entity Mapping via CanonicalMapper

## Status

Proposed

## Date

2026-03-21 (revised 2026-04-27)

## Context

Users of Presidio Evaluator bring their own entity label vocabularies — from custom datasets,
fine-tuned models, or third-party tools. Before evaluation can happen, every user-defined label
must be resolved to a common vocabulary so that dataset annotations and model predictions can be
compared on equal footing.

The core use case is **comparing multiple models against the same dataset**. The dataset defines
the evaluation contract — its entity vocabulary is the ground truth. Models are the variable;
the dataset is the constant.

The naive mapping design has the following pain points:

1. **Depth mismatch is the norm, not the exception.** A model predicting `PERSON` and a dataset annotating `FIRST_NAME` both resolve to different hierarchy levels. They never align, even though the model found the right span.

2. **The mapping target should come from the dataset, not from a parameter.** When comparing multiple models against the same dataset, the dataset's entity vocabulary is the evaluation contract. Models should be projected onto that vocabulary, not onto a dynamic mapping.

3. **No structured resolution pipeline** — the previous mapping does not allow for a simple and structured resolution flow that mitigates the mapping issues to the full extent.

4. **Issue triage lacks priority.** Issues surface in a flat list. Users with large label vocabularies need to see the highest-impact problems first — the collisions that affect the most tokens, not alphabetically sorted edge cases.

## Decision

### Core principle: the dataset defines the evaluation surface

Introduce a single, stateful class — `CanonicalMapper` — as the sole entry point for resolving user-defined labels to evaluation entities. The mapper operates in two phases:

1. **Identify** — locate each raw label (annotation and prediction) in the `EntityHierarchy` taxonomy. This uses resolution tiers (EXACT → COUNTRY → COUNTRY_FALLBACK → FUZZY → UNRESOLVED).
2. **Project** — determine the evaluation surface via majority-vote depth from annotations, then project **all** identified labels (both annotations and predictions) onto that surface. The canonical surface is the set of hierarchy nodes at the computed depth.

### Auto-discovered hierarchy depth (majority vote)

Instead of requiring the user to specify the granularity upfront, the mapper inspects the dataset's annotation labels after identification, determines their depth in the hierarchy, and uses a **majority vote** to select the dominant depth. This becomes the default evaluation depth.
The calculation uses the depth of each entity on the tree, times the number of tokens with this label. In other words, it's a weighted average of the entities level on the hierarchy tree:
`round(granularity) = Σ(depth × tokens) / Σ(tokens)`
To avoid overly-specific entities, labels at depth > 3 are treated as level 3.
This prevents highly granular sub-labels (e.g. `MAIDEN_NAME`) from pulling the evaluation depth deeper than the standard hierarchy levels.

Example:
- Dataset labels: `FIRST_NAME` (depth 4, 200 tokens), `LAST_NAME` (depth 4, 200), `EMAIL_ADDRESS` (depth 3, 50),
  `PHONE_NUMBER` (depth 3, 10), `STREET_ADDRESS` (depth 3, 50), `SSN` (depth 3, 10), `PERSON` (depth 2, 100).
- With a cap of max 3: `depth ≈ 2.84 = 3`.

The evaluation depth is purely data-driven — there is no `canonical_depth` or `eval_entities` parameter. This avoids the cognitive overhead of choosing a depth upfront and ensures the evaluation surface always reflects what the dataset actually contains.

### Identification tiers

The identification step locates each raw label in the hierarchy. It attempts each tier in order, stopping at the first match:

| Resolution Tier | Matching Condition | Output |
|---|---|---|
| **EXACT** | Label (after normalization) matches a known alias | Hierarchy node |
| **COUNTRY** | Label begins with a recognized country prefix; remainder resolves to a known document type | Hierarchy node |
| **COUNTRY_FALLBACK** | Label begins with a recognized country prefix; remainder is not a known document type | `NATIONAL_ID` (with warning) |
| **FUZZY** | Approximate string match against the alias vocabulary meets the confidence threshold | Hierarchy node |
| **SEMANTICALLY SIMILAR** (**NOT IMPLEMENTED**) | Which known entity is semantically closest to the given entity name  | Hierarchy node |
| **UNRESOLVED** | None of the above | Requires user action |

Before any tier is attempted, BIO/BIOES/BILOU tagging scheme prefixes and suffixes are stripped
transparently (e.g. `B-PERSON` is looked up as `PERSON`). The original label remains the key in
all outputs.

### Projection rules

After identification, every annotation and prediction label has a position in the hierarchy.
The projection step maps **all labels** (both annotations and predictions) onto the evaluation surface — the set of hierarchy nodes at the majority-vote depth:

| Situation | Action | Auto/Manual |
|---|---|---|
| **Exact match** — label already in the canonical surface | Keep as-is | Auto |
| **Descendant** — label is a child of an eval-surface entity (e.g., `FIRST_NAME` → `NAME` at depth 3) | Map up to the ancestor eval-surface entity | Auto (COLLISION_TRIVIAL, flagged to user) |
| **Ancestor, unambiguous** — label is a parent of exactly one eval-surface entity (e.g., `PERSON` → `NAME`) | Map down to that eval-surface entity | Auto (COLLISION_TRIVIAL, flagged to user) |
| **Ancestor, ambiguous** — label is a parent of multiple eval-surface entities | Flag as COLLISION_AMBIGUOUS | Manual — user decides |
| **Same branch, lateral** — label and eval-surface entity share an ancestor but are siblings/cousins | Flag as COLLISION_CROSS_BRANCH | Manual — user decides |
| **Different branch** — no shared lineage below the root | Flag as COLLISION_CROSS_BRANCH | Manual — user decides |
| **No hierarchy match** — label not in hierarchy at all | Flag as UNRESOLVED | Manual (same as identification phase) |
 

### Collision detection with frequency-based priority

When the mapper detects a conflict (ambiguous ancestor, cross-branch overlap, etc.), it counts
the **number of affected tokens** in the results DataFrame and ranks issues by descending
frequency. The most impactful problems surface first.

For ambiguous collisions, the mapper uses **actual per-row token overlap** to rank suggestions.
For a prediction entity that maps to multiple dataset entities, it counts how many DataFrame rows
have that prediction label alongside each candidate annotation label. This gives data-driven
suggestions (e.g., “`PERSON` overlaps with `NAME` on 450 tokens, `TITLE` on 30 tokens, `USERNAME`
on 5 tokens”).

Note that due to prediction errors, we are expected to have wrong collisions: prediction entities
colliding with multiple dataset entities or dataset entities colliding with multiple prediction
entities. In such cases, the tool will present the most common collision first.
For example, if `PERSON` tokens in the dataset collide with both `TITLE` and `ORG`, the tool
should first try to map to the more common match.

Systematic mispredictions (e.g., a model consistently predicting `LOCATION` where the dataset has
`PERSON`) are surfaced as **cross-branch overlap insights** in the audit table (`render_html()`),
but are not a separate issue type. They fall under the existing `COLLISION_CROSS_BRANCH` category.

### Issue types and severity (after projection)

| Issue Type | Severity | Description | Default Action |
|---|---|---|---|
| **UNRESOLVED** | ERROR | Label could not be placed in the hierarchy at all | User must map or suppress |
| **COLLISION_AMBIGUOUS** | WARNING | Prediction entity maps to multiple eval-surface entities | User decides (ranked suggestions via token overlap) |
| **COLLISION_CROSS_BRANCH** | WARNING | Prediction and eval-surface entity are on entirely different branches | User decides |
| **PREDICTION_ONLY** | WARNING | Prediction entity exists but no eval-surface entity corresponds to it | User decides: suppress (`None`), remap, or keep-as-FP (`map to self`) |
| **COLLISION_TRIVIAL** | INFO | Annotation or prediction entity is an ancestor or descendant of an eval-surface entity on the same branch | Auto-fix (project to eval-surface entity); logged and visible in audit table |
| **DATASET_ONLY** | INFO | Entity exists in annotations but no prediction entity maps to it | Default: **keep** (FN counted, shows model gaps). User can suppress. |

Issues are sorted by: severity (ERROR > WARNING > INFO), then by affected token count (descending).

**COUNTRY_FALLBACK** (country prefix recognized but document type unknown → `NATIONAL_ID`) is
logged at `INFO` level but is **not** a separate issue type — it is part of the identification
phase and resolves to a hierarchy node.

### Blocking behavior

`get_mapped_results_dataframe()` raises `IncompleteMapping` if any **ERROR or WARNING** issues
remain unresolved. This means the user must explicitly handle:
- **UNRESOLVED** labels (map to an entity or suppress)
- **COLLISION_AMBIGUOUS** (pick a target entity)
- **COLLISION_CROSS_BRANCH** (map, suppress, or keep-as-FP)
- **PREDICTION_ONLY** entities (suppress, remap, or map to self to accept FP)

**INFO** issues (COLLISION_TRIVIAL, DATASET_ONLY) never block.

To resolve a PREDICTION_ONLY entity, call one of:
- `mapper.map({"EMAIL": None})` — suppress from evaluation
- `mapper.map({"EMAIL": "CONTACT"})` — remap to an eval-surface entity
- `mapper.map({"EMAIL": "EMAIL"})` — keep as-is (every prediction counts as FP)

### Workflow

```python
mapper = CanonicalMapper()
mapper.analyze(results_df)          # Phase 1: identify + Phase 2: project

# Trivial collisions are auto-fixed; WARNING+ issues need attention
mapper.render_html()                # Audit table — issues sorted by impact
for issue in mapper.get_issues():   # Most impactful first
    print(issue)

# Resolve remaining issues (all WARNING+ must be resolved before getting results)
mapper.map({"AMBIGUOUS_LABEL": "TARGET_ENTITY"})
mapper.map({"UNWANTED_LABEL": None})           # suppress
mapper.map({"KEEP_AS_FP": "KEEP_AS_FP"})       # keep (will count as FP)

# Get the mapped DataFrame (blocks if WARNING+ issues remain)
mapped_df = mapper.get_mapped_results_dataframe()
```

Comparing multiple models against the same dataset:

```python
mapper = CanonicalMapper()

# Model A — first analyze() discovers canonical surface and locks it
mapper.analyze(results_df_model_a)
mapper.map({...})                   # resolve Model A's issues
mapped_a = mapper.get_mapped_results_dataframe()

# Model B — same canonical surface (locked), different prediction issues
mapper.analyze(results_df_model_b)
mapper.map({...})                   # resolve Model B's issues
mapped_b = mapper.get_mapped_results_dataframe()

# Both mapped DataFrames use the same entity vocabulary
# → scores are directly comparable
```

### Eval-surface locking

The canonical surface (set of entities at the majority-vote depth) is **locked after the first
`analyze()` call**. Subsequent `analyze()` calls for different models reuse the same surface
to ensure cross-model comparability. Issues are per-model (cleared and recomputed on each
`analyze()` call).

To start fresh with a different dataset, create a new `CanonicalMapper` instance.

### Input

The input is the per-token comparison of predictions and actuals.
A user can get this by running the typical flow in presidio-evaluator,
or generate this in any other way.

Format:

| Column | Type | Description |
|---|---|---|
| `sentence_id` | `int` | Index of the source sentence in the dataset |
| `token` | `str` | Token string |
| `annotation` | `str` | Ground-truth entity tag (from `InputSample.tags`) |
| `prediction` | `str` | Model-predicted entity tag |

For mapping, only the `annotation` and `prediction` are used.
The mapper returns a new DataFrame with:
- `annotation` and `prediction` columns rewritten to eval-surface entities
  (suppressed labels become `"O"`).
- `original_annotation` and `original_prediction` columns preserving the raw labels.

### Logging

Every resolution decision is logged at `INFO` level, with country-prefix fallback at `WARNING`
and unresolvable labels at `WARNING`. Trivial auto-fixes are logged at `INFO` with the fix
applied. A summary line after analysis reports the auto-discovered depth, canonical surface entities,
resolution counts, auto-fixes applied, and number of issues requiring attention.

Logger name: `presidio_evaluator.entity_mapping`.

### HTML audit table

`render_html()` shows two sections:
1. **Summary bar** — auto-discovered depth, counts per issue type, total auto-fixes applied.
2. **Detail table** — one row per raw label, sorted by severity then token count (most
   impactful first). Badge colours: COLLISION_TRIVIAL = blue, exact match = green,
   UNRESOLVED = red, PREDICTION_ONLY = grey, DATASET_ONLY = amber.

For cross-branch overlaps with high token counts, the audit table surfaces systematic
misprediction insights (e.g., “LOCATION predicted on 200 PERSON tokens”) to help users
understand model weaknesses.

Uses `IPython.display.HTML` in Jupyter; falls back to plain text. Never raises.

### Interactive resolution

`resolve_interactively(prompt_fn=input)` prompts only for issues requiring user action
(WARNING+ severity). COLLISION_TRIVIAL and DATASET_ONLY (INFO) are skipped.

For each issue, it shows:
- The issue type and affected token count.
- Ranked suggestions (using token overlap for ambiguous collisions).
- A prompt accepting: a suggestion number, a free-text entity name, or `NONE` to suppress.

The `prompt_fn` parameter allows injection for testing.

## Consequences

### Positive

- **Dataset-anchored evaluation** — multiple models can be compared against the same dataset
  without each needing its own mapping configuration. The dataset is the contract.
- **Auto-discovered depth** — users no longer need to guess the right granularity; the
  mapper infers it from the data via weighted majority vote.
- **Eval-surface locking** — the first `analyze()` call locks the canonical surface, ensuring
  all subsequent models are compared against the same entity vocabulary.
- **Both sides projected** — annotations and predictions are both projected to the same
  depth, eliminating granularity mismatches without manual intervention.
- **Frequency-driven triage** — the most impactful mapping problems surface first, reducing time
  to a working evaluation.
- **Token-overlap suggestions** — ambiguous collisions are ranked by actual per-row overlap,
  giving data-driven recommendations instead of alphabetical guesses.
- **Trivial auto-fix** — ancestor/descendant mismatches within the same branch are resolved
  automatically (INFO level), drastically reducing manual `map()` calls.
- **Explicit blocking** — `get_mapped_results_dataframe()` blocks on WARNING+ issues,
  preventing silent evaluation errors.
- **Single resolution entry point** — all mapping logic lives in one place.
- **Transparency** — every resolution decision is logged and visible in the HTML audit table.
- **Composable** — `get_mapping()` returns a plain `dict[str, str | None]` that can be passed
  to any downstream evaluation code, serialised, or version-controlled.
- **Atomic batch updates** — `map()` validates all entries before applying any.
- **Original labels preserved** — output DataFrame includes `original_annotation` and
  `original_prediction` columns for traceability.

### Negative / Trade-offs

- **Dataset must be present** — the mapper requires a results DataFrame with both annotations
  and predictions to determine the evaluation surface. There is no label-only or
  depth-override mode; the mapper is purely data-driven.
- **Context-dependent mapping** — the same prediction label may map to different targets depending
  on the dataset. This is intentional (evaluation is relative to a dataset) but may surprise users
  who expect a static label→canonical dict.
- **Auto-fix may be wrong** — trivially collapsing `FIRST_NAME` → `NAME` is usually correct, but
  in a dataset that intentionally distinguishes them, the auto-fix would be harmful. The
  COLLISION_TRIVIAL flag + audit table mitigate this.
- **`IncompleteMapping` is a hard stop** — pipelines that previously silently dropped unknown
  labels will now fail explicitly until all WARNING+ issues are resolved. This is intentional
  but requires pipeline changes for existing users.
- **PREDICTION_ONLY requires action** — unlike the previous design which silently suppressed
  prediction-only entities, this design forces the user to explicitly decide (suppress, remap,
  or keep-as-FP). This adds friction but ensures the user understands the evaluation surface.
- **Canonical surface locked after first call** — switching datasets requires creating a new
  `CanonicalMapper` instance. This is by design (ensures comparability) but may surprise users
  who expect `analyze()` to always recompute the surface.

## Alternatives Considered

### 1. Symmetric mapping to fixed canonical depth (previous design)

Both annotation and prediction labels are mapped independently to the same canonical depth
(default 3). Rejected because real-world datasets use mixed granularity, and depth mismatch
between models and datasets is the norm. A global depth parameter cannot represent a dataset that
mixes `PERSON` (depth 2) with `STREET_ADDRESS` (depth 3).

### 2. Meet-in-the-Middle (LCA-based per-pair convergence)

For each (annotation, prediction) pair, find their Lowest Common Ancestor in the hierarchy and
map both sides to that level. Rejected because different pairs converge at different depths,
making results hard to interpret. It also modifies the dataset's annotations — undermining the
principle that the dataset is ground truth.

### 3. Evaluation-Goal-Driven (user specifies entity set upfront)

The user declares `eval_entities=["PERSON", "LOCATION", ...]` and everything is projected onto
that set. Rejected entirely (not even as an override) because it requires upfront knowledge of
the right granularity, which most users don't have when starting an evaluation. The purely
data-driven approach (majority-vote depth) makes this unnecessary.

### 4. Score against own labels (no mapping)

Each model is evaluated only on the entity types it natively supports, with no mapping at all.
Makes cross-model comparison impossible.

### 5. Manual mapping only

Require users to supply a complete `dict[str, str]` upfront. Too burdensome for large
vocabularies where most mappings are straightforward.

### 6. Semantic similarity (embedding-based matching)

Use a sentence-transformer model to embed labels and pick nearest neighbours. Rejected due to
heavy ML dependency for a finite label vocabulary, and non-deterministic results across model
versions.

### 7. Embed mapping inside the model prediction step

Rejected because mapping is a property of the evaluation goal, not the model. Keeping mapping
separate allows the same model output to be evaluated under different configurations.

## Example mappings

Mapping of raw entity labels (as found in HuggingFace NER models and datasets) to canonical entities in the `EntityHierarchy`.

| Raw Label | Canonical Entity |
|-----------|-----------------|
| `ACCOUNT` | `FINANCIAL` |
| `ACCOUNTNAME` | `USERNAME` |
| `ACCOUNTNUM` | `FINANCIAL` |
| `ACCOUNTNUMBER` | `FINANCIAL` |
| `ACCOUNT_NUMBER` | `FINANCIAL` |
| `ADDRESS` | `ADDRESS` |
| `AGE` | `AGE` |
| `AMOUNT` | `FINANCIAL` |
| `BANKACCOUNT` | `FINANCIAL` |
| `BIC` | `FINANCIAL` |
| `BIOID` | `FACE` |
| `BIOMETRIC` | `BIOMETRIC` |
| `BIRTHDAY` | `BIRTH_DATE` |
| `BITCOINADDRESS` | `FINANCIAL` |
| `BOD` | `BIRTH_DATE` |
| `BUILDING` | `BUILDING` |
| `BUILDINGNUM` | `ADDRESS` |
| `BUILDINGNUMBER` | `ADDRESS` |
| `BUILDING_NUMBER` | `ADDRESS` |
| `CARDISSUER` | `FINANCIAL` |
| `CITY` | `ADDRESS` |
| `CNPJ` | `BUSINESS_ID` |
| `COMPANYNAME` | `COMPANY` |
| `COMPANY_NAME` | `COMPANY` |
| `CONDITION` | `HEALTH_CONDITION` |
| `CONT` | `ADDRESS` |
| `COORDINATE` | `GEO_COORDINATES` |
| `COUNTRY` | `ADDRESS` |
| `COUNTY` | `ADDRESS` |
| `CRD` | `FINANCIAL` |
| `CREDITCARD` | `FINANCIAL` |
| `CREDITCARDCVV` | `FINANCIAL` |
| `CREDITCARDISSUER` | `FINANCIAL` |
| `CREDITCARDNUMBER` | `FINANCIAL` |
| `CREDIT_CARD` | `FINANCIAL` |
| `CREDIT_CARD_NUMBER` | `FINANCIAL` |
| `CURRENCY` | `FINANCIAL` |
| `CURRENCYCODE` | `FINANCIAL` |
| `CURRENCYNAME` | `FINANCIAL` |
| `CURRENCYSYMBOL` | `FINANCIAL` |
| `CVV` | `FINANCIAL` |
| `DATE` | `DATE` |
| `DATEOFBIRTH` | `BIRTH_DATE` |
| `DATE_OF_BIRTH` | `BIRTH_DATE` |
| `DATE_TIME` | `DATE_TIME` |
| `DEVICE` | `DEVICE_ID` |
| `DLN` | `DRIVER_LICENSE` |
| `DOB` | `BIRTH_DATE` |
| `DOCTOR` | `NAME` |
| `DOCTOR_NAME` | `NAME` |
| `DRIVERLICENSE` | `DRIVER_LICENSE` |
| `DRIVERLICENSENUM` | `DRIVER_LICENSE` |
| `DRIVER_LICENSE_NUMBER` | `DRIVER_LICENSE` |
| `EMA` | `EMAIL_ADDRESS` |
| `EMAIL` | `EMAIL_ADDRESS` |
| `EMAIL_ADDRESS` | `EMAIL_ADDRESS` |
| `ETHEREUMADDRESS` | `FINANCIAL` |
| `EYECOLOR` | `PHYSICAL_DESCRIPTOR` |
| `FACILITY` | `FACILITY` |
| `FAX` | `FAX` |
| `FINANCIAL` | `FINANCIAL` |
| `FIRSTNAME` | `NAME` |
| `FIRST_NAME` | `NAME` |
| `FULLNAME` | `NAME` |
| `GENDER` | `GENDER` |
| `GEOCOORD` | `GEO_COORDINATES` |
| `GIVENNAME` | `NAME` |
| `GIVENNAME1` | `NAME` |
| `GIVENNAME2` | `NAME` |
| `GPSCOORDINATES` | `GEO_COORDINATES` |
| `HEALTHPLAN` | `HEALTH_INSURANCE_ID` |
| `HEALTH_PLAN` | `HEALTH_INSURANCE_ID` |
| `HEIGHT` | `PHYSICAL_DESCRIPTOR` |
| `HOSPITAL` | `MEDICAL_FACILITY` |
| `HOSPITAL_NAME` | `MEDICAL_FACILITY` |
| `IBAN` | `FINANCIAL` |
| `IBAN_CODE` | `FINANCIAL` |
| `ID` | `ID` |
| `IDCARD` | `NATIONAL_ID` |
| `IDCARDNUM` | `NATIONAL_ID` |
| `IDNUM` | `NATIONAL_ID` |
| `ID_CARD_NUMBER` | `FINANCIAL` |
| `IMEI` | `IMEI` |
| `IPADDRESS` | `IP_ADDRESS` |
| `IPV4` | `IP_ADDRESS` |
| `IPV6` | `IP_ADDRESS` |
| `IP_ADDRESS` | `IP_ADDRESS` |
| `JOBAREA` | `JOB_DEPARTMENT` |
| `JOBDEPARTMENT` | `JOB_DEPARTMENT` |
| `JOBDESCRIPTOR` | `JOB_DESCRIPTOR` |
| `JOBTITLE` | `JOB_TITLE` |
| `JOBTYPE` | `JOB_DESCRIPTOR` |
| `LASTNAME` | `NAME` |
| `LASTNAME1` | `NAME` |
| `LASTNAME2` | `NAME` |
| `LASTNAME3` | `NAME` |
| `LAST_NAME` | `NAME` |
| `LICENSE` | `PROFESSIONAL_LICENSE` |
| `LICENSEPLATENUM` | `LICENSE_PLATE_NUMBER` |
| `LICENSE_PLATE` | `LICENSE_PLATE` |
| `LITECOINADDRESS` | `FINANCIAL` |
| `LOC` | `LOC` |
| `LOCATION` | `LOCATION` |
| `LOCATION-OTHER` | `LOCATION_OTHER` |
| `MAC` | `MAC_ADDRESS` |
| `MACADDRESS` | `MAC_ADDRESS` |
| `MAC_ADDRESS` | `MAC_ADDRESS` |
| `MASKEDNUMBER` | `FINANCIAL` |
| `MEDICALRECORD` | `PATIENT_ID` |
| `MIDDLENAME` | `NAME` |
| `MISC` | `MISCELLANEOUS` |
| `MRN` | `MRN` |
| `NAME` | `NAME` |
| `NATIONALID` | `NATIONAL_ID` |
| `NEARBYGPSCOORDINATE` | `GEO_COORDINATES` |
| `NO_RESPONSE` | `NATIONAL_ID` |
| `NRP` | `NATIONALITY` |
| `NUMBER` | `NATIONAL_ID` |
| `OCCUPATION` | `JOB_TITLE` |
| `ORDINALDIRECTION` | `LOCATION_OTHER` |
| `ORG` | `ORG` |
| `ORGANIZATION` | `ORGANIZATION` |
| `OTHER_NAME` | `ALIAS` |
| `PASS` | `PASSWORD` |
| `PASSPORT` | `PASSPORT` |
| `PASSPORTID` | `PASSPORT` |
| `PASSPORTNUM` | `PASSPORT` |
| `PASSWORD` | `PASSWORD` |
| `PATIENT` | `PATIENT_ID` |
| `PATIENT_ID` | `PATIENT_ID` |
| `PATIENT_NAME` | `NAME` |
| `PER` | `NAME` |
| `PERSON` | `PERSON` |
| `PHN` | `PHONE_NUMBER` |
| `PHONE` | `PHONE_NUMBER` |
| `PHONEIMEI` | `IMEI` |
| `PHONENUMBER` | `PHONE_NUMBER` |
| `PHONE_NUMBER` | `PHONE_NUMBER` |
| `PHOTO` | `FACE` |
| `PIN` | `PIN` |
| `POSTCODE` | `ADDRESS` |
| `PREFIX` | `PREFIX` |
| `PROFESSION` | `JOB_TITLE` |
| `PROVIDER` | `JOB_TITLE` |
| `PSP` | `PASSPORT` |
| `PWD` | `PASSWORD` |
| `ROUTING_NUMBER` | `FINANCIAL` |
| `RRN` | `NATIONAL_ID` |
| `SECADDRESS` | `ADDRESS` |
| `SECONDARYADDRESS` | `ADDRESS` |
| `SECURITYTOKEN` | `TOKEN` |
| `SEX` | `GENDER` |
| `SOCIALNUM` | `SSN` |
| `SOCIALNUMBER` | `SSN` |
| `SSN` | `SSN` |
| `STATE` | `ADDRESS` |
| `STREET` | `ADDRESS` |
| `STREETADDRESS` | `ADDRESS` |
| `SUFFIX` | `SUFFIX` |
| `SURNAME` | `NAME` |
| `SWIFT_CODE` | `FINANCIAL` |
| `TAXNUM` | `TAX_ID` |
| `TAX_NUMBER` | `NATIONAL_ID` |
| `TEL` | `PHONE_NUMBER` |
| `TELEPHONENUM` | `PHONE_NUMBER` |
| `TIME` | `TIME` |
| `TITLE` | `TITLE` |
| `URL` | `URL` |
| `USERAGENT` | `USER_AGENT` |
| `USERNAME` | `USERNAME` |
| `USER` | `USERNAME` |
| `US_BANK_NUMBER` | `NATIONAL_ID` |
| `US_DRIVER_LICENSE` | `DRIVER_LICENSE` |
| `US_ITIN` | `TAX_ID` |
| `US_LICENSE_PLATE` | `LICENSE_PLATE` |
| `US_PASSPORT` | `PASSPORT` |
| `US_SSN` | `SSN` |
| `VEHICLE` | `VEHICLE_ID` |
| `VEHICLEVIN` | `VIN` |
| `VEHICLEVRM` | `VEHICLE_ID` |
| `VIN` | `VIN` |
| `VRM` | `LICENSE_PLATE` |
| `ZIP` | `ADDRESS` |
| `ZIPCODE` | `ADDRESS` |
| `account_number` | `FINANCIAL` |
| `account_pin` | `FINANCIAL` |
| `api_key` | `API_KEY` |
| `audio_duration_range` | `NATIONAL_ID` |
| `audio_longer_than` | `NATIONAL_ID` |
| `audio_min_duration` | `DURATION` |
| `bank_routing_number` | `FINANCIAL` |
| `biometric_identifier` | `FACE` |
| `bitcoin_address` | `FINANCIAL` |
| `blood_type` | `BLOOD_TYPE` |
| `certificate_license_number` | `PROFESSIONAL_LICENSE` |
| `company_name` | `COMPANY` |
| `credit_card` | `FINANCIAL` |
| `credit_card_number` | `FINANCIAL` |
| `credit_card_security_code` | `FINANCIAL` |
| `credit_debit_card` | `FINANCIAL` |
| `customer_id` | `CUSTOMER_ID` |
| `date_of_birth` | `BIRTH_DATE` |
| `date_time` | `DATE_TIME` |
| `device_identifier` | `DEVICE_ID` |
| `driver_license_number` | `DRIVER_LICENSE` |
| `education_level` | `EDUCATION_LEVEL` |
| `employee_id` | `EMPLOYEE_ID` |
| `employment_status` | `EMPLOYMENT_STATUS` |
| `fax_number` | `FAX` |
| `first_name` | `NAME` |
| `health_plan_beneficiary_number` | `HEALTH_INSURANCE_ID` |
| `http_cookie` | `HTTP_COOKIE` |
| `ip_address` | `IP_ADDRESS` |
| `last_name` | `NAME` |
| `license_plate` | `LICENSE_PLATE` |
| `mac_address` | `MAC_ADDRESS` |
| `medical_record_number` | `MRN` |
| `phone_number` | `PHONE_NUMBER` |
| `political_view` | `POLITICAL_AFFILIATION` |
| `race_ethnicity` | `ETHNICITY` |
| `religious_belief` | `RELIGION` |
| `street_address` | `ADDRESS` |
| `swift_bic` | `FINANCIAL` |
| `swift_bic_code` | `FINANCIAL` |
| `tax_id` | `TAX_ID` |
| `unique_id` | `CUSTOMER_ID` |
| `user_name` | `USERNAME` |
| `vehicle_identifier` | `LICENSE_PLATE` |
| `AU_TAX_ID` | `TAX_ID` |
| `DE_TAX_ID` | `TAX_ID` |
| `CA_DRIVER_LICENSE` | `DRIVER_LICENSE` |
| `IN_DRIVER_LICENSE` | `DRIVER_LICENSE` |
| `UK_SSN` | `SSN` |
| `SG_SSN` | `SSN` |
| `FR_UNKNOWN_ENTITY` | `NATIONAL_ID` |
| `ES_UNKNOWN_ENTITY` | `NATIONAL_ID` |
| `AUSTRIA_PASSPORT_NUMBER` | `PASSPORT` |
| `HAITI_TAX_ID` | `TAX_ID` |
| `GERMANY_AAABBB` | `NATIONAL_ID` |
| `JAPAN_VEHICLE_NUMBER` | `NATIONAL_ID` |
| `NIGERIAN_NATIONAL_ID` | `NATIONAL_ID` |
| `FRENCH_PASSPORT` | `PASSPORT` |

## Proposed hierarchical entity mapping dictionary
All entities are mapped to the 3rd level (canonical) by default when the majority-vote depth computes to 3.
- The 2nd level: `PERSON, DEMOGRAPHIC, CONTACT, LOCATION, ORGANIZATION, EMPLOYMENT, GOVERNMENT_ID, FINANCIAL_PII, DEVICE_IDENTIFIER, BIOMETRIC, NETWORK_IDENTIFIER, AUTHENTICATION, PHI, VEHICLE_PII, LEGAL_PII, TRAVEL_PII, EDUCATION, DATE_TIME`.
- The 3rd level: `NAME, ..., TITLE, USERNAME, ..., AGE, GENDER, ..., ADDRESS, ..., COMPANY, SSN, PASSPORT, TAX_ID, NATIONAL_ID, FINANCIAL, DEVICE_ID,...`

The evaluation depth is determined automatically from the dataset's annotations via weighted majority vote.

```py
HIERARCHY: dict = {
    "PII": {
        "PERSON": {
            "NAME": {
                "FIRST_NAME": [
                    "FIRSTNAME",
                    "NAME_GIVEN",
                    "GIVENNAME",
                    "GIVENNAME1",
                    "GIVENNAME2",
                ],
                "MIDDLE_NAME": ["MIDDLENAME"],
                "LAST_NAME": [
                    "LASTNAME",
                    "LASTNAME1",
                    "LASTNAME2",
                    "LASTNAME3",
                    "SURNAME",
                    "NAME_FAMILY",
                ],
                "FULL_NAME": [
                    "FULLNAME",
                    "DOCTOR",
                    "PATIENT_NAME",
                    "DOCTOR_NAME",
                    "HCW",
                    "NAME_MEDICAL_PROFESSIONAL",
                ],
                "MAIDEN_NAME": [],
                "PER": [],
            },
            "PREFIX": [],
            "SUFFIX": [],
            "TITLE": [],
            "USERNAME": ["USER_NAME", "DISPLAYNAME", "ACCOUNTNAME", "ACCOUNT_NAME"],
            "ALIAS": ["OTHER_NAME"],
        },
        "DEMOGRAPHIC": {
            "AGE": ["AGE_GROUP", "AGE_RANGE", "AGE_IN_YEARS"],
            "GENDER": ["SEX", "SEXTYPE"],
            "SEXUAL_ORIENTATION": ["SEXUALITY"],
            "RELIGION": ["RELIGIOUS_BELIEF", "BELIEF"],
            "ETHNICITY": ["RACE_ETHNICITY", "ORIGIN", "RACE"],
            "NATIONALITY": ["NRP", "NORP"],
            "MARITAL_STATUS": [],
            "LANGUAGE": [],
            "POLITICAL_AFFILIATION": ["POLITICAL_VIEW"],
            "ZODIAC_SIGN": [],
            "DEMOGRAPHIC_ATTRIBUTE": ["DEM"],
            "PHYSICAL_DESCRIPTOR": {
                "PHYSICAL_ATTRIBUTE": [],
                "SKIN_COLOR": ["SKIN_TONE", "COMPLEXION"],
                "EYE_COLOR": ["EYECOLOR"],
                "HAIR_COLOR": ["HAIRCOLOR"],
                "HEIGHT": [],
                "WEIGHT": [],
                "BODY_MEASUREMENT": ["BODY_MEASURE", "MEASUREMENTS"],
            },
        },
        "CONTACT": {
            "EMAIL_ADDRESS": ["EMAIL", "EMA"],
            "PHONE_NUMBER": ["PHONE", "TEL", "TELEPHONENUM", "PHONENUMBER", "PHN", "MOBILE"],
            "FAX": ["FAX_NUMBER"],
            "SOCIAL_HANDLE": ["QQ"],  # QQ: Chinese messaging platform ID
        },
        "LOCATION": {
            "ADDRESS": {
                "STREET_ADDRESS": [
                    "STREET",
                    "STREETADDRESS",
                    "LOCATION_ADDRESS",
                    "LOCATION_ADDRESS_STREET",
                    "ADDRESS",
                ],
                "BUILDING_NUMBER": ["BUILDINGNUMBER", "BUILDINGNUM"],
                "SECONDARY_ADDRESS": ["SECONDARYADDRESS", "SECADDRESS"],
                "CITY": ["LOCATION_CITY"],
                "COUNTY": ["PROVINCE"],
                "STATE": ["LOCATION_STATE"],
                "POSTAL_CODE": [
                    "ZIPCODE",
                    "ZIP",
                    "POSTCODE",
                    "LOCATION_ZIP",
                    "UK_POSTCODE",
                    "CEP",       # BR Código de Endereçamento Postal
                    "CEP_CODE",  # common compound form e.g. BRAZIL_CEP_CODE
                    "PLZ",       # DE Postleitzahl
                ],
                "COUNTRY": ["LOCATION_COUNTRY", "COUNTRY_OR_REGION"],
            },
            "BUILDING": [],
            "FACILITY": [],
            "GEO_COORDINATES": [
                "GPSCOORDINATES",
                "GPS_COORDINATES",
                "COORDINATE",
                "LOCATION_COORDINATE",
                "LATITUDE_LONGITUDE",
                "NEARBYGPSCOORDINATE",
                "GEOCOORD",
                "LAT",
                "LONG",
                "LATITUDE",
                "LONGITUDE",
            ],
            "LOCATION_OTHER": ["LOCATION-OTHER", "ORDINALDIRECTION"],
            "GPE": ["GLOBAL_POLITICAL_ENTITY"],
            "LOC": [],
            "GEO": [],
        },
        "ORGANIZATION": {
            "COMPANY": [
                "COMPANYNAME",
                "COMPANY_ID",
                "COMPANY_NAME",
                "CORPORATION",
                "VENDOR",
            ],
            "GOVERNMENT_AGENCY": ["GOVERNMENT"],
            "SCHOOL": ["SCHOOL_ID"],
            "MEDICAL_FACILITY": [
                "ORGANIZATION_MEDICAL_FACILITY",
                "HOSPITAL",
                "HOSPITAL_NAME",
            ],
            "OTHER_ORG": [],
            "ORG": [],
        },
        "EMPLOYMENT": {
            "JOB_TITLE": ["JOBTITLE", "OCCUPATION", "PROFESSION", "PROVIDER", "POSITION"],
            "JOB_DEPARTMENT": ["JOBDEPARTMENT", "JOBAREA"],
            "JOB_DESCRIPTOR": ["JOBDESCRIPTOR", "JOBTYPE"],
            "EMPLOYEE_ID": ["EMPLOYEE"],
            "CUSTOMER_ID": ["CUSTOMER", "UNIQUE", "UNIQUE_ID"],
            "EMPLOYMENT_STATUS": [],
            "LICENSE": [],
        },
        "GOVERNMENT_ID": {
            "SSN": [
                "SOCIALNUMBER",
                "SOCIALNUM",
                "SOCIAL_SECURITY",
                "SOCIAL_SECURITY_NUMBER",
                "US_SSN",
                "UK_NINO",
                # Common generic forms used as standalone labels
                "SOCIAL_INSURANCE",
                "NATIONAL_INSURANCE",
                "INSURANCE_NUMBER",
            ],
            "PASSPORT": [
                "PSP",
                "PASSPORT_NUMBER",
                "PASSPORT_ID",
                "US_PASSPORT",
                "UK_PASSPORT",
            ],
            "DRIVER_LICENSE": [
                "DRIVERLICENSE",
                "DRIVERLICENSENUM",
                "DLN",
                "DRIVERS_LICENSE",
                "DRIVER_LICENSE_ID",
                "US_DRIVER_LICENSE",
                "IT_DRIVER_LICENSE",
                "DRIVER",  # generic suffix keyword (e.g. GERMANY_DRIVER_LICENSE)
            ],
            "TAX_ID": [
                "TAXNUM",
                "US_ITIN",
                "AU_TFN",
                "IN_PAN",
                "IN_GSTIN",
                "ES_NIE",
                "ES_NIF",
                # Common country-specific tax codes used as standalone labels
                "CPF",   # BR Cadastro de Pessoas Físicas
                "RFC",   # MX Registro Federal de Contribuyentes
                "RUT",   # CL Rol Único Tributario
                "NIT",   # CO Número de Identificación Tributaria
            ],
            "NATIONAL_ID": [
                "IDCARD",
                "IDCARDNUM",
                "ID_NUM",
                "IDNUM",
                "IT_FISCAL_CODE",
                "IT_IDENTITY_CARD",
                "IN_AADHAAR",
                "PL_PESEL",
                "SG_NRIC_FIN",
                "FI_PERSONAL_IDENTITY_CODE",
                "KR_RRN",
                "KR_FRN",
                "KR_BRN",
                "NG_NIN",
                "TH_TNIN",
                # Common country-specific ID codes used as standalone labels
                "AADHAAR",                # IN Aadhaar card
                "DNI",                    # ES/AR/PE Documento Nacional de Identidad
                "CITIZENSHIP_CARD",       # CO Cédula de Ciudadanía
                "CURP",                   # MX Clave Única de Registro de Población
                "RG_NUMBER",              # BR Registro Geral
                "RUN",                    # CL Rol Único Nacional
                "NATIONAL_IDENTIFICATION",  # generic suffix (e.g. FRANCE_NATIONAL_IDENTIFICATION_NUMBER)
                "RRN",                    # KR standalone Resident Registration Number
            ],
            "VOTER_ID": ["VOTER", "IN_VOTER", "UK_ELECTORAL_ROLL_NUMBER", "ELECTORAL"],
            "IMMIGRATION_ID": ["IMMIGRATION"],
            "PROFESSIONAL_LICENSE": [
                "LICENSE",
                "CERTIFICATE_LICENSE_NUMBER",
                "PROFESSIONAL_LICENSE_ID",
                "IT_VAT_CODE",
            ],
            "BUSINESS_ID": [
                "BUSINESS", "SG_UEN", "AU_ABN", "AU_ACN",
                "HANDELSREGISTER",   # DE Handelsregisternummer
                "REGISTRO_MERCANTIL",  # ES/CO Registro Mercantil
                "CNPJ",              # BR Cadastro Nacional da Pessoa Jurídica
            ],
            "PUBLIC_TRANSPORT_CARD": [],
            "ID": ["NUMERIC_PII", "CODE"],  # CODE: generic coded identifier
            "VIN": ["VIN_ID", "VEHICLE_IDENTIFICATION_NUMBER"],
            "LICENSE_PLATE_NUMBER": [
                "LICENSE_PLATE_ID",
                "VEHICLE_REGISTRATION_NUMBER",
                "VRN",
                "LICENSE_PLATE",
                "KFZ",          # DE Kraftfahrzeugkennzeichen
                "KENNZEICHEN",  # DE Kennzeichen
            ],
        },
        "FINANCIAL_PII": {
            "FINANCIAL": {
                "CREDIT_CARD": {
                    "CARD_NUMBER": [
                        "CREDITCARD",
                        "CREDIT_CARD",
                        "CREDITCARDNUMBER",
                        "CREDIT_DEBIT_CARD",
                        "CRD",
                    ],
                    "CVV": ["CREDITCARDCVV", "CREDIT_CARD_SECURITY_CODE"],
                    "EXPIRATION": ["CREDIT_CARD_EXPIRATION"],
                    "CARD_ISSUER": ["CREDITCARDISSUER", "CARDISSUER"],
                    "MASKED_NUMBER": ["MASKEDNUMBER"],
                },
                "BANK_ACCOUNT": {
                    "ACCOUNT_NUMBER": [
                        "ACCOUNT",
                        "BANKACCOUNT",
                        "BANK_ACCOUNT",
                        "ACCOUNTNUMBER",
                        "ACCOUNTNUM",
                        "ACC",
                        "BBAN",
                    ],
                    "IBAN": ["IBAN_CODE"],
                    "SWIFT_BIC": ["BIC", "SWIFT_CODE"],
                    "ROUTING_NUMBER": ["BANK_ROUTING_NUMBER"],
                },
                "CRYPTO_WALLET": {
                    "BITCOIN": ["BITCOINADDRESS", "CRYPTO"],
                    "ETHEREUM": ["ETHEREUMADDRESS"],
                    "LITECOIN": ["LITECOINADDRESS"],
                },
                "FINANCIAL_AMOUNT": {
                    "CURRENCY": ["CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL"],
                    "AMOUNT": ["MONEY"],
                },
                "INSURANCE": {
                    "POLICY_NUMBER":  ["INSURANCE_POLICY", "POLICY_ID"],
                    "CLAIM_NUMBER":   ["CLAIM_ID", "INSURANCE_CLAIM"],
                    "POLICY_HOLDER": [],
                },
            },
        },
        "DEVICE_IDENTIFIER": {
            "DEVICE_ID": ["DEVICE", "DEVICE_IDENTIFIER"],
            "SERIAL_NUMBER": [],
            "IMEI": ["PHONEIMEI"],
            "IMSI": ["SUBSCRIBER_IDENTITY", "MOBILE_SUBSCRIBER_ID"],
            "ICCID": ["SIM_CARD_NUMBER", "SIM_ID"],
            "MAC_ADDRESS": ["MACADDRESS", "MAC"],
            "ADVERTISING_ID": ["ADVERTISING"],
            "USER_AGENT": ["USERAGENT"],
            "FILE_PATH": ["FILENAME"],  # file/path reference that may contain PII
        },
        "BIOMETRIC": {
            "FINGERPRINT": [
                "FINGERPRINT_ID",
                "FINGERPRINT_DATA",
                "FINGERPRINT_TEMPLATE",
            ],
            "FACE": [
                "FACE_ID",
                "FACE_RECOGNITION",
                "FACIAL_SCAN",
                "FACE_TEMPLATE",
                "BIOID",
                "BIOMETRIC_IDENTIFIER",
                "PHOTO",
                "FACIAL_IMAGE",
                "FACE_IMAGE",
            ],
            "IRIS": ["IRIS_SCAN", "IRIS_TEMPLATE"],
            "RETINA": ["RETINA_SCAN"],
            "VOICE_PRINT": ["VOICE_RECOGNITION", "VOICE_TEMPLATE", "VOICEPRINT"],
            "DNA": ["DNA_SEQUENCE", "GENETIC_DATA"],
            "PALM_PRINT": ["PALM_TEMPLATE", "PALM_VEIN"],
        },
        "NETWORK_IDENTIFIER": {
            "IP_ADDRESS": ["IPADDRESS", "IP", "IPV4", "IPV6"],
            "URL": ["URI", "HYPERLINK"],
            "DOMAIN": ["DOMAINNAME", "DOMAIN_NAME"],
            "WEBSITE": ["WEB", "WEBPAGE", "WEBADDRESS"],
            "HTTP_COOKIE": ["COOKIE_ID", "HTTP_COOKIE_ID"],
            "CONNECTION_STRING": [],
        },
        "AUTHENTICATION": {
            "PASSWORD": ["PASS", "PWD"],
            "PIN": [],
            "API_KEY": [],
            "PRIVATE_KEY": [],
            "TOKEN": ["SECURITYTOKEN"],
        },
        "PHI": {
            "PATIENT_ID": [
                "PATIENT",
                "MEDICALRECORD",
                "MEDICAL_RECORD_NUMBER",
                "MEDICAL_RECORD",
            ],
            "HEALTH_INSURANCE_ID": [
                "HEALTH_INSURANCE",
                "HEALTH_PLAN",
                "HEALTHPLAN",
                "HEALTHCARE_NUMBER",
                "HEALTH_PLAN_BENEFICIARY_NUMBER",
                "UK_NHS",
                "AU_MEDICARE",
                "KVNR",              # DE Krankenversicherungsnummer
                "KRANKENVERSICHERUNG",  # DE full word form
            ],
            "MEDICAL_LICENSE": ["MEDICAL_LICENSE_ID", "US_NPI", "US_MBI"],
            "HEALTH_CONDITION": [
                "CONDITION",
                "MEDICAL_DISEASE_DISORDER",
                "MEDICAL_BIOLOGICAL_ATTRIBUTE",
                "MEDICAL_BIOLOGICAL_STRUCTURE",
            ],
            "MEDICATION": ["DRUG", "DOSE", "MEDICAL_MEDICATION"],
            "PROCEDURE": [
                "MEDICAL_PROCESS",
                "MEDICAL_THERAPEUTIC_PROCEDURE",
                "MEDICAL_CLINICAL_EVENT",
            ],
            "INJURY": [],
            "BLOOD_TYPE": [],
            "FAMILY_HISTORY": ["MEDICAL_FAMILY_HISTORY", "MEDICAL_HISTORY"],
            "STATISTICS": [],
            "MRN": ["MEDICAL_RECORD_NUMBER"],
            "PLAN": ["HEALTHCARE_PLAN"],
            "PROTECTED_HEALTH_INFORMATION": [],
            # Clinical research / trials
            "STUDY_PARTICIPANT_ID": ["SUBJECT_ID", "TRIAL_PARTICIPANT_ID", "PARTICIPANT_ID"],
            "PROTOCOL_ID":          ["IRB_NUMBER", "PROTOCOL_NUMBER", "STUDY_ID"],
            "COHORT_ID":            ["COHORT", "ARM_ID"],
        },
        "VEHICLE_PII": {
            "LICENSE_PLATE": [
                "VRM",
                "VEHICLE_IDENTIFIER",
                "VEHICLE_REGISTRATION",
                "VRN",
            ],
            "VIN": ["VEHICLEVIN", "VEHICLE_IDENTIFICATION_NUMBER"],
            "VEHICLE_ID": ["VEHICLEVRM"],
            "CAR_TYPE": ["VEHICLE_TYPE", "VEHICLE_MAKE", "MAKE_MODEL"],
        },
        "LEGAL_PII": {
            "CASE_NUMBER": [],
            "COURT_RECORD": [],
            "ARREST_RECORD": [],
            "INMATE_ID": ["INMATE"],
            "MISCELLANEOUS": ["MISC"],  # catch-all for unclassified NER labels
        },
        "TRAVEL_PII": {
            "PNR": ["PASSENGER_NAME_RECORD", "PNR_NUMBER"],
            "ETIX": ["ELECTRONIC_TICKET", "ETICKET"],
            "WTN": ["WTN_NUMBER", "WORLD_TRACER_NUMBER"],
        },
        "EDUCATION": {
            "STUDENT_ID":       ["STUDENT_NUMBER", "LEARNER_ID", "ENROLLMENT_NUMBER", "STUDENT"],
            "ACADEMIC_RECORD":  ["TRANSCRIPT", "GRADE_REPORT", "GPA"],
            "EDUCATION_LEVEL":  [],
            "INSTITUTION_ID":   ["SCHOOL_CODE", "UNIVERSITY_ID"],
            "PARENT_GUARDIAN_ID": [],
            "PARENT": [],
            "TEACHER_ID": ["TEACHER_NUMBER", "FACULTY_ID", "TEACHER", "FACULTY"],

        },
        "DATE_TIME": {
            "DATE": [],
            "TIME": [],
            "EPOCH": [],
            "BIRTH_DATE": ["DATEOFBIRTH", "DATE_OF_BIRTH", "DOB", "BIRTHDAY", "BOD"],
            "DEATH_DATE": [],
            "DATE_INTERVAL": [],
            "DURATION": [],
            "EVENT": [],
            "DATES": [],
        },
    }
}
```

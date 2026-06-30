# Entity Mapping — Brainstorming Scenarios

A catalogue of real and realistic mapping challenges to stress-test the `CanonicalMapper` design.
Each scenario includes a concrete example; solutions are intentionally omitted.

---

## 1. Model Doesn't Support a Common Dataset Entity

The model lacks a recognizer for entity types that appear frequently in the dataset.

| Dataset entities | Model entities | What happens |
|---|---|---|
| `ORGANIZATION` ("Exversion", "Persint") | *(not supported)* | Every ORG annotation becomes a false negative |
| `AGE` ("25", "67 years old") | *(not supported)* | All AGE spans are missed |

**Real example (Notebook 4):** Default Presidio Analyzer has no `ORGANIZATION` recognizer. The synth dataset contains 200+ ORG annotations — all silently missed unless manually remapped.

**Issue type produced:** `DATASET_ONLY` (INFO) — non-blocking. Entities that the model never predicts are flagged so you're aware, but they don't block `get_mapped_results_dataframe()`. The false negatives are still counted in recall.

---

## 2. Massive Model Entity Set vs. Small Dataset

The model emits thousands of fine-grained labels; the dataset uses ≤10 broad categories.

| Model (1000+ entities) | Dataset (10 entities) |
|---|---|
| `US_SSN`, `UK_NINO`, `AU_TFN`, `IN_PAN`, `ES_NIF`, `PL_PESEL`, `FI_PERSONAL_IDENTITY_CODE`, `SG_NRIC_FIN`, `KR_RRN`, … | `ID` |
| `STREET_ADDRESS`, `CITY`, `STATE`, `POSTAL_CODE`, `COUNTRY`, `GEO_COORDINATES`, `GPE`, `LOC` | `LOCATION` |
| `FIRST_NAME`, `LAST_NAME`, `FULL_NAME`, `MAIDEN_NAME`, `PREFIX`, `TITLE`, `USERNAME` | `PERSON` |

**Real example (Notebook 5):** The OpenMed HuggingFace model predicts `PERSON`, `LOCATION`, `ORGANIZATION`, etc., while the synth dataset labels `city`, `country`, `street_address`, `state`, `county`, `coordinate`, `postcode` as separate entities.

**Issue type produced:** `COLLISION_AMBIGUOUS` (WARNING) — blocking. When a depth-2 ancestor like `LOCATION` is seen in predictions but the canonical surface is at depth 3 (computed by majority vote from the dataset annotations), the mapper can't determine which depth-3 entity to project onto. The user must call `map({'LOCATION': 'LOC'})` (or another appropriate target) to resolve.

**Projection rules in action:** If instead the dataset uses depth-2 labels and the canonical surface locks at depth 2, then fine-grained model labels like `STREET_ADDRESS` auto-collapse to `LOCATION` as `COLLISION_TRIVIAL` (INFO, non-blocking).

---

## 3. Semantic Ambiguity Requiring Human Judgment

Two entity types from different branches are semantically plausible for the same text span.

| Text span | Dataset label | Model prediction | Ambiguity |
|---|---|---|---|
| "64677" | `POSTAL_CODE` (under `LOCATION`) | `ID` (under `GOVERNMENT_ID`) | Zip codes look like generic numeric IDs |
| "Dr." | `TITLE` (under `PERSON`) | `JOB_TITLE` (under `EMPLOYMENT`) | Honorific prefix vs. profession |
| "Mr.", "Mrs." | `PREFIX` (under `PERSON`) | `TITLE` (under `PERSON`) | Both are valid canonical nodes |
| "Southern France" | `GPE` | `LOCATION` → partial overlap | "Southern" is a modifier, not a separate entity |
| "1977" | `DATE_TIME` | `AGE` or `CARDINAL` | Bare years are ambiguous |

**Real example (Notebook 4):** `NRP` (Nationality/Religious/Political group) auto-maps to `NATIONALITY`, which sits in the `DEMOGRAPHIC` branch. The synth dataset treats "Tunisian", "French" as location-adjacent. The user must override with `mapper.map({'NRP': 'NATIONALITY'})` or another target.

---

## 4. Same Dataset Label, Different Granularity Across Samples

The dataset uses a single label for spans that would naturally split into different canonical entities.

| Dataset label | Actual spans in data | Natural canonical split |
|---|---|---|
| `PERSON` | "Krisztián Szöllösy", "Rubija", "Dr. Smith" | `FULL_NAME`, `FIRST_NAME`, `FULL_NAME` (with `PREFIX`) |
| `STREET_ADDRESS` | "6750 Koskikatu 25 Apt. 864, Artilleros, CO, Uruguay 64677" | `BUILDING_NUMBER` + `STREET` + `SECONDARY_ADDRESS` + `CITY` + `STATE` + `COUNTRY` + `POSTAL_CODE` |
| `DATE_TIME` | "1977", "March 15, 2024", "3:00 PM" | `DATE`, `DATE`, `TIME` |

**Real example:** The synth dataset tags the entire address block as `STREET_ADDRESS`, but a model trained on CoNLL or OntoNotes might predict `LOC` for the city portion and `GPE` for the country portion only.

---

## 5. Country-Prefixed Labels from Regulatory Datasets

Datasets built for compliance (GDPR, HIPAA, PCI-DSS) use country-prefixed entity names that may or may not decompose cleanly.

| Raw label | Expected canonical | Risk |
|---|---|---|
| `GERMANY_PASSPORT_NUMBER` | `PASSPORT` | Works — `COUNTRY` tier: country prefix stripped, remainder resolves |
| `BRAZIL_CPF` | `TAX_ID` | Works — `COUNTRY` tier: `CPF` is an alias of `TAX_ID` |
| `FRANCE_CHEESE_LICENSE` | ??? | Falls back to `NATIONAL_ID` silently — wrong |
| `IN_PAN` | `TAX_ID` | `COUNTRY_FALLBACK` tier: two-letter `IN` = India, but `IT_SOMETHING` matches Italy even if unrelated |
| `US_DRIVER_LICENSE` | `DRIVER_LICENSE` | Works via `COUNTRY` tier, but dataset might just say `DRIVER_LICENSE` — are they the same? |
| `AUSTRALIAN_PASSPORT` vs `AU_PASSPORT` | Both should → `PASSPORT` | Demonym form vs ISO code — both must resolve identically |

**Identification tiers:** Country-prefix matching uses two tiers — `COUNTRY` (demonym or ISO code with remaining alias lookup) and `COUNTRY_FALLBACK` (strip leading country code and try again). Both fire before fuzzy matching.

---

## 6. Overlapping / Nested Entities

The model and dataset disagree on span boundaries because one tags nested entities and the other tags flat ones.

| Text | Dataset annotation | Model prediction |
|---|---|---|
| "Dr. Amelia Thornton" | `PERSON` (full span) | `PREFIX` ("Dr.") + `PERSON` ("Amelia Thornton") |
| "Bank of America" | `ORGANIZATION` (full span) | `GPE` ("America") ← model only sees the inner entity |
| "123 Main St, Springfield, IL 62704" | `STREET_ADDRESS` (full) | `ADDRESS` ("123 Main St") + `CITY` ("Springfield") + `STATE` ("IL") + `POSTAL_CODE` ("62704") |
| "Social Security Number: 123-45-6789" | `SSN` ("123-45-6789") | `SSN` ("Social Security Number: 123-45-6789") ← model over-captures |

---

## 7. Label Collision Across Hierarchy Branches

The same string alias appears under multiple canonical entities in the hierarchy.

| Alias | Branch 1 | Branch 2 | Problem |
|---|---|---|---|
| `LICENSE` | `EMPLOYMENT → LICENSE` | `GOVERNMENT_ID → PROFESSIONAL_LICENSE` | Last-write-wins in alias map |
| `VIN` | `GOVERNMENT_ID → VIN` | `VEHICLE_PII → VIN` | Duplicate canonical node |
| `LICENSE_PLATE` | `GOVERNMENT_ID → LICENSE_PLATE_NUMBER` | `VEHICLE_PII → LICENSE_PLATE` | Two branches claim the same concept |
| `VRN` | alias of `LICENSE_PLATE_NUMBER` | alias of `LICENSE_PLATE` (under `VEHICLE_PII`) | Ambiguous resolution |
| `MRN` / `MEDICAL_RECORD_NUMBER` | `PHI → MRN` | alias of `PATIENT_ID` | Same concept, two canonical targets |

**Issue types produced:**
- `COLLISION_CROSS_BRANCH` (WARNING) — blocking. Raised when a label resolves to a canonical entity that has co-occurring labels on the same tokens mapping to a different hierarchy branch. Must be resolved with `map()` before extracting results.
- `COLLISION_AMBIGUOUS` (WARNING) — blocking. Raised when a depth-2 ancestor maps to multiple depth-3 entities on the canonical surface (the top co-occurring candidate is shown in `overlap_counts`). Use `map({'LABEL': 'CANONICAL'})` to pick the right one.

---

## 8. Model Predicts Entities the Dataset Doesn't Annotate (Prediction-Only)

The model finds PII types the dataset creators never labeled — every detection is a false positive.

| Model prediction | Appears in dataset? | Example |
|---|---|---|
| `CREDIT_CARD` | No | Model detects "4532-XXXX-XXXX-1234" but the dataset focuses on names/locations only |
| `IP_ADDRESS` | No | Model finds "192.168.1.1" in free text but dataset ignores it |
| `URL` | No | "https://example.com" detected but not annotated |
| `CRYPTO` | No | Bitcoin address detected in medical notes dataset |
| `IBAN_CODE` | No | IBAN format matched in a US-only dataset |

**Real example (Notebook 5):** After mapping, several Presidio predictions had no dataset counterpart.

**Issue type produced:** `PREDICTION_ONLY` (WARNING) — blocking. These labels inflate precision with false positives. You have three resolution options:
1. **Suppress** — `mapper.map({'CREDIT_CARD': None})` excludes the label from evaluation entirely
2. **Remap** — `mapper.map({'CREDIT_CARD': 'FINANCIAL'})` counts detections against the `FINANCIAL` annotation set
3. **Keep as FP** — if you want these counted as false positives deliberately, this isn't directly supported; suppression is the recommended path

---

## 9. Dataset Has Entities the Model Can Never Produce (Annotation-Only)

The dataset annotates entity types that the model's architecture or recognizer set can't detect.

| Dataset entity | Why the model can't produce it | Impact |
|---|---|---|
| `BLOOD_TYPE` | No regex or NER for "O+", "AB-" | 100% FN for blood types |
| `SEXUAL_ORIENTATION` | Sensitive category, no recognizer | Every annotation is missed |
| `ZODIAC_SIGN` | Not a standard PII category | "Pisces" never detected |
| `FAMILY_HISTORY` | Requires clinical NLP, not NER | "Mother had diabetes" tagged but never predicted |
| `POLITICAL_AFFILIATION` | Inference-based, not pattern-based | "registered Democrat" is annotated but invisible to the model |

**Unlike prediction-only, the mapper currently has no `ANNOTATION_ONLY` issue type to flag this.**

---

## 10. Depth-Level Mismatch Within a Single Evaluation

The dataset and model operate at different hierarchy depths, producing ancestor–descendant pairs in the same result row.

| Annotation (depth 3) | Prediction (depth 2) | Relationship |
|---|---|---|
| `EMAIL_ADDRESS` | `CONTACT` | `CONTACT` ⊃ `EMAIL_ADDRESS` |
| `STREET_ADDRESS` | `LOCATION` | `LOCATION` ⊃ `STREET_ADDRESS` |
| `FIRST_NAME` | `PERSON` | `PERSON` ⊃ `FIRST_NAME` |
| `SSN` | `GOVERNMENT_ID` | `GOVERNMENT_ID` ⊃ `SSN` |

**Real example (Notebook 5):** Dataset labels `city`, `street_address`, `postcode` (depth 3), model predicts `LOCATION` (depth 2).

**How the new API handles this:** The canonical depth is computed automatically by majority vote from the dataset annotations. If the dataset is predominantly depth-3, the canonical surface is depth-3 and depth-2 model predictions trigger `COLLISION_AMBIGUOUS` (WARNING). Resolve with `map({'LOCATION': 'LOC'})` to pick the right depth-3 target.

---

## 11. BIO/IOB Tag Scheme Artifacts

The model outputs BIO-tagged labels; the mapper must strip them correctly.

| Raw model output | Expected strip | Edge case |
|---|---|---|
| `B-PERSON` | `PERSON` | Standard — works |
| `I-LOCATION` | `LOCATION` | Standard — works |
| `B-I-PERSON` | ??? | Nested prefix — regex strips only outer `B-`, leaves `I-PERSON` |
| `PERSON-B` | ??? | Suffix scheme — not handled by default prefix regex |
| `S-ORG` | `ORG` | BIOES scheme — single-token entity |
| `U-LOC` | `LOC` | BILOU scheme — unit entity |

---

## 12. Fuzzy Matching False Positives

Short or similar entity names get incorrectly matched by fuzzy resolution.

| Input label | Fuzzy match (≥0.80) | Correct? |
|---|---|---|
| `AGE` | `PAGE` | No — completely different concept |
| `FAX` | `TAX` | No — telecommunications vs. government |
| `PIN` | `VIN` | No — authentication vs. vehicle |
| `DOB` | `JOB` | No — date of birth vs. employment |
| `LOC` | `DOC` | No — location vs. document |
| `MAC` | `FAC` (facility) | No — network identifier vs. location |

---

## 13. Multi-Language / Multi-Script Entity Names

Datasets with non-English PII use native terms or transliterated labels.

| Dataset label | Language/origin | Expected canonical |
|---|---|---|
| `NOME` | Portuguese for NAME | `FULL_NAME` |
| `ADRESSE` | French/German for ADDRESS | `STREET_ADDRESS` |
| `NÚMERO_FISCAL` | Spanish for TAX NUMBER | `TAX_ID` |
| `PERSONNUMMER` | Swedish personal number | `NATIONAL_ID` |
| `ΕΘΝΙΚΟΤΗΤΑ` | Greek for NATIONALITY | `NATIONALITY` |
| `住所` | Japanese for ADDRESS | `STREET_ADDRESS` |

The current hierarchy aliases are English-only. Any non-English label goes straight to `UNRESOLVED`.

---

## 14. Conflicting Mapping Decisions at Different Evaluation Depths

A mapping override that makes sense at depth 2 becomes wrong at depth 3.

| Manual override at depth 2 | Meaning at depth 2 | What happens at depth 3 |
|---|---|---|
| `NRP → LOCATION` | Merges nationalities into location | `NRP` canonically resolves to `NATIONALITY` (under `DEMOGRAPHIC`) — mapping it to `LOCATION → ADDRESS`? `LOCATION → GPE`? Target is ambiguous |
| `AGE → DATE_TIME` | Merges age into temporal group | `AGE` should be `AGE` (under `DEMOGRAPHIC`), `DATE_TIME` splits into `DATE`, `TIME`, `EPOCH` — which one? |
| `ORGANIZATION → LOCATION` | Workaround for missing ORG support | At depth 3, `ORGANIZATION` has children (`COMPANY`, `SCHOOL`, `MEDICAL_FACILITY`) — they'd all route to `LOCATION`, but which `LOCATION` child? |

---

## 15. Composite / Multi-Entity Spans

A single text span semantically contains multiple entity types, but only one label is assigned.

| Text span | Single dataset label | Decomposed entities |
|---|---|---|
| "john.doe@acme.com" | `EMAIL_ADDRESS` | `USERNAME` ("john.doe") + `DOMAIN` ("acme.com") + `EMAIL_ADDRESS` (full) |
| "SSN: 123-45-6789, DOB: 01/15/1990" | `PERSON` (entire record) | `SSN` + `BIRTH_DATE` |
| "+1 (555) 123-4567 ext. 890" | `PHONE_NUMBER` | `PHONE_NUMBER` (base) + potentially `COUNTRY` (from +1) |
| "Dr. Jane Smith, MD, FACP" | `PERSON` | `PREFIX` + `FIRST_NAME` + `LAST_NAME` + `SUFFIX` + `SUFFIX` |

---

## 16. Temporal Ambiguity: Dates, Ages, and Durations

Numeric and temporal spans are inherently ambiguous across multiple entity types.

| Text span | Possible labels | Dataset vs. model |
|---|---|---|
| "25" | `AGE`, `BUILDING_NUMBER`, `ID`, `AMOUNT` | Dataset: `AGE` → Model: `CARDINAL` / nothing |
| "1977" | `DATE`, `YEAR`, `AGE`, `POSTAL_CODE`, `ID` | Dataset: `DATE_TIME` → Model: `DATE_TIME` (but via year regex or NER?) |
| "3 months" | `DURATION`, `AGE` | Dataset: `AGE` → Model: `DATE_TIME` |
| "March 15" | `DATE`, `BIRTH_DATE`, `DEATH_DATE` | All canonicalize differently at depth 3 |

---

## 17. Suppressed Entities That Still Have Annotations

An entity is suppressed (`→ None`) because the model can't detect it, but the dataset has annotations. Those ground-truth spans silently vanish from the evaluation.

| Suppressed entity | Annotation count in dataset | Effect |
|---|---|---|
| `EDUCATION_LEVEL` | 50+ | 50 annotated spans removed → recall inflated |
| `OCCUPATION` | 80+ | FN count drops but model still misses them |
| `LICENSE_PLATE` | 30+ | User thinks no LP problem, but it was hidden |

**Real example (Notebook 5):** `EDUCATION_LEVEL`, `OCCUPATION`, and `LICENSE_PLATE` are pre-mapped to `None`. After suppression the dataset annotations for those entities are excluded from the evaluation — this is intentional, but be aware that recall can appear inflated. The mapper does not produce a blocking issue for this; it's the user's explicit choice.

---

## 18. One-to-Many: A Single Model Entity Maps to Multiple Dataset Entities

The model predicts one label for spans that the dataset splits across categories.

| Model prediction | Dataset entities (all annotated separately) |
|---|---|
| `LOCATION` | `CITY`, `STATE`, `COUNTRY`, `POSTAL_CODE`, `STREET_ADDRESS`, `GPE`, `GEO_COORDINATES` |
| `PERSON` | `FIRST_NAME`, `LAST_NAME`, `FULL_NAME`, `MAIDEN_NAME`, `PREFIX` |
| `DATE_TIME` | `DATE`, `TIME`, `BIRTH_DATE`, `DEATH_DATE`, `DURATION`, `EPOCH` |
| `ID` | `SSN`, `PASSPORT`, `DRIVER_LICENSE`, `NATIONAL_ID`, `TAX_ID`, `VOTER_ID` |

At coarse depth they merge fine. At fine depth, the model can't distinguish — every prediction is technically correct but imprecise.

---

## 19. Cross-Domain Dataset Merge

A user combines two datasets from different domains, each with its own labeling convention.

| Dataset A (medical) | Dataset B (financial) | Collision |
|---|---|---|
| `ID` = patient MRN | `ID` = account number | Same label, completely different semantics |
| `DATE` = admission date | `DATE` = transaction date | Same label, different context (HIPAA vs PCI) |
| `LOCATION` = hospital name | `LOCATION` = branch address | Same label, one is really an ORG |
| `NAME` = patient name | `NAME` = account holder | Same label, both are PERSON but different privacy impact |

The mapper sees identical labels and merges them — but the underlying entities aren't equivalent.

---

## 20. Model Confidence Threshold Effects on Mapping

Entity mapping is evaluated *after* the model has already made decisions based on a score threshold — changing that threshold shifts which entities even appear.

| Threshold | Entities that appear/disappear | Mapping impact |
|---|---|---|
| 0.7 (high) | Only high-confidence detections survive → `PHONE_NUMBER`, `EMAIL_ADDRESS`, `CREDIT_CARD` | Fewer prediction-only issues, but many annotation-only gaps |
| 0.4 (medium) | Mid-confidence entities appear: `PERSON`, `LOCATION`, `DATE_TIME` | Balanced — typical mapping scenario |
| 0.1 (low) | Low-confidence noise: `AGE` (via custom regex), `TITLE`, spurious `PERSON` on common nouns | Many more prediction types to map, many more false positive collisions |

**Real example:** Notebook 4 uses threshold 0.4, Notebook 5 uses 0.3. The lower threshold in NB5 introduces `AGE` predictions (score 0.01 base) that only survive with context boosting — these are absent at 0.4.

---

## Summary Matrix

| # | Scenario | Core tension |
|---|---|---|
| 1 | Model missing common entities | Annotation-only blind spot |
| 2 | 1000s of model entities vs. 10 dataset | Many-to-few collapse |
| 3 | Semantic ambiguity | Human judgment needed |
| 4 | Inconsistent dataset granularity | Same label, different scopes |
| 5 | Country-prefixed labels | Prefix decomposition edge cases |
| 6 | Nested / overlapping spans | Boundary disagreement |
| 7 | Alias collision across branches | Hierarchy design conflict |
| 8 | Prediction-only entities | FP inflation |
| 9 | Annotation-only entities | FN inflation (no detection) |
| 10 | Depth mismatch | Ancestor–descendant pairs |
| 11 | BIO/IOB artifacts | Tag scheme stripping |
| 12 | Fuzzy matching false positives | Short-name confusion |
| 13 | Multi-language labels | English-only aliases |
| 14 | Depth-dependent overrides | Override breaks at new depth |
| 15 | Composite spans | Multi-entity in one span |
| 16 | Temporal ambiguity | Dates/ages/durations overlap |
| 17 | Suppressed with annotations | Silent recall inflation |
| 18 | One-to-many mapping | Imprecise predictions |
| 19 | Cross-domain dataset merge | Same label, different semantics |
| 20 | Confidence threshold effects | Threshold changes entity landscape |

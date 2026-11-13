# ADR-004: Column-Level Evaluation for presidio-structured

## Status

Proposed

## Date

2026-11-13

## Context

[#98](https://github.com/data-privacy-stack/presidio-research/issues/98) asks for a way to evaluate how well
`presidio-structured` identifies PII columns in tables and JSON (#136 was closed as a duplicate of it). On #98,
the maintainers asked for a design before code, answering three questions:

1. How does this integrate with the rest of `presidio-evaluator`?
2. How are metrics calculated, given that PII work usually cares more about recall than precision?
3. Which datasets are used, and how is the ground truth generated?

The Decision section answers them in that order.

### What presidio-structured predicts

`PandasAnalysisBuilder.generate_analysis(df, n=None, language="en", selection_strategy="most_common",
mixed_strategy_threshold=0.5)` samples `n` rows (`random_state=123`), runs the analyzer on every cell, and
returns a `StructuredAnalysis` whose `entity_mapping` holds **one entity type per column**:

- All three strategies return a single entity. `mixed` picks `highest_confidence` when the top score is above
  the threshold and `most_common` otherwise.
- `most_common` counts only cells that had a hit. A column where 14 of 50 cells matched something is still
  labelled with the most common of those 14 hits.
- A column becomes `NON_PII` only when no cell had any hit, and `NON_PII` columns are dropped from
  `entity_mapping`.
- Detection reads values, not headers. Renaming every column to `col_1 … col_n` gives the same mapping.

`JsonAnalysisBuilder.generate_analysis(data)` returns one entity per key, using the first recognizer result.
Nested objects come back as dot paths (`customer.address.street`). Keys whose value is a list are skipped, so
they never appear in the mapping.

`StructuredEngine` then anonymizes only the columns listed in `entity_mapping`, with the operator configured for
that entity type (or `DEFAULT`). This is what makes the column the unit that matters: a missed column is left in
the clear in full, and a wrongly typed column is still anonymized, only with a different operator.

### What presidio-evaluator has today

Following ADR-001, the results DataFrame is the interface between the model and the evaluator. The current
pipeline is `model.predict_dataset()` → `CanonicalMapper.analyze()` → `get_mapped_results_dataframe()`, which
returns `MappedResults` with `binary`, `branch` and `detailed` frames → `calculate_hierarchical_scores()` →
`EvaluationResult` and `Plotter`. Everything after `predict_dataset()` works on the DataFrame. Nothing in the
package scores a column → entity mapping.

## Decision

Evaluate `presidio-structured` **one column (or JSON key path) at a time**, using a 4-column results DataFrame
and a thin `StructuredEvaluator`. Everything after the adapter is the existing pipeline, unchanged.

### Column results DataFrame

| Column | Type | Description |
|---|---|---|
| `table_id` | `str` | Table or JSON document the column belongs to |
| `column` | `str` | Column name, or the dot path for JSON (`customer.address.city`) |
| `annotation` | `str` | Ground-truth entity, or `O` for a non-PII column |
| `prediction` | `str` | Entity from `entity_mapping`, or `O` |

An adapter, `structured_analysis_to_results_df(analysis, ground_truth, table_id)`, builds it with three rules:

1. One row per ground-truth column. A column in `entity_mapping` that is missing from the ground truth raises an
   error rather than being ignored, so a false positive can't fall out of the score unnoticed.
2. A column absent from `entity_mapping`, or mapped to `NON_PII`, becomes `O`. This also gives the Pandas
   builder (drops `NON_PII`) and the JSON builder (never emits it) the same rule. The label has to be `O`, not
   `NON_PII`: `CanonicalMapper` resolves the string `NON_PII` to `PII` (`get_mapping()` returns
   `{"NON_PII": "PII"}`), which would turn every non-PII column into a hit.
3. The adapter reads only `analysis.entity_mapping`, so `presidio-evaluator` never imports
   `presidio-structured`. A plain `dict[str, str]` from any other structured detector works the same way.

```text
table_id   column         annotation     prediction
customers  name           PERSON         PERSON
customers  email          EMAIL_ADDRESS  EMAIL_ADDRESS
customers  phone          PHONE_NUMBER   PERSON
customers  ssn            US_SSN         O
customers  age            O              PERSON
customers  order_id       O              O
customers  price          O              O
customers  internal_code  O              O
```

### 1. Integration with presidio-evaluator

`StructuredEvaluator` subclasses `TokenEvaluator`. `TokenEvaluator` already counts one `(annotation, prediction)`
pair per row, which is per-column counting once each row is a column. The subclass only:

- turns the 4-column frame into the token schema (`table_id` → `sentence_id`, `column` → `token`, column position
  → `start_indices`) and calls `TokenEvaluator.calculate_score_on_df`
- defaults `skip_words` to `[]`. Column names are not tokens, and `name`, `age`, `address`, `street`, `city`,
  `state`, `zip`, `country`, `year`, `month` and `day` are all default skip words. In the example above, renaming
  `phone` to `address` and `ssn` to `zip` and keeping the default skip words lifts every score to 1.00, because
  the errors on those columns, and the `age` false positive, are skipped instead of counted.

```python
import pandas as pd
from presidio_structured import PandasAnalysisBuilder

from presidio_evaluator.entity_mapping import CanonicalMapper
from presidio_evaluator.evaluation import Plotter, StructuredEvaluator  # StructuredEvaluator is new
from presidio_evaluator.evaluation.structured_evaluator import structured_analysis_to_results_df  # new

tables, ground_truth = load_structured_dataset("data/structured_v1")  # new, see section 3

builder = PandasAnalysisBuilder()
results_df = pd.concat(
    [
        structured_analysis_to_results_df(
            builder.generate_analysis(df, selection_strategy="most_common"),
            ground_truth[table_id],
            table_id=table_id,
        )
        for table_id, df in tables.items()
    ],
    ignore_index=True,
)

mapper = CanonicalMapper()
mapper.analyze(results_df)
mapped_results = mapper.get_mapped_results_dataframe()

evaluator = StructuredEvaluator()
hierarchical_results = evaluator.calculate_hierarchical_scores(mapped_results, beta=2)

Plotter(results=hierarchical_results["detailed"], model_name="presidio-structured", beta=2).plot_scores()
```

| Component | Change |
|---|---|
| `evaluation/structured_evaluator.py` | **New**: `StructuredEvaluator` and `structured_analysis_to_results_df` |
| `CanonicalMapper` | None. It needs only `annotation` and `prediction` and carries the other columns through |
| `BaseEvaluator.calculate_hierarchical_scores` | None. It calls `calculate_score_on_df` once per level |
| `TokenEvaluator`, `SpanEvaluator` | None. Their 5-column contract is untouched |
| `EvaluationResult`, `ModelError`, `Plotter` | None. `model_errors` holds the column name in `token`, so error analysis reads `phone: PHONE_NUMBER → PERSON`. `plot_most_common_tokens()` has nothing useful to show here |
| `ExperimentTracker` | None. The builder configuration is logged as parameters (section 2) |
| `pyproject.toml` | `presidio-structured` as an optional extra, used only by the notebook and the integration test |

I checked this design with a short prototype of the adapter and `StructuredEvaluator` against `main`. The 4-column frame goes
through `CanonicalMapper`, `calculate_hierarchical_scores`, `to_confusion_df()` and `Plotter` with no changes,
and two tables that share column names stay separate.

### 2. How metrics are calculated

The unit is one column. Each level of `calculate_hierarchical_scores` answers a different question:

- **binary**: did the detector flag every PII column? This is the privacy number, because a false negative here
  means the whole column ships without anonymization.
- **branch**: did it get the category right (`CONTACT`, `LOCATION`, `GOVERNMENT_ID`, …)?
- **detailed**: did it get the type right at the dataset's own granularity? The ADR-003 rule is inherited as is:
  a prediction is projected to the deepest annotated ancestor of its own label.

| Case | binary | branch / detailed |
|---|---|---|
| Prediction equals the ground truth | TP | TP for that entity |
| PII predicted, ground truth `O` | FP | FP for the predicted entity |
| `O` predicted (or column missing), ground truth PII | FN | FN for the true entity |
| Type A predicted, type B is the ground truth | TP (the column still gets anonymized) | FP for A + FN for B |
| Both `O` | TN (shown in the confusion matrix only) | TN |

`beta=2` stays the default, as in the rest of the package. The reasoning is stronger for columns than for tokens.
A missed column is every value in that column left in the clear. A false alarm costs utility, since one non-PII
column gets anonymized.

Results are reported as they are today: per-type precision, recall, F2 and counts, plus the `pii_*` aggregate.
The single micro-averaged number per level in the examples below is computed from the per-type counts, only to
keep the examples short. A dataset has hundreds of columns, not hundreds of thousands of tokens, so the report
shows `num_annotated` next to every per-type score. Two misses on a 4-column type are a 50% recall drop.

A result is only comparable with another if the builder configuration is recorded with it. These are logged
through `ExperimentTracker.log_parameters`: `selection_strategy`, `n`, `mixed_strategy_threshold`, the analyzer's
score threshold, the spaCy model, and the `presidio-analyzer` and `presidio-structured` versions.

#### Worked example (unit-test fixture)

The 8-column table above, run through the prototype:

| Level | TP | FP | FN | Precision | Recall | F2 |
|---|---|---|---|---|---|---|
| binary | 3 | 1 | 1 | 0.75 | 0.75 | 0.75 |
| branch | 2 | 2 | 2 | 0.50 | 0.50 | 0.50 |
| detailed | 2 | 2 | 2 | 0.50 | 0.50 | 0.50 |

`phone` is a binary TP but a type error (`PHONE_NUMBER` → `PERSON`). `ssn` is a miss (`US_SSN` resolves to
`SSN`). `age` is a false positive. The binary level says three of four PII columns would be anonymized; the other
two levels show the types are wrong half the time.

#### Example on real presidio-structured output

50 rows of the repo's fake-person table (see section 3), with Faker card numbers and four non-PII columns
(`order_id` as six random digits, `quantity`, `price`, `status`). Run with `presidio-structured` 0.0.10,
`presidio-analyzer` 2.2.364 and `en_core_web_lg` 3.8.0:

| Column | Ground truth | `most_common` | `highest_confidence` |
|---|---|---|---|
| `name` | `PERSON` | `PERSON` | `PERSON` |
| `email` | `EMAIL_ADDRESS` | `EMAIL_ADDRESS` | `EMAIL_ADDRESS` |
| `phone_number` | `PHONE_NUMBER` | `PHONE_NUMBER` | `DATE_TIME` |
| `street_name` | `STREET_ADDRESS` | `PERSON` | `PERSON` |
| `city` | `CITY` | `LOCATION` | `LOCATION` |
| `date_of_birth` | `DATE_OF_BIRTH` | `DATE_TIME` | `DATE_TIME` |
| `credit_card_number` | `CREDIT_CARD` | `US_BANK_NUMBER` | `CREDIT_CARD` |
| `national_id` | `NATIONAL_ID` | `US_DRIVER_LICENSE` | `DATE_TIME` |
| `zipcode` | `ZIP_CODE` | `DATE_TIME` | `DATE_TIME` |
| `order_id` | `O` | `US_DRIVER_LICENSE` | `DATE_TIME` |
| `quantity`, `price`, `status` | `O` | not in mapping | not in mapping |

| Level (F2) | `most_common` | `highest_confidence` |
|---|---|---|
| binary | 0.98 (P 0.90, R 1.00) | 0.98 (P 0.90, R 1.00) |
| branch | 0.76 (P 0.70, R 0.78) | 0.54 (P 0.50, R 0.56) |
| detailed | 0.43 (P 0.40, R 0.44) | 0.33 (P 0.30, R 0.33) |

What the evaluator shows here, that a single number would not:

- Every PII column would be anonymized (binary recall 1.00). The one false positive is `order_id`: all 50
  six-digit IDs match the `US_DRIVER_LICENSE` pattern.
- The types are often wrong. Street addresses come back as `PERSON`, and `zipcode` becomes `DATE_TIME` from hits
  in only 14 of 50 cells.
- The strategy matters. `highest_confidence` fixes the card column but turns `phone_number` and `national_id`
  into `DATE_TIME` (`mixed` gave the same mapping as `highest_confidence` here).
- `credit_card_number` counts as a TP at the detailed level even under `most_common`, because the mapper resolves
  both `CREDIT_CARD` and `US_BANK_NUMBER` to `FINANCIAL`. That is ADR-002 behaviour and worth knowing when reading
  the numbers.

A nested JSON record, built from the first row of the same table plus a list of phone numbers:

```json
{
  "customer": {
    "name": "Marie Hamanova",
    "email": "MarieHamanova@armyspy.com",
    "address": {"street": "P.O. Box 255", "city": "Kangerlussuaq", "zip": "3910"}
  },
  "order_id": "448213",
  "phones": ["+1 212-555-0187", "+44 20 7946 0958"]
}
```

`JsonAnalysisBuilder` returns `customer.name: PERSON`, `customer.email: EMAIL_ADDRESS`,
`customer.address.street: PERSON` and `order_id: DATE_TIME`. `customer.address.city`, `customer.address.zip`
and `phones` are not in the mapping, so they score as false negatives.

### 3. Datasets and ground truth

**v1 source.** The repo already ships synthetic tabular PII: `FakeNameGenerator.com_3000.csv`, loaded by
`load_fake_person_df()` (3000 rows, 37 columns after the loader's renames). The loader drops the card number in
favour of Faker's, so v1 adds a Faker `credit_card_number` column. The values are already messy in useful ways:
national IDs come from many countries, are often empty (30 of the 50 sampled rows) and are sometimes
spreadsheet-mangled (`5.70611E+12`), and phone numbers and postcodes come in many national formats.

**Non-PII columns.** These are generated with a fixed seed and chosen to tempt the recognizers: order IDs and
SKUs (digit runs), quantities, prices and status values. The real run above shows why they are needed:
`quantity`, `price` and `status` were never flagged, so without `order_id` there would be no false positive to
find and binary precision would read 1.00.

**Ground truth.** A schema file written by hand, one row per column, with labels based on the values, not the
header:

```csv
table_id,column,entity
people_01,name,PERSON
people_01,email,EMAIL_ADDRESS
people_01,phone_number,PHONE_NUMBER
people_01,street_name,STREET_ADDRESS
people_01,city,CITY
people_01,zipcode,ZIP_CODE
people_01,date_of_birth,DATE_OF_BIRTH
people_01,national_id,NATIONAL_ID
people_01,credit_card_number,CREDIT_CARD
people_01,order_id,NON_PII
people_01,quantity,NON_PII
```

**Tables.** About ten tables, each built from a different seeded subset of 8–15 columns and 50 rows. That gives a
few hundred labelled columns, which is enough for stable numbers on common types and small enough to review by
hand.

**Header-neutral copies.** Each table also ships with its columns renamed to `col_1 … col_n`. Today's builders
ignore headers, so the two copies score the same. A future builder that reads headers would score differently,
and this check would catch that.

**JSON.** Nested records are built from the same rows, labelled by dot path, including a list-valued key (`phones`).

**Not in v1.** Free-text columns are left out. For example, a `notes` value "Call Marie after 5pm" gets one column
label (`PERSON`), which is a poor fit, and the existing span evaluator is the right tool for free text.

No real PII is used at any point.

## Consequences

### Positive

- Closes #98 with one adapter and one small subclass. Scoring, mapping, hierarchy, plotting and experiment
  tracking are all reused, so structured and token results read the same way.
- Keeps ADR-001's principle: the DataFrame is the interface, so results can be saved and re-scored without
  re-running the analyzer.
- Separates "did we miss a PII column?" (binary) from "did we get the type right?" (branch and detailed). For
  `presidio-structured`, these map directly onto "is the column anonymized" and "is it anonymized with the right
  operator".
- No new required dependency.

### Negative / Trade-offs

- Scoring by column is coarse. A correctly labelled column can still have values the analyzer missed, and v1
  will not see them. The docs must say so.
- `StructuredEvaluator` inherits `TokenEvaluator`, so a change to token counting changes column scores. The
  worked example becomes a unit test to pin that down.
- Column counts are small, so per-type numbers on rare types are noisy.
- Synthetic values are cleaner than real columns. The non-PII columns and the messy national IDs partly offset
  this, but the numbers will still be optimistic.

## Implementation Plan

1. **Adapter and evaluator.** Add `presidio_evaluator/evaluation/structured_evaluator.py` and export
   `StructuredEvaluator` from `presidio_evaluator.evaluation`. Unit tests in
   `tests/evaluation/test_structured_evaluator.py`:
   - the worked example
   - a column named `address` (skip words)
   - `NON_PII` and an absent column, both becoming `O`
   - a predicted column missing from the ground truth (raises)
   - two tables with the same column names
2. **Dataset builder.** Add a seeded generator in `presidio_evaluator/data_generator/` that writes the tables,
   the header-neutral copies, the JSON records and the schema CSV from `load_fake_person_df()`.
3. **Integration test.** One small table through `PandasAnalysisBuilder`, marked `integration`, skipped when
   `presidio-structured` is not installed.
4. **Docs and notebook.** A `7_Evaluate_Presidio_Structured.ipynb` notebook following notebook 4, a
   `docs/structured_evaluation.md` page in the mkdocs nav, and a CHANGELOG entry.

## Alternatives Considered

### A new scorer subclassing `BaseEvaluator`

Write the per-column counting from scratch. Rejected because `TokenEvaluator` already counts per row, and a
second copy of the precision, recall and `EvaluationResult` assembly would drift from the first.

### Use `TokenEvaluator` directly, with no subclass

Ask users to rename the columns and pass `skip_words=[]`. Rejected because forgetting either step gives wrong
scores silently (the skip-words example above went from 0.75 to 1.00), and the error is easy to make.

### Extend `SpanEvaluator`

Rejected. Its span and IoU logic means nothing for columns.

### Score cells, not columns

Rejected for v1. The builder decides per column and `StructuredEngine` acts per column, so cell-level scoring
would measure something the user never acts on. It is a reasonable v2 addition.

### Make `presidio-structured` a required dependency

Rejected. It ties the evaluator's install and releases to another package for the sake of one adapter, and it
would make evaluating any other structured detector harder.

## Open Questions

1. Subclassing `TokenEvaluator` keeps the new code small. Would the maintainers prefer a standalone class that
   reuses only `TokenEvaluator`'s aggregation?
2. Should the v1 tables be committed under `data/`, or generated from the seed at test time?
3. Should the notebook compare all three selection strategies, or show `most_common` only?
4. Should list-valued JSON keys be in the v1 ground truth? They will show up as misses until
   `JsonAnalysisBuilder` handles lists.

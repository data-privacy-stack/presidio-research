# Span Evaluation in Presidio Evaluator

This document explains the span evaluation process implemented in Presidio Evaluator, covering how spans are created,
matched, and evaluated, along with comparisons to other evaluation paradigms.

## Span Creation and Processing

### Span Creation

Spans are created from token-level annotations in the input data. Each span represents a continuous sequence of tokens
with the same entity type annotation. The basic properties of a span include:

- `entity_type`: The type of entity (e.g., PERSON, LOCATION)
- `entity_value`: The actual text of the entity
- `start_position` and `end_position`: Character-level boundaries


#### Span Normalization

For more advanced processing, spans also include normalized versions of the text:

- `normalized_tokens`: List of normalized tokens that make up the span (typically lowercased and with special characters
  removed)
- `normalized_start_index` and `normalized_end_index`: Character indices in the normalized text

Normalization helps with more consistent matching between variations of the same entity (e.g., "John Smith" vs "john
smith") and by better handling of punctuation marks and skip words.

### Span Merging

In some cases, multiple separate tokens may need to be merged into a single span:

1. **Adjacent tokens of same type**: Consecutive tokens with the same entity type are merged into a single span
2. **Skip words handling**: Certain configurable words (like punctuation marks or skip words) can be included in spans even if
   they are annotated as non-entities, allowing for more natural entity boundaries

Example of skip words:

```
Text: "University of Washington"
Without skip words: [ORG, O, ORG]
With "of" as skip word: [ORG, ORG, ORG] (treated as one span)
```

The `skip_words` parameter in the `SpanEvaluator` constructor determines which words can be skipped when merging
adjacent spans of the same entity type.

`CanonicalMapper` adds `annotation_merge_key` and `prediction_merge_key` to all
four `MappedResults` DataFrames, including `.original`. The keys contain the
detailed scoring labels (after prediction projection to the gold vocabulary).
A change of key ends a token run even when the visible label is unchanged.
Adjacent spans merge only when **both** their scored labels and merge keys agree.
This keeps, for example, a name and an age separate when both are scored as `PII`.
The original input columns and values are preserved in `.original`; its column
set gains the two metadata columns.

Without the paired metadata column, span creation and merging retain label-only
behavior. Unrelated label columns such as `pred_a` never use `prediction_merge_key`.
When metadata is present, an invalid sentence-relative `token_start` raises
`ValueError` instead of silently disabling the keys.

Merge keys are labels, not source span identities: distinct same-type entities
such as `"Paris , London"` can still merge. Preserving source span identities
through tokenization is a separate follow-up, not part of this fix.

## Span Matching Strategy

The evaluator compares annotation spans (gold standard) with prediction spans (model output) using an Intersection over
Union (IoU) approach. This can be either character-based or token-based, controlled by the `char_based` parameter.

### IoU Calculation

- **Character-based IoU**: Calculates the character overlap between spans
- **Token-based IoU**: Calculates the token overlap between spans

IoU = (Intersection) / (Union)

An IoU threshold determines whether spans match sufficiently.

### Matching Annotations to Predictions

The matching process follows these steps:

1. Assign each prediction to at most one overlapping annotation: prefer a same-type
   match meeting the IoU threshold, then the greatest IoU. Ties prefer the same
   type, then the earliest gold start and end, then the entity type in descending
   lexical order. This assignment is independent of annotation input order.
2. Group overlapping prediction spans by entity type
3. Calculate combined IoU for each entity type group
4. Determine match status based on IoU threshold and entity type

> For a detailed breakdown of different matching scenarios and examples, see
> the [Span Matching Strategies](span_matching_strategies.md) document.

## Metric Calculation

The evaluator calculates both per-entity-type metrics and global PII metrics.
Recall is counted per annotation: each annotation is a true positive if
predictions of its type cover it at IoU ≥ threshold, and a false negative
otherwise. Precision is counted per prediction span: each span is counted once
in `num_predicted`, either credited by a successful match or counted as a
false positive.

### Per-Entity-Type Metrics

- **Precision**: (num_predicted − FP) / num_predicted
- **Recall**: TP / num_annotated
- **F-beta**: (1 + beta²) * (precision * recall) / (beta² * precision + recall)

Note that precision is not TP / num_predicted: TP counts covered annotations,
and two fragments jointly covering one annotation are one TP but two credited
predictions.

### Global PII Metrics

- Treat every entity type as if it were a single PII type
- Calculate global precision, recall, and F-score on PII/not PII values

### Confusion Matrix and Error Records

The confusion matrix (`EvaluationResult.results`) is annotation-centric:

- Every annotation lands in exactly one cell: `(type, type)` when covered,
  `(type, predicted type)` when a different type covers it at IoU ≥ threshold,
  and `(type, "O")` when nothing does. Row totals therefore equal
  `num_annotated` per type. When several types reach the threshold on the same
  annotation (possible at thresholds of 0.5 or below), the annotation's own
  type claims the cell; otherwise the wrong type with the highest IoU does.
  Spans of the other types are false positives in the `"O"` row.
- The `"O"` row holds prediction spans that appear in no annotation cell: false
  positives that overlap nothing, or overlap an annotation below the threshold.
  A prediction already represented by a `(type, predicted type)` cell is not
  added to the `"O"` row again.
- Column totals are not the prediction ledger. One prediction that covers two
  annotations appears in two cells while counting once in `num_predicted`. Use
  `num_predicted` and `false_positives` for prediction-side totals.

Confusion-matrix cells and `ModelError` records are written by the per-type
pass only. The global PII pass updates the `pii_*` counters and nothing else,
so `calculate_score_on_df(level="both")` records each error once.


## Evaluation Process

1. Assign predictions to annotations using the exclusive assignment described above
2. Group overlapping spans by entity type
3. Calculate the same-type coverage (pairwise IoU for a single span, combined
   IoU for several)
4. Determine match status based on IoU and entity type
5. Count each prediction span once: credited if it participated in a
   successful match, otherwise a false positive

See more info on the [Span Matching Strategies](span_matching_strategies.md) document.

## Counting Strategy

- Every prediction is assigned to at most one overlapping annotation before
  scoring: a same-type match meeting the threshold wins, then the greatest IoU,
  with deterministic ties. A prediction is therefore never a TP for two
  annotations
- Every annotation is counted once in `num_annotated` and receives one verdict
  (TP or FN), regardless of how many predictions or types intersect with it
- Every prediction span is counted once in `num_predicted` — a span is not
  re-counted per annotation it overlaps, and grouped spans that jointly fail
  count one FP each
- Different entity types are counted separately

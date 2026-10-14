# Span Evaluation in Presidio Evaluator

This document explains the span evaluation process implemented in Presidio Evaluator, covering how spans are created,
matched, and evaluated, along with comparisons to other evaluation paradigms.

## Span Creation and Processing

### Span Creation

A span is one entity in the text, such as a name, a date or an email address, and
may cover several tokens. Each sentence has two independent sets of spans: the
annotated spans (the ground truth) and the predicted spans (what the model found).
The two sets can differ in number, position and type, and may not overlap at all.

Presidio Evaluator scores spans at three levels of detail:

- **Binary**: is it PII or not.
- **Branch**: a canonical entity such as person or location.
- **Detailed**: a more granular entity type, if one exists in the dataset or is
  produced by the model.

Where each span starts and ends is decided once, from the original data before
entity types are converted to any of these levels, and kept at every level. The
results data records it on every token, in the `annotation_span_id` and
`prediction_span_id` columns, so the evaluator knows which tokens belong to the
same entity.

Each span has:

- `entity_type`: The type of entity (e.g., PERSON, LOCATION)
- `entity_value`: The actual text of the entity
- `start_position` and `end_position`: Character-level boundaries

!!! note
    In most cases the dataset and the model mark where each entity begins and ends.
    When they don't, two adjacent entities of the same type are read as one span.
    For example, in "Invite Alice Bob", if both names are labeled `PERSON` and
    nothing marks where one ends, they are evaluated as a single span, "Alice Bob".

#### Span Normalization

For more advanced processing, spans also include normalized versions of the text:

- `normalized_tokens`: List of normalized tokens that make up the span (typically lowercased and with special characters
  removed)
- `normalized_start_index` and `normalized_end_index`: Character indices in the normalized text

Normalization helps with more consistent matching between variations of the same entity (e.g., "John Smith" vs "john
smith") and by better handling of punctuation marks and skip words.

#### Skip Words

Skip words (punctuation, stop words) are ignored when comparing spans. A span made
only of skip words is dropped. Skip words do not join separate spans: in
`"Michael and Bob"`, with `and` as a skip word, `Michael` and `Bob` stay two spans.

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

1. For each annotation span, find all overlapping prediction spans
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
and a single prediction covering two annotations is two TPs but one prediction.

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

1. For each annotation, find all overlapping prediction spans
2. Group overlapping spans by entity type
3. Calculate the same-type coverage (pairwise IoU for a single span, combined
   IoU for several)
4. Determine match status based on IoU and entity type
5. Count each prediction span once: credited if it participated in a
   successful match, otherwise a false positive

See more info on the [Span Matching Strategies](span_matching_strategies.md) document.

## Counting Strategy

- Every annotation is counted once in `num_annotated` and receives one verdict
  (TP or FN), regardless of how many predictions or types intersect with it
- Every prediction span is counted once in `num_predicted` — a span is not
  re-counted per annotation it overlaps, and grouped spans that jointly fail
  count one FP each
- Different entity types are counted separately

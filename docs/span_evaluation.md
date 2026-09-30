# Span Evaluation in Presidio Evaluator

This document explains the span evaluation process implemented in Presidio Evaluator, covering how spans are created,
matched, and evaluated, along with comparisons to other evaluation paradigms.

## Span Creation and Processing

### Span Creation

Spans are reconstructed from the per-token results DataFrame produced by
`BaseModel.predict_dataset()`. Besides the label columns (`annotation`,
`prediction`) that frame carries two **span-id** columns, `annotation_span_id`
and `prediction_span_id`: the entity instance that covers each token, `None`
for `O`. A span is a maximal run of tokens sharing an id. The basic properties
of a span include:

- `entity_type`: The type of entity (e.g., PERSON, LOCATION)
- `entity_value`: The actual text of the entity
- `start_position` and `end_position`: Character-level boundaries

Ids come from the source, not from the labels:

- **Gold**: `InputSample` records which span produced each token's tag
  (`span_ids`) when tags are created from spans. Samples built from BIO/BILUO
  tags recover the ids from the `B-`/`U-` prefixes; plain IO tags give label
  runs, so two touching entities of the same type read as one.
- **Predictions**: `BaseModel.batch_predict_spans()` returns the model's spans.
  The Presidio wrappers return their `RecognizerResult`s directly; the default
  implementation converts the tags from `batch_predict()` at that boundary,
  with the same BIO-exact / IO-runs rule as above.

Because `CanonicalMapper` only rewrites the label columns, the ids are identical
at every mapping level and so are the spans. Two adjacent entities stay separate
at the binary level even though both read `PII`, and even when they are of the
same type (`"Paris , London"` is two spans). The ground truth therefore never
depends on the level being scored.

`SpanEvaluator` requires the span-id columns and raises `ValueError` when they
are missing. Build the DataFrame with `predict_dataset()`.

#### Span Normalization

For more advanced processing, spans also include normalized versions of the text:

- `normalized_tokens`: List of normalized tokens that make up the span (typically lowercased and with special characters
  removed)
- `normalized_start_index` and `normalized_end_index`: Character indices in the normalized text

Normalization helps with more consistent matching between variations of the same entity (e.g., "John Smith" vs "john
smith") and by better handling of punctuation marks and skip words.

#### Skip Words

The `skip_words` parameter of `SpanEvaluator` lists tokens (punctuation, titles,
stop words) that are dropped from a span's normalized form before IoU is
computed, so `"Mr. John Smith"` and `"John Smith"` compare equal when `mr.` is
a skip word. A span made only of skip words is dropped. Skip words never merge
two spans: a span's boundaries are fixed by its id, and a skip token inside a
source span (the comma in `"New, York"`) already carries that span's id.

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

The evaluator calculates both per-entity-type metrics and global PII metrics:

### Per-Entity-Type Metrics

- **Precision**: TP / num_predicted
- **Recall**: TP / num_annotated
- **F-beta**: (1 + beta²) * (precision * recall) / (beta² * precision + recall)

### Global PII Metrics

- Treat every entity type as if it were a single PII type
- Calculate global precision, recall, and F-score on PII/not PII values


## Evaluation Process

1. Assign predictions to annotations using the exclusive assignment described above
2. Group overlapping spans by entity type
3. Calculate combined IoU for each group
4. Determine match status based on IoU and entity type
5. Mark remaining predictions (with no overlap) as FPs

See more info on the [Span Matching Strategies](span_matching_strategies.md) document.

## Counting Strategy

- Multiple predictions of the same type overlapping with one annotation count as a single prediction
- A prediction participates in only one annotation's group. It cannot be counted
  as a TP or FP again for another gold span. This preserves fragment aggregation;
  `num_predicted` is still a count of evaluated groups, not always raw model spans.
- Different entity types are counted separately
- An annotation is only counted once as annotated (denominator for precision and recall),
  regardless of how many types intersect with it


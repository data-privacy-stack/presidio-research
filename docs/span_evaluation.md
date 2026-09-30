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


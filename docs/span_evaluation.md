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

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

Counting is **two-sided**: recall is counted per annotation and precision is
counted per prediction span. The two sides keep separate numerators, because
one prediction may satisfy several annotations (or several predictions may
jointly satisfy one annotation) without being consumed.

- **Recall side (per annotation)**: every annotation gets exactly one verdict.
  It is a *true positive* if the predictions of its own type cover it at
  IoU ≥ threshold (a single span measured pairwise, several spans measured by
  their combined IoU), and a *false negative* otherwise.
- **Precision side (per prediction span)**: every prediction span enters
  `num_predicted` exactly once, no matter how many annotations it overlaps. A
  span that participated in at least one successful same-type match is
  *credited*; any other span is a *false positive*.

### Per-Entity-Type Metrics

- **Precision**: (num_predicted − FP) / num_predicted — the fraction of
  emitted spans that were credited. Note this is *not* TP / num_predicted:
  TP counts covered annotations, and one wide span covering two annotations
  is two recall hits but a single credited prediction.
- **Recall**: TP / num_annotated
- **F-beta**: (1 + beta²) * (precision * recall) / (beta² * precision + recall)

Two invariants always hold, and make results easy to sanity-check:

1. `TP + FN == num_annotated` — every annotation is counted exactly once.
2. `num_predicted` equals the number of prediction spans the model actually
   emitted (after span merging), each counted exactly once as credited or FP.

### Global PII Metrics

- Treat every entity type as if it were a single PII type
- Calculate global precision, recall, and F-score on PII/not PII values


## Evaluation Process

1. **Recall pass** — for each annotation: find all overlapping prediction
   spans, group them by entity type, measure the same-type group's coverage
   (pairwise IoU for one span, combined IoU for several), and issue one
   verdict: TP if coverage ≥ threshold, else FN. Wrong-type groups covering
   the annotation at ≥ threshold are recorded as `WrongEntity` errors for
   error analysis.
2. **Precision pass** — for each prediction span: count it once in
   `num_predicted`; if it never participated in a successful match, count it
   once as FP (whether it overlapped an annotation insufficiently or nothing
   at all).

See more info on the [Span Matching Strategies](span_matching_strategies.md) document.

## Counting Strategy

- Every annotation is counted exactly once in `num_annotated` and receives
  exactly one verdict (TP or FN), regardless of how many predictions or types
  intersect with it.
- Every prediction span is counted exactly once in `num_predicted` and, if
  unmatched, exactly once as FP — grouped same-type spans that jointly fail
  count one FP *each*, and a span overlapping several annotations is *not*
  re-counted per annotation.
- Different entity types are counted separately.
- In the confusion matrix each span appears in exactly one cell: a wrong-type
  detection at ≥ threshold is recorded as (annotation type, predicted type),
  replacing both the (annotation type, "O") entry for the gold and the
  ("O", predicted type) entry for the prediction.

# Span Matching Strategies in Presidio Evaluator

This document explains how the span evaluation works in Presidio Evaluator, focusing on different overlap scenarios
between annotated and predicted spans.

With span evaluation, there could be multiple different scenarios of span overlaps: Overlap with zero spans, overlap
with one span, overlap with multiple spans. In addition, overlaps could occur with spans of the same entity type, spans
of different types, or one span overlapping with multiple spans from multiple types. This document will break down the
different overlap scenarios, and the expected aggregations on each scenario.

## Key Concepts

- **Span**: A continuous sequence of tokens representing an entity
- **IoU (Intersection over Union)**: Measures the overlap between spans
- **Threshold**: Minimum IoU value to consider a match

## Counting Rules (Two-Sided)

Counting is two-sided — the two metrics have different natural units, and each
unit is counted exactly once:

- **Recall is counted per annotation.** Every annotation gets exactly one
  verdict: TP if predictions of its own type cover it at IoU ≥ threshold
  (one span measured pairwise, several spans by their combined IoU), FN
  otherwise. Consequently `TP + FN == num_annotated`.
- **Precision is counted per prediction span.** Every span the model emitted
  enters `num_predicted` exactly once and is either *credited* (participated
  in at least one successful match) or an *FP*. Precision is
  `(num_predicted − FP) / num_predicted`.
- The two numerators may differ: one wide prediction covering two annotations
  is two recall hits (TP=2) but a single credited prediction
  (num_predicted=1). Predictions are **not consumed** by matching — a span can
  satisfy several annotations, and which annotation is evaluated first never
  changes any verdict.

## General Cases:

When comparing spans, the following general cases are considered:

1. ** High IoU, same type**: If the IoU is above a certain threshold (e.g., 0.75), the spans are considered a match.
2. ** High IoU, different type**: If the IoU is above the threshold but the types differ, it counts as a false positive
   for the predicted type and a false negative for the annotated type.
3. ** Low IoU**: If the IoU is below the threshold, it counts as *both* a false positive and a false negative,
   regardless of type.
4. **No Overlap**: If there is no overlap with a given annotation, it counts as a false negative for the annotation. If
   there is no overlap with a prediction, it counts as a false positive for the prediction.

## Basic Overlap Scenarios

### 1. Exact Match (Same Type, High IoU)

When a predicted span correctly matches an annotated span:

- **Example**:
    - Text: "John Smith visited Boston"
    - Annotation: [PERSON, PERSON, O, O]
    - Prediction: [PERSON, PERSON, O, O]
- **Result**: True Positive (TP)

### 2. Missing Entity (No Prediction)

When an annotated entity has no corresponding prediction:

- **Example**:
    - Text: "John Smith visited Boston"
    - Annotation: [PERSON, PERSON, O, LOCATION]
    - Prediction: [PERSON, PERSON, O, O]
- **Result**: False Negative (FN) for "Boston"

### 3. False Detection (No Annotation)

When a prediction has no corresponding annotation:

- **Example**:
    - Text: "The report was filed yesterday"
    - Annotation: [O, O, O, O, O]
    - Prediction: [O, ORGANIZATION, O, O, O]
- **Result**: False Positive (FP) for "report"

### 4. Type Mismatch (Different Type, High IoU)

When there's significant overlap but entity types differ:

- **Example**:
    - Text: "New York is a city"
    - Annotation: [LOCATION, LOCATION, O, O, O]
    - Prediction: [ORGANIZATION, ORGANIZATION, O, O, O]
- **Result**: Both FN for LOCATION and FP for ORGANIZATION

### 5. Partial Match (Low IoU)

When there's insufficient overlap between spans, they are treated as both false negatives and false positives:

- **Same Type Example (Scenario 5a)**:
    - Text: "John Smith Johnson visited"
    - Annotation: [PERSON, PERSON, PERSON, O]
    - Prediction: [PERSON, O, O, O]
    - IoU: 0.33 (below threshold of 0.75)
    - Result: Both FN (annotation not detected) and FP (prediction counted as separate entity)
    - num_predicted: +1 (prediction is counted)
    - Confusion matrix: (PERSON, O) and (O, PERSON)

- **Different Type Example (Scenario 5b)**:
    - Text: "New York Mets won"
    - Annotation: [ORGANIZATION, ORGANIZATION, ORGANIZATION, O]
    - Prediction: [LOCATION, O, O, O]
    - IoU: 0.2 (below threshold)
    - Result: FN for ORGANIZATION and FP for LOCATION
    - num_predicted: +1 (prediction is counted)
    - Confusion matrix: (ORGANIZATION, O) and (O, LOCATION)

## Multiple Span Scenarios

When an annotation overlaps with multiple prediction spans:

### 1. Multiple Spans of Same Type

Spans of the same type are combined, and their collective IoU is calculated.
The combined IoU decides the *annotation's* verdict; the spans themselves are
still counted individually on the precision side:

- **Example**:
    - Text: "New York Mets"
    - Annotation: [ORGANIZATION, ORGANIZATION, ORGANIZATION]
    - Prediction: [ORGANIZATION, O, ORGANIZATION]
    - Combined IoU = 0.67
    - If threshold = 0.5: Treated as a match — 1 TP; both spans are credited
      (num_predicted: +2, FP: 0)
    - If threshold = 0.75: Treated as a miss — 1 FN; each failed span is its
      own false positive (num_predicted: +2, FP: +2)

### 2. Multiple Spans of Different Types

Each entity type is evaluated separately against the annotation:

- **Example**:
    - Text: "John Smith Johnson"
    - Annotation: [PERSON, PERSON, PERSON]
    - Prediction: [PERSON, LOCATION, PERSON]
    - PERSON IoU = 0.67, LOCATION IoU = 0.33
    - If threshold = 0.5: PERSON is a match but wrong type for LOCATION portion
    - Result: TP for PERSON, FP for LOCATION

## One Prediction Spanning Multiple Annotations

The mirror image of the combined-IoU case: a single (usually too-wide)
prediction overlapping several annotations. Each annotation measures its own
pairwise IoU against that prediction independently; the prediction itself is
counted once.

### 1. Covers both annotations sufficiently (lenient thresholds)

- **Example**:
    - Text: "John Smith met Mary Jones"
    - Annotation: [PERSON, PERSON, O, PERSON, PERSON]
    - Prediction: [PERSON, PERSON, PERSON, PERSON, PERSON] (one span)
    - IoU vs each annotation ≈ 0.4
    - If threshold = 0.3: both annotations are covered — TP: 2;
      the span is credited once (num_predicted: +1, FP: 0)
    - Note: this case only exists at thresholds ≤ ~0.5, since two disjoint
      annotations can each occupy at most half of the covering span.

### 2. Covers neither annotation sufficiently (strict thresholds)

- Same example with threshold = 0.9: both annotations are missed — FN: 2;
  the span is one false positive, not two (num_predicted: +1, FP: +1).

### 3. Matches one annotation, swallows another

- **Example**:
    - Text: "John met Mary Jones Wilson Brown"
    - Annotation: [PERSON, O, PERSON, PERSON, PERSON, PERSON]
    - Prediction: one PERSON span over the whole text
    - IoU ≈ 0.7 vs the long annotation, ≈ 0.1 vs "John"
    - If threshold = 0.6: TP for the long annotation, FN for "John";
      the span is credited via the long match, so FP: 0 (num_predicted: +1)
    - The swallowed annotation costs recall once; the prediction is not
      additionally punished on precision.

## Confusion Matrix Convention

Each span — gold or predicted — appears in exactly one confusion-matrix cell:

- A matched annotation: (type, type).
- A missed annotation with no wrong-type detection over it: (type, "O").
- A wrong-type detection at ≥ threshold: a single (annotation type,
  predicted type) cell representing *both* the gold and the prediction —
  neither falls back to the "O" row/column.
- An uncredited prediction not represented by a wrong-entity cell:
  ("O", predicted type).

## Real-world Examples

### Example 1: Complex Name with Multiple Parts

- **Text**: "Dr. Jane Smith-Johnson, PhD."
- **Annotation**: [TITLE, PERSON, PERSON, PERSON, O, TITLE]
- **Prediction**: [TITLE, PERSON, PERSON, O, O, O]
- **Result**:
    - TP for TITLE "Dr."
    - TP for PERSON parts "Jane Smith" (combined IoU > threshold)
    - FN for "PhD." part

### Example 2: Address with Mixed Types

- **Text**: "123 Main Street, New York, NY 10001"
- **Annotation**: [ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS]
- **Prediction**: [ADDRESS, ADDRESS, ADDRESS, LOCATION, LOCATION, LOCATION, ADDRESS]
- **Result**:
    - If the combined IoU for ADDRESS is above threshold, it counts as TP, else FN
    - LOCATION is counted as a FP

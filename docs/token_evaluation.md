# Token Evaluation in Presidio Evaluator

This document explains the token-based evaluation process implemented in Presidio Evaluator, covering how it works, its
strengths and limitations, and how it compares to span-based evaluation.

## Overview

Token evaluation is a traditional approach to evaluating Named Entity Recognition (NER) models, where each token in the
text is evaluated individually against the ground truth annotation. This approach has been widely used in NER benchmarks
such as CoNLL.

## How Token Evaluation Works

### Basic Principle

In token evaluation, the text is segmented into tokens, and each token is assigned a label (entity type or "O" for
non-entity). The evaluation compares the predicted label for each token with its ground truth label.

### Token Evaluation Process

1. **Tokenization**: Text is split into individual tokens.
2. **Labeling**: Each token has a ground truth label and a predicted label.
3. **Comparison**: For each token, the predicted label is compared with the ground truth.
4. **Confusion Matrix**: Results are aggregated in a confusion matrix that counts matches and mismatches between entity
   types.
5. **Metric Calculation**: Precision, recall, and F-score are calculated based on the confusion matrix.

### Handling Different Tagging Schemes

The `TokenEvaluator` can handle different tagging schemes:

- **IO**: Simple Inside/Outside scheme (e.g., "PERSON" or "O")
- **BIO**: Beginning/Inside/Outside (e.g., "B-PERSON", "I-PERSON", "O")
- **BILUO**: Beginning/Inside/Last/Unit/Outside

The evaluator expects labels already in IO scheme. BIO/BILUO prefix stripping is performed by `CanonicalMapper` before evaluation.

## Metrics Calculation

The token evaluator calculates several metrics:

### Per-Entity Type Metrics

For each entity type (e.g., PERSON, LOCATION):

- **Precision**: Percentage of correctly predicted tokens of a given type out of all predictions of that type
- **Recall**: Percentage of correctly predicted tokens of a given type out of all actual tokens of that type
- **F-score**: Harmonic mean of precision and recall, with a configurable beta parameter

### Global PII Metrics

Aggregated across all entity types:

- **PII Precision**: Total true positives / Total predicted PII tokens
- **PII Recall**: Total true positives / Total annotated PII tokens
- **PII F-score**: Combined F-score for all entity types

## Strengths and Limitations

### Strengths

- **Simplicity**: Easy to understand and implement
- **Standard Approach**: Widely used in NER evaluation, making results comparable with other research
- **Token-level Insights**: Provides detailed information about which specific tokens are misclassified

### Limitations

- **Boundary Insensitivity**: May not accurately reflect entity boundary detection performance
- **Entity Fragmentation**: Multi-token entities can be partially correct but counted as multiple errors
- **All-or-Nothing**: No partial credit for almost-correct entity boundaries
- **Equal Weighting**: All tokens are weighted equally, regardless of importance

## Comparison with Span Evaluation

### When to Use Token Evaluation

Token evaluation is most appropriate when:

- Comparing with traditional NER benchmarks that use token-level evaluation
- You need to analyze performance at the token level for error analysis

## Implementation Details

The `TokenEvaluator` class in Presidio Evaluator:

1. Inherits from `BaseEvaluator` to leverage common functionality
2. Implements the `calculate_score` method to compute token-level metrics
3. Provides backward compatibility through the `Evaluator` alias (deprecated)

## Example Usage

```python
from presidio_evaluator.evaluation import TokenEvaluator

from presidio_evaluator.entity_mapping import CanonicalMapper

# Initialize the evaluator
evaluator = TokenEvaluator(entities_to_keep=["PERSON", "LOCATION", "ORGANIZATION"])

# 1. Get predictions as a DataFrame
results_df = my_model.predict_dataset(dataset)

# 2. Map entity types to the same namespace
mapper = CanonicalMapper()
mapped_df = mapper.get_mapped_results_dataframe(results_df)

# 3. Evaluate
final_result = evaluator.calculate_score_on_df(
    mapped_df,
    entities=["PERSON", "LOCATION"],
    beta=1.0  # F1 score
)

# Access metrics
print(f"Precision: {final_result.pii_precision}")
print(f"Recall: {final_result.pii_recall}")
print(f"F-score: {final_result.pii_f}")

# Access per-entity metrics
for entity, precision in final_result.entity_precision_dict.items():
    recall = final_result.entity_recall_dict[entity]
    print(f"{entity} - Precision: {precision}, Recall: {recall}")
```

## Related Resources

- [Evaluation Overview](evaluation.md) - General evaluation capabilities in Presidio Evaluator
- [Span Evaluation](span_evaluation.md) - Entity boundary-focused evaluation with IoU matching

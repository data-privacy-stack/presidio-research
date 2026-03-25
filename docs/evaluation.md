# Evaluation Capabilities in Presidio Evaluator

This document provides an overview of the evaluation capabilities in the Presidio Evaluator package, which helps assess
the performance of PII (Personally Identifiable Information) detection models.

## Overview

Presidio Evaluator offers a comprehensive framework for evaluating NER (Named Entity Recognition) models that identify
and classify PII entities in text.
The package supports multiple evaluation strategies to accommodate different use
cases and evaluation needs.

## Evaluation Components

The evaluation module consists of several key components:

### 1. Base Evaluator

The `BaseEvaluator` class provides the foundation for all evaluation strategies. It implements common functionality such
as:

- Comparing ground truth annotations with predictions (expects pre-mapped IO-format data from `CanonicalMapper`)
- Error detection and classification
- Calculation of global and class specific metrics (precision, recall, F-score)
- `calculate_score_on_df()` — the primary entry point, accepts a 5-column DataFrame

### 2. Evaluation Strategies

Presidio Evaluator supports two main evaluation strategies:

- **Token Evaluation**: Evaluates entity detection at the token level, comparing each predicted token label with its
  corresponding ground truth label.
- **Span Evaluation**: Evaluates entity detection at the entity span level, considering the boundaries and types of
  entire entities rather than individual tokens.

Each strategy has its own advantages and is suitable for different use cases.
Span evaluation is less prone to issues with multi-token entities and provides a more accurate representation of entity
boundaries,
but comes with several assumptions which might not hold for every case.
Token evaluation is simpler and widely used in traditional NER benchmarks.
For detailed information about each strategy, refer to:

- [Span Evaluation](span_evaluation.md)
- [Token Evaluation](token_evaluation.md)

It is possible to extend Presidio's evaluation capabilities by implementing custom evaluators that inherit from
`BaseEvaluator`.
For instance, one can create an evaluator that leverages LLMs (Large Language Models) as a judge to compare predicted
and actual entities.

### 3. Evaluation Results

The `EvaluationResult` class encapsulates the results of an evaluation, including:

- Confusion matrix between predicted and actual entity types (`EvaluationResult.results`)
- Precision, recall, and F-score metrics (both global (`EvaluationResult.pii_precision`, `EvaluationResult.pii_recall`,
  EvaluationResult.pii_f`, and per-entity-type `EvaluationResult.per_type`)
- Detailed error information for analysis, such as the error type (false positive, false negative, wrong entity type)
  and the context of the error.

#### Global vs. Per-Entity Metrics

Global metrics provide an overall view of model performance on detecting private entities in general,
while per-entity metrics break down performance by specific entity types (e.g., PERSON, LOCATION).
This allows for targeted analysis of model strengths and weaknesses.
In the global evaluation flow, the model would be measured on how well it detects any PII entity, regardless of type.
In the per-entity evaluation flow, the model would be measured on how well it detects each specific type of PII entity,
and would penalize the model for misclassifying entities of one type as another. Since there is often some overlap
between entities (e.g. ZIP_CODE and ADDRESS or TITLE and PERSON),
the metric values might seem lower than expected, but this is a more accurate representation of the model's performance.

### 4. Error Analysis

The evaluation framework provides robust error analysis capabilities through:

- Error type classification (false positives, false negatives, wrong entity type)
- Detailed error information (context, tokens, entity types)
- Visualization tools for error analysis

### 5. Visualization

The `Plotter` class provides visualization capabilities for evaluation results:

- Confusion matrices
- Performance metrics charts
- Common error visualizations

## Common Evaluation Workflows

### Basic End to End Evaluation Workflow

```python
# 1. Initialize an evaluator with a presidio instance
from typing import List

from presidio_analyzer import AnalyzerEngine
from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import SpanEvaluator, Plotter

dataset: List[InputSample] = [...]  # Load your dataset here
f_beta = 2
analyzer = AnalyzerEngine(default_score_threshold=0.3)

from presidio_evaluator.models import PresidioAnalyzerWrapper
from presidio_evaluator.entity_mapping import CanonicalMapper

model = PresidioAnalyzerWrapper(analyzer_engine=analyzer)
evaluator = SpanEvaluator(model=None)

# 2. Get predictions as a DataFrame
results_df = model.predict_dataset(dataset)

# 3. Map entity types to the same namespace
mapper = CanonicalMapper()
mapped_df = mapper.get_mapped_results_dataframe(results_df)

# 4. Evaluate
results = evaluator.calculate_score_on_df(mapped_df, beta=f_beta)

# 5. Extract confusion matrix and entities
entities, confmatrix = results.to_confusion_matrix()

# 6. Visualize results
plotter = Plotter(results=results,
                  model_name=model.name,
                  beta=f_beta)

plotter.plot_scores()
plotter.plot_confusion_matrix(entities=entities, confmatrix=confmatrix)
```

## Customization Options

The evaluation framework offers several customization options:

- **Entity Filtering**: Focus evaluation on specific entity types
- **Entity Mapping**: Use `CanonicalMapper.get_mapped_results_dataframe()` to resolve entity types to a shared canonical namespace before evaluation
- **Generic Entities**: Compare specific entities to generic entities that may not have specific types, like "ID" or "
  PII". (only available in token evaluation)
- **Skip Words**: Configure words to ignore during evaluation
- **IoU Threshold**: For span evaluation, set the threshold for entity boundary matches (only available in span
  evaluation)
- **Character vs. Token-based IoU**: For span evaluation, choose between character-level or token-level span
  intersection-over-union (only
  available in span evaluation))

## Evaluators Comparison

This section provides examples of different evaluation strategies applied to the same NER task, showing how results can
vary based on the evaluation methodology.

### Example Data

Consider this simple example:

```
Text: "United States of America"
Tokens: ["United", "States", "of", "America"]
True: [['B-LOC', 'I-LOC', 'O', 'I-LOC']]
Pred: [['B-LOC', 'I-LOC', 'I-LOC', 'I-LOC']]
```

In this example:

- The ground truth has "United States" and "America" as separate location entities
- The prediction treats the entire phrase "United States of America" as one location entity

### SemEval 2013 Evaluation

Using the `nervaluate` package which implements SemEval 2013 evaluation metrics:
source: https://github.com/MantisAI/nervaluate

```python
#pip install nervaluate
from nervaluate.evaluator import Evaluator

tokens = ["United", "States", "of", "America"]
true = [['B-LOC', 'I-LOC', 'O', 'I-LOC']]
pred = [['B-LOC', 'I-LOC', 'I-LOC', 'I-LOC']]

evaluator = Evaluator(true, pred, tags=['PER', 'ORG', 'LOC', 'DATE'], loader="list")
results = evaluator.evaluate()
print(results["overall"])
```

**Strict Evaluation (exact boundary match):**

```
- correct=0
- incorrect=1
- partial=0
- missed=0
- spurious=1
- precision=0.0000
- recall=0.0000
- f1=0.0000
```

**Partial Evaluation (partial boundary match):**

```
- correct=0
- incorrect=0
- partial=1
- missed=0
- spurious=1
- precision=0.2500
- recall=0.5000
- f1=0.3333
```

### CoNLL Evaluation

The CoNLL evaluation (used in CoNLL-2003 shared task) for the same example:
code (from https://github.com/sighsmile/conlleval):

```python
tokens = ["United", "States", "of", "America"]
true = ['B-LOC', 'I-LOC', 'I-LOC', 'I-LOC']

pred = ['B-LOC', 'I-LOC', 'I-LOC', 'I-LOC']

evaluate(true, pred)
```

**Output:**

```
processed 4 tokens with 2 phrases; found: 1 phrases; correct: 0.
accuracy: 100.00%; (non-O)
accuracy:  75.00%; precision:   0.00%; recall:   0.00%; FB1:   0.00
              LOC: precision:   0.00%; recall:   0.00%; FB1:   0.00
```

### Presidio Token Evaluator

Using Presidio's token evaluator without skip words configuration:

```python
from presidio_evaluator.evaluation import EvaluationResult, TokenEvaluator

# Create a sample
evaluation_result = EvaluationResult(
    tokens=["United", "States", "of", "America"],
    actual_tags=["B-LOC", "I-LOC", "O", "B-LOC"],
    predicted_tags=["I-LOC", "I-LOC", "I-LOC", "I-LOC"],
    start_indices=[0, 7, 14, 17]
)

# Initialize evaluator and evaluate
evaluator = TokenEvaluator(model=None, skip_words=[])
final_result = evaluator.calculate_score([evaluation_result])

print(f"Precision: {final_result.pii_precision:.4f}")
print(f"Recall: {final_result.pii_recall:.4f}")
print(f"F1: {final_result.pii_f:.4f}")
```

**Output:**

```
Precision: 0.7500
Recall: 1.0000
F1: 0.8571
```

### Presidio Token Evaluator (with skip words)

Using Presidio's token evaluator with "of" configured as a skip word:

```python
# Initialize token evaluator with "of" as a skip word
from presidio_evaluator.evaluation import EvaluationResult, TokenEvaluator

# Create a sample
evaluation_result = EvaluationResult(
    tokens=["United", "States", "of", "America"],
    actual_tags=["B-LOC", "I-LOC", "O", "B-LOC"],
    predicted_tags=["I-LOC", "I-LOC", "I-LOC", "I-LOC"],
    start_indices=[0, 7, 14, 17]
)

# Initialize evaluator and evaluate
evaluator = TokenEvaluator(model=None)
final_result = evaluator.calculate_score([evaluation_result])

print(f"Precision: {final_result.pii_precision:.4f}")
print(f"Recall: {final_result.pii_recall:.4f}")
print(f"F1: {final_result.pii_f:.4f}")
```

**Output:**

```
Precision: 1.0000
Recall: 1.0000
F1: 1.0000
```

With "of" as a skip word, the token evaluator ignores this token in the evaluation, resulting in perfect precision and
recall.

### Presidio Span Evaluator (without skip words)

Using Presidio's span evaluator with no skip words:

```python
from presidio_evaluator.evaluation import SpanEvaluator, EvaluationResult

# Same setup as before
evaluation_result = EvaluationResult(
    tokens=["United", "States", "of", "America"],
    actual_tags=["B-LOC", "I-LOC", "O", "B-LOC"],
    predicted_tags=["I-LOC", "I-LOC", "I-LOC", "I-LOC"],
    start_indices=[0, 7, 14, 17]
)

# Initialize span evaluator with no skip words
evaluator = SpanEvaluator(iou_threshold=0.5, model=None, skip_words=[])
scores = evaluator.calculate_score([evaluation_result])

print(f"Precision: {scores.pii_precision:.4f}")
print(f"Recall: {scores.pii_recall:.4f}")
print(f"F1: {scores.pii_f:.4f}")
```

**Output:**

```
Precision: 1.0 # IoU is higher than the threshold
Recall: 0.5 # The IoU between "America" and "United States of America" is less than the threshold, so it's a FN.
F2: 0.55556
```

The span evaluator is stricter about entity boundaries, treating "United States of America" as a different entity than
the separate "United States" and "America" entities.

### Presidio Span Evaluator (with skip words)

Now with "of" as a skip word:

```python
from presidio_evaluator.evaluation import SpanEvaluator, EvaluationResult

# Same setup as before
evaluation_result = EvaluationResult(
    tokens=["United", "States", "of", "America"],
    actual_tags=["B-LOC", "I-LOC", "O", "B-LOC"],
    predicted_tags=["I-LOC", "I-LOC", "I-LOC", "I-LOC"],
    start_indices=[0, 7, 14, 17]
)

# Initialize span evaluator with no skip words
evaluator = SpanEvaluator(iou_threshold=0.5, model=None)
scores = evaluator.calculate_score([evaluation_result])

print(f"Precision: {scores.pii_precision:.4f}")
print(f"Recall: {scores.pii_recall:.4f}")
print(f"F1: {scores.pii_f:.4f}")
```

**Output:**

```
Precision: 1.000 
Recall: 1.0000
F2: 1.000
```

With "of" as a skip word, the span evaluator can now merge "United States" and "America" into one entity, resulting in a
perfect match with the prediction.

### Key Takeaways

- Use **token evaluation** when individual token classification is most important
- Use **span evaluation** if tokenization has a negative effect on results
  and when evaluation should be done on entire PII spans rather than individual tokens
- Configure **skip words** when certain connecting words should not affect the comparison of predicted and actual
  entities.
- Adjust the **IoU threshold** in span evaluation to control the strictness of intersection-over-union matching.

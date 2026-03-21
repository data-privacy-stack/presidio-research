# ADR-001: Simplified Evaluation Pipeline Using DataFrame as Clean Interface

## Status

Proposed

## Date

2026-03-21

## Context

The current evaluation pipeline in `presidio_evaluator` is tightly coupled and requires several intermediate steps to go from a dataset and a model to evaluation metrics. The full flow today is:

```
Dataset (List[InputSample])
    → Evaluator(model=model)            # model is embedded in evaluator
    → evaluator.evaluate_all(dataset)   # runs inference AND builds per-sample EvaluationResult objects
    → List[EvaluationResult]            # intermediate per-sample carriers
    → evaluator.calculate_score(...)    # calls get_results_dataframe() → DataFrame → scoring
    → EvaluationResult (aggregated)
```

This design has four concrete pain points:

1. **Model is coupled to the Evaluator** — `BaseEvaluator.__init__` takes a `model` argument, and `evaluate_all()` calls `model.batch_predict`, `model.filter_tags_in_supported_entities`, and `model.to_scheme` internally. This makes it impossible to evaluate a pre-computed result set without also instantiating a model.

2. **`evaluate_all()` does two things** — it runs model inference AND builds per-sample `EvaluationResult` objects. These objects are simple data carriers holding `(tokens, actual_tags, predicted_tags, start_indices)`, yet they require callers to go through the evaluator just to get predictions into a usable shape.

3. **The real interface already exists but is buried** — `SpanEvaluator.calculate_score()` and `TokenEvaluator.calculate_score()` internally call `get_results_dataframe()` to convert `List[EvaluationResult]` to a DataFrame, and then call `calculate_score_on_df()`. The `EvaluationResult` list is a wasteful intermediate step; the DataFrame is the actual computational surface.

4. **Entity mapping logic is scattered** — entity type remapping lives in `BaseModel.align_entity_types`, `BaseEvaluator.align_entity_types` (static method), and is applied independently in both models and evaluators. There is no single, composable entry point for this transformation.

A new contributor trying to run an evaluation must understand all of these layers before being able to get a single metric out of the system. The goal of this ADR is to establish a clean, linear pipeline with a single well-defined interface point between the model and the evaluator.

## Decision

Adopt the **results DataFrame** as the canonical interface boundary between the model stage and the evaluation stage. The DataFrame has the following five-column schema:

| Column | Type | Description |
|---|---|---|
| `sentence_id` | `int` | Index of the source sentence in the dataset |
| `token` | `str` | Token string |
| `annotation` | `str` | Ground-truth entity tag (from `InputSample.tags`) |
| `prediction` | `str` | Model-predicted entity tag |
| `start_indices` | `int` | Character start offset of the token within the sentence |

This schema is already produced by `get_results_dataframe()` and consumed by `calculate_score_on_df()`. The changes below make it the first-class output of the model stage instead of an internal implementation detail of the evaluator.

### Proposed simplified pipeline (5 steps)

```python
# 1. Load dataset
dataset = InputSample.read_dataset_json("data/dataset.json")

# 2. Choose model and run predictions → get DataFrame directly
model = SpacyModel(model=nlp)  # or PresidioAnalyzerWrapper, etc.
results_df = model.predict_dataset(dataset)  # NEW: returns the DataFrame directly

# 3. Map entities (optional, operates on DataFrame)
results_df = map_entities(results_df, mapping={"GPE": "LOCATION", "LOC": "LOCATION"})

# 4. Score (evaluator is model-free, takes only the DataFrame)
evaluator = SpanEvaluator()   # no model argument needed!
results = evaluator.calculate_score_on_df(per_type=True, results_df=results_df)

# 5. Analyze/plot
plotter = Plotter(results=results)
plotter.plot_scores()
```

### Concrete changes

| Component | Current | Proposed |
|---|---|---|
| `BaseModel` | `predict()` → `List[str]` tags per sample | Add `predict_dataset()` → `pd.DataFrame` with the 5-column schema |
| `BaseEvaluator` | `__init__(model=...)`, `evaluate_all(dataset)` | `model` becomes optional; primary entry point becomes `calculate_score_on_df()` |
| Entity mapping | Scattered across `BaseModel`, `BaseEvaluator`, and `InputSample` | Single `map_entities(df, mapping)` utility operating on the DataFrame |
| `evaluate_all()` | Required step in the pipeline | Becomes a convenience wrapper (calls `predict_dataset` then `calculate_score_on_df`) |
| `EvaluationResult` (per-sample) | Carrier object for tokens/tags | Eliminated — the DataFrame IS the carrier |

### `predict_dataset` method on `BaseModel`

```python
def predict_dataset(self, dataset: List[InputSample], **kwargs) -> pd.DataFrame:
    """Run model on dataset and return a standardized results DataFrame."""
    predictions = self.batch_predict(dataset, **kwargs)
    rows = []
    for i, (sample, prediction) in enumerate(zip(dataset, predictions)):
        prediction = self.filter_tags_in_supported_entities(prediction)
        prediction = self.to_scheme(prediction)
        for token, tag, pred, start in zip(
            sample.tokens, sample.tags, prediction, sample.start_indices
        ):
            rows.append({
                "sentence_id": i,
                "token": str(token),
                "annotation": tag,
                "prediction": pred,
                "start_indices": start,
            })
    return pd.DataFrame(rows)
```

### `map_entities` utility function

```python
def map_entities(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Map entity types in both annotation and prediction columns."""
    df = df.copy()
    df["annotation"] = df["annotation"].replace(mapping)
    df["prediction"] = df["prediction"].replace(mapping)
    return df
```

## Consequences

### Positive

- **Reduced conceptual overhead** — new contributors only need to understand `predict_dataset` + `calculate_score_on_df`; the full evaluator internals are optional knowledge.
- **Decoupled model and evaluator** — models and evaluators can be developed, tested, and swapped independently. `SpanEvaluator` and `TokenEvaluator` no longer require a `model` instance.
- **Composable entity mapping** — `map_entities` is a pure function on DataFrames; it can be chained, applied selectively, or omitted without touching model or evaluator configuration.
- **Easier offline/batch evaluation** — a results DataFrame can be saved to disk and re-evaluated later without re-running inference. This is currently impossible without serializing `List[EvaluationResult]` objects.
- **`calculate_score_on_df` already exists and is tested** — `SpanEvaluator.calculate_score_on_df` and `TokenEvaluator.calculate_score_on_df` are already implemented and covered by tests in `test_span_evaluator.py`. The new pipeline plugs directly into existing code.
- **Eliminates the `EvaluationResult` per-sample carrier** — removes a class whose only purpose is to ferry data between `evaluate_all()` and `get_results_dataframe()`.

### Negative / Trade-offs

- **`predict_dataset` materializes all predictions in memory** — for very large datasets, the full DataFrame is held in RAM. The current `evaluate_all()` loop can in principle be streamed (though it isn't today).
- **`evaluate_all()` becomes a thin wrapper** — code that currently calls `evaluate_all()` and inspects `List[EvaluationResult]` directly will need to be updated if it relies on the per-sample `EvaluationResult` structure. Backward compatibility is preserved at the call site, but the internal representation changes.
- **Schema enforcement is implicit** — the 5-column schema is a convention, not enforced by a type. Callers that hand-construct DataFrames must respect column names and types.
- **`EvaluationResult` (per-sample) removal is a breaking change** — any downstream code (e.g., notebooks, scripts) that imports or type-annotates `EvaluationResult` will need updating. A deprecation period should be provided.

## Implementation Plan

1. **Add `BaseModel.predict_dataset()`** — implement the method as sketched above in `presidio_evaluator/models/base_model.py`. Add a unit test in `tests/` that verifies the 5-column schema and correct row count for a small synthetic dataset.

2. **Add `map_entities()` utility** — add the function (and `Dict` import) to `presidio_evaluator/evaluation/` (e.g., in a new `utils.py` or alongside `get_results_dataframe`). Add a unit test verifying that both `annotation` and `prediction` columns are remapped.

3. **Make `model` optional in `BaseEvaluator`** — change `BaseEvaluator.__init__(self, model, ...)` so that `model` defaults to `None`. Add a runtime check that raises a clear error if `evaluate_all()` is called when `model is None`.

4. **Update `evaluate_all()` to delegate to `predict_dataset` + `calculate_score_on_df`** — refactor `SpanEvaluator.evaluate_all()` and `TokenEvaluator.evaluate_all()` to call `self.model.predict_dataset(dataset)` and then pass the result to `calculate_score_on_df()`. This ensures a single code path for both old and new usage.

5. **Deprecate per-sample `EvaluationResult`** — add a `DeprecationWarning` to the `EvaluationResult` class (the per-sample variant) and update `get_results_dataframe()` to note it will be removed in a future release.

6. **Update documentation and notebooks** — revise `docs/evaluation.md` and any Jupyter notebooks under `notebooks/` to demonstrate the new 5-step pipeline. Keep the old `evaluate_all()` example with a note that it is the backward-compatible convenience path.

7. **Add integration test** — add an end-to-end test that runs `predict_dataset` → `map_entities` → `calculate_score_on_df` on a small synthetic dataset and asserts that the returned metrics match expected values.

## Alternatives Considered

### Keep the current architecture and add a thin adapter function

A helper function like `run_evaluation(model, dataset, evaluator)` could hide the complexity without changing any classes. This was rejected because it does not address the root coupling: `evaluate_all()` would still conflate inference with result formatting, and entity mapping would remain scattered.

### Use a custom result object instead of a DataFrame

A typed dataclass or `NamedTuple` (e.g., `TokenPrediction`) could replace the per-sample `EvaluationResult`. This was rejected because `calculate_score_on_df` already exists and is the correct computational surface. Introducing another intermediate type would add a conversion step rather than remove one.

### Make `BaseModel` return a list of tagged spans instead of a DataFrame

Returning `List[List[Tuple[str, str, str, int]]]` (one list per sentence) would preserve laziness but lose the benefits of columnar operations and Pandas compatibility. Rejected because the DataFrame is already the internal format used for all scoring, and forcing a conversion at the last moment adds no value.

### Move entity mapping into `BaseModel.predict_dataset` as a parameter

Passing `mapping` directly to `predict_dataset(dataset, entity_mapping=...)` would keep mapping close to prediction. This was rejected in favor of a standalone `map_entities` utility because it makes the transformation composable and testable in isolation, and because mapping is a property of the evaluation goal (what entities to score), not of the model.

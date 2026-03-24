# Migration Guide: Presidio Evaluator Redesign

This guide helps you migrate from the previous version of `presidio-evaluator` to the new three-layer architecture.

## Overview of Changes

The evaluation pipeline now has three explicit layers:

```
Model → CanonicalMapper → Evaluator
```

1. **Model** (`BaseModel.predict_dataset()`) — runs the model and returns a raw-predictions DataFrame.
2. **CanonicalMapper** (`CanonicalMapper.get_mapped_results_dataframe()`) — aligns entity labels between the dataset and the model.
3. **Evaluator** (`SpanEvaluator.calculate_score_on_df()` / `TokenEvaluator.calculate_score_on_df()`) — scores the mapped predictions.

---

## Quick Migration Example (SpanEvaluator)

### Before

```python
from presidio_evaluator.evaluation import SpanEvaluator

evaluator = SpanEvaluator(
    model=wrapped_analyzer,
    entity_mapping={"STREET_ADDRESS": "LOCATION"},
    iou_threshold=0.75,
)
results = evaluator.evaluate_all(dataset)
scores = evaluator.calculate_score(results)
```

### After

```python
from presidio_evaluator.entity_mapping import CanonicalMapper
from presidio_evaluator.evaluation import SpanEvaluator

# Step 1: Predict
results_df = wrapped_analyzer.predict_dataset(dataset)

# Step 2: Map
mapper = CanonicalMapper.from_dataset(dataset)
mapped_df = mapper.get_mapped_results_dataframe(results_df)

# Step 3: Score
entities_to_keep = {v for v in mapper.get_mapping().values() if v is not None}
evaluator = SpanEvaluator(
    model=wrapped_analyzer,
    entities_to_keep=list(entities_to_keep),
    iou_threshold=0.75,
)
scores = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)
```

---

## Quick Migration Example (TokenEvaluator)

### Before

```python
from presidio_evaluator.evaluation import TokenEvaluator

evaluator = TokenEvaluator(
    model=my_model,
    entity_mapping={"PERSON": "PERSON"},
)
results = evaluator.evaluate_all(dataset)
scores = evaluator.calculate_score(results)
```

### After

```python
from presidio_evaluator.entity_mapping import CanonicalMapper
from presidio_evaluator.evaluation import TokenEvaluator

results_df = my_model.predict_dataset(dataset)
mapper = CanonicalMapper.from_dataset(dataset)
mapped_df = mapper.get_mapped_results_dataframe(results_df)

evaluator = TokenEvaluator(model=my_model)
scores = evaluator.calculate_score_on_df(results_df=mapped_df)
```

---

## Removed / Renamed API

| Old | New | Notes |
|-----|-----|-------|
| `evaluator.evaluate_all(dataset)` | `model.predict_dataset(dataset)` + `calculate_score_on_df()` | Three-step pipeline |
| `SpanEvaluator(entity_mapping=...)` | `CanonicalMapper.get_mapped_results_dataframe()` | Mapping moved to mapper |
| `TokenEvaluator(entity_mapping=...)` | `CanonicalMapper.get_mapped_results_dataframe()` | Mapping moved to mapper |
| `evaluator.get_results_dataframe()` | `model.predict_dataset()` | Returns same 5-column DataFrame |
| `BaseEvaluator(compare_by_io=True)` | default behaviour (IO only) | BIO stripping moved to mapper |
| `EntityMappingHelper` | `CanonicalMapper` | Richer auto-resolution |
| `BaseEvaluator.from_dataset()` | removed | Use `predict_dataset()` directly |
| `FlairModel`, `SpacyModel`, `StanzaModel`, `AzureAITextAnalyticsWrapper` | removed | Use Presidio wrappers instead |

---

## Handling `DeprecationError` from `evaluate_all()`

If you call `evaluate_all()` you will get a hard `DeprecationError` at runtime:

```
DeprecationError: evaluate_all() has been removed. Use the three-step pipeline instead:
    results_df = model.predict_dataset(dataset)
    mapped_df  = mapper.get_mapped_results_dataframe(results_df)
    scores     = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)
```

**Fix:** Follow the three-step pipeline shown above.

---

## Handling `DeprecationWarning` from `get_results_dataframe()`

Calling `evaluator.get_results_dataframe()` emits a soft `DeprecationWarning`:

```
DeprecationWarning: get_results_dataframe() is deprecated. Use model.predict_dataset() instead.
```

**Fix:** Replace `evaluator.get_results_dataframe(dataset)` with `model.predict_dataset(dataset)`.

---

## Entity Mapping with CanonicalMapper

`CanonicalMapper` replaces `EntityMappingHelper`. It auto-resolves labels through four tiers:

| Tier | Description |
|------|-------------|
| `EXACT` | Label is already a canonical name or known alias |
| `COUNTRY` | Label has a country prefix (e.g. `GERMANY_PASSPORT_NUMBER` → `PASSPORT`) |
| `FUZZY` | High-similarity string match to a known alias |
| `PENDING` | No automatic match — requires `.map()` |

```python
from presidio_evaluator.entity_mapping import CanonicalMapper

mapper = CanonicalMapper.from_dataset(dataset)

# Review auto-resolution (HTML table in Jupyter)
mapper.render_html()

# Manually resolve pending labels
mapper.map({
    "MY_ENTITY": "PERSON",   # resolve
    "NOISE": None,            # suppress from evaluation
})

# Get final mapping dict
entity_mapping = mapper.get_mapping()
```

---

## Package Manager: Poetry → uv

The project now uses [uv](https://docs.astral.sh/uv/) instead of Poetry.

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -m "not integration"

# Run linter
uv run ruff check
```

---

## Python Version Support

Minimum Python version is now **3.11** (previously 3.10). This is required by `numpy >= 2.4.0`.

---

## Test Directory Changes

Tests have been reorganised into topic-based subdirectories:

| Old path | New path |
|----------|----------|
| `tests/test_faker_sentences.py` | `tests/data_generator/` |
| `tests/test_canonical_mapper.py` | `tests/entity_mapping/` |
| `tests/test_evaluator.py` | `tests/evaluation/` |
| `tests/test_base_model.py` | `tests/models/` |
| `tests/test_presidio_analyzer_wrapper.py` | `tests/integration/` |
| `tests/test_data_objects.py` | `tests/integration/` |

Integration tests are tagged with `pytest.mark.integration` and can be excluded:

```bash
pytest -m "not integration"
```

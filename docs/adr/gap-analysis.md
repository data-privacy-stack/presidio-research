# Gap Analysis: ADR-001 & ADR-002 vs. Current Implementation

## Summary

The codebase has **partially implemented ADR-002** (CanonicalMapper / EntityHierarchy) but has **not started ADR-001** (simplified DataFrame pipeline). The entity mapping infrastructure is largely in place; the evaluation pipeline refactoring is the main remaining work.

---

## ADR-002: Entity Mapping via CanonicalMapper

### ✅ Implemented

| ADR-002 Requirement | Current State | Location |
|---|---|---|
| `CanonicalMapper` class | Implemented with full resolution pipeline | `entity_mapping/mapper.py` |
| `EntityHierarchy` class | Implemented with `canonicalize()`, `get_branch()`, aliases | `entity_mapping/mapper.py` |
| `HIERARCHY` dictionary | Defined with full nested taxonomy | `entity_mapping/definitions.py` |
| Resolution tiers: EXACT, COUNTRY, COUNTRY_FALLBACK, FUZZY, PENDING | All tiers implemented in `_auto_resolve_one()` | `entity_mapping/mapper.py` |
| BIO/BIOES prefix stripping | `_strip_bio()` strips prefixes before resolution | `entity_mapping/mapper.py` |
| `from_dataset()` constructor | Extracts labels from `InputSample.spans`, auto-resolves | `entity_mapping/mapper.py` |
| Input is results DataFrame | ADR-002 now specifies the mapper takes a results DataFrame (sentence_id, token, annotation, prediction) | See updated workflow below |
| `map()` for manual resolution | Implemented with validation (atomic batch updates) | `entity_mapping/mapper.py` |
| `get_mapping()` returns `dict[str, str \| None]` | Implemented | `entity_mapping/mapper.py` |
| `IncompleteMapping` exception | Defined and raised when pending labels remain | `entity_mapping/mapper.py` |
| Logging at INFO/WARNING levels | Implemented per the ADR spec | `entity_mapping/mapper.py` |
| Country prefix handling with `COUNTRIES` set | `COUNTRIES` defined in `definitions.py` | `entity_mapping/definitions.py` |
| Tests for CanonicalMapper | Dedup, auto-resolve, manual override, BIO strip | `tests/test_canonical_mapper.py` |
| Tests for EntityHierarchy | Canonicalize, get_branch, normalization | `tests/test_entity_hierarchy.py` |
| Label coverage tests | Wide set of HF/vendor labels tested | `tests/test_label_coverage.py` |
| `entity_mapping` removed from BaseModel | Raises `ValueError` if passed (deprecated) | `models/base_model.py` |
| `entity_mapping` required in BaseEvaluator | Required param, validated as non-None | `evaluation/base_evaluator.py` |
| `_normalize_entity_for_comparison()` | Uses mapping to translate dataset→model entities | `evaluation/base_evaluator.py` |

### ⚠️ Partially Implemented / Gaps

| ADR-002 Requirement | Gap | Notes |
|---|---|---|
| `get_mapping(mode='html')` — HTML audit table | **Not verified** — ADR mentions `render_html()` for transparency | Need to check if HTML rendering exists in `CanonicalMapper` |
| `resolve_interactively()` — guided manual resolution | **Not verified** — ADR mentions ranked fuzzy suggestions | Need to check if interactive mode exists |
| `from_results_data_frame(results_df)` constructor | **Does not exist.** ADR-002 now specifies the mapper is constructed from a results DataFrame, extracting labels from both `annotation` and `prediction` columns. Currently only `from_dataset()` (from `InputSample` list) exists | New factory method needed |
| `get_mapped_results_dataframe()` method | **Does not exist.** ADR-002 specifies this returns a new DataFrame with both columns remapped to canonical entities | New method needed |
| `from_model(label_extractor)` constructor | **Removed from ADR-002** — no longer in the spec. `from_results_data_frame` is the primary entry point | N/A — dropped requirement |

### ✅ ADR-002 Verdict: ~75% implemented. Core resolution infrastructure is solid, but the newly specified DataFrame-based entry point (`from_results_data_frame`) and output method (`get_mapped_results_dataframe`) are missing. These bridge the mapper directly into the ADR-001 pipeline.

---

## ADR-001: Simplified Evaluation Pipeline

### Current Pipeline (what exists today)

```
Dataset (List[InputSample])
    → Evaluator(model=model, entity_mapping={...})
    → evaluator.evaluate_all(dataset)          # runs inference + builds List[EvaluationResult]
    → List[EvaluationResult]                   # per-sample carriers
    → evaluator.calculate_score(eval_results)  # internally: get_results_dataframe() → calculate_score_on_df()
    → EvaluationResult (aggregated)
```

### Target Pipeline (what ADR-001 proposes)

**Simple (single hierarchy level):**
```python
# 1. Load dataset
dataset = InputSample.read_dataset_json("data/dataset.json")

# 2. Choose model and run predictions → get DataFrame directly
model = PresidioAnalyzerWrapper(analyzer_engine=AnalyzerEngine())
results_df = model.predict_dataset(dataset)           # NEW

# 3. Map entities
mapper = CanonicalMapper.from_results_data_frame(results_df)  # NEW: takes the DataFrame
results_df_mapped = mapper.get_mapped_results_dataframe()      # NEW: returns mapped DataFrame

# 4. Evaluate
evaluator = SpanEvaluator()
results = evaluator.calculate_score_on_df(results_df=results_df_mapped)

# 5. Analyze/plot
plotter = Plotter(results=results)
plotter.plot_scores()
```

**Multi-hierarchy (evaluate at different granularities):**
```python
# 1-2. Same as above
results_df = model.predict_dataset(dataset)
mapper = CanonicalMapper()

# 3-4. Loop over hierarchy levels and evaluate each
evaluator = SpanEvaluator()
results_per_hierarchy = []
for hierarchy in [1, 2, 3]:
    results_df_hierarchy = mapper.map_entities(results_df, hierarchy=hierarchy)
    results_per_hierarchy.append(evaluator.calculate_score_on_df(results_df=results_df_hierarchy))

# 5. Analyze/plot
plotter = Plotter(results=results_per_hierarchy[0])
plotter.plot_scores()
```

### Gap-by-Gap Breakdown

| # | ADR-001 Change | Current State | Status | Work Required |
|---|---|---|---|---|
| 1 | `BaseModel.predict_dataset()` → returns 5-column DataFrame | **Does not exist.** Models only have `predict()` → `List[str]` and `batch_predict()` → `List[List[str]]` | 🔴 Not started | Add method to `BaseModel` that calls `batch_predict()` and assembles the DataFrame with columns `(sentence_id, token, annotation, prediction, start_indices)` |
| 2 | `model` becomes optional in `BaseEvaluator.__init__` | Model is already optional — `model=None` is handled with a warning | ✅ Done | None — already implemented |
| 3 | `calculate_score_on_df()` as the primary entry point | **Exists for SpanEvaluator** (`SpanEvaluator.calculate_score_on_df()`). **Does NOT exist for TokenEvaluator** — `TokenEvaluator.calculate_score()` works on `Counter` objects from `EvaluationResult.results` | 🟡 Partial | Add `TokenEvaluator.calculate_score_on_df()` to match SpanEvaluator's interface |
| 4 | `evaluate_all()` delegates to `predict_dataset()` + `calculate_score_on_df()` | `evaluate_all()` still calls `batch_predict()` in a loop, builds `EvaluationResult` per sample, then passes to `calculate_score()` | 🔴 Not started | Refactor `evaluate_all()` to use `predict_dataset()` → `calculate_score_on_df()` internally |
| 5 | `CanonicalMapper.from_results_data_frame(results_df)` — construct mapper from DataFrame | **Does not exist.** Currently only `from_dataset(samples)` exists, which takes `List[InputSample]`. ADR-002 now specifies constructing from the results DataFrame, extracting unique labels from both `annotation` and `prediction` columns | 🔴 Not started | Add factory method that extracts labels from both DataFrame columns and delegates to `__init__` |
| 6 | `CanonicalMapper.get_mapped_results_dataframe()` — return mapped DataFrame | **Does not exist.** ADR-002 specifies this returns a new DataFrame with both `annotation` and `prediction` columns remapped to canonical entities using the resolved mapping | 🔴 Not started | Add method that applies `get_mapping()` dict to both columns of the stored DataFrame, mapping `None` values to `"O"` |
| 6b | `mapper.map_entities(results_df, hierarchy=N)` — hierarchical mapping | **Does not exist.** ADR-001 shows a multi-hierarchy loop where the mapper remaps entities at different granularity levels (1=PII, 2=PERSON/CONTACT/etc., 3=NAME/EMAIL/etc.). `EntityHierarchy.get_branch()` exists but there's no `map_entities()` method that accepts a `hierarchy` parameter and remaps DataFrame columns to the requested level | 🔴 Not started | Add `map_entities(results_df, hierarchy)` method that uses `get_branch()` to remap entities at the requested hierarchy level. This is complementary to `get_mapped_results_dataframe()` — the latter maps to canonical (level 3), while `map_entities` maps to any level |
| 7 | Deprecate per-sample `EvaluationResult` usage | `EvaluationResult` is still used as both per-sample carrier AND aggregated result (dual purpose). `evaluate_sample()`, `evaluate_all()`, and `get_results_dataframe()` all depend on `List[EvaluationResult]` | 🔴 Not started | Add `DeprecationWarning` to per-sample usage paths; document the DataFrame-based alternative |
| 8 | Per-sample `EvaluationResult` fields eliminated | `EvaluationResult` still has `tokens`, `actual_tags`, `predicted_tags`, `start_indices` fields used only in per-sample mode | 🟡 Future | Not blocking — can be deprecated first, removed later |
| 9 | Decouple SpanEvaluator from model | `SpanEvaluator(model=None)` works but `calculate_score_on_df()` still requires going through `calculate_score(evaluation_results)` → `get_results_dataframe()` | 🟡 Partial | `calculate_score_on_df()` can already be called directly, but the entity mapping normalization is done in `get_results_dataframe()`, so calling `calculate_score_on_df()` directly skips mapping |
| 10 | Documentation & notebooks updated | Notebooks still use the old `evaluate_all()` → `calculate_score()` pattern | 🔴 Not started | Update notebooks 4, 5, 6 to show the new 5-step pipeline |

---

## Integration Gap: Connecting CanonicalMapper to the Evaluation Pipeline

The updated ADRs have **resolved the ambiguity** about where mapping lives. The bridge between ADR-001 and ADR-002 is now clearly defined as two `CanonicalMapper` methods:

1. **`CanonicalMapper.from_results_data_frame(results_df)`** — constructs the mapper from the DataFrame produced by `predict_dataset()`, extracting labels from both `annotation` and `prediction` columns.
2. **`CanonicalMapper.get_mapped_results_dataframe()`** — returns a new DataFrame with both columns remapped to canonical entities.

This replaces the earlier ambiguity between a standalone `map_entities()` function vs. a mapper method. The mapper owns state (pending resolutions, manual overrides), so it makes sense for the mapping to be a method on the mapper rather than a pure function.

### What's needed:

```python
# New factory method on CanonicalMapper
@classmethod
def from_results_data_frame(cls, results_df: pd.DataFrame, **kwargs) -> "CanonicalMapper":
    """Extract unique labels from annotation + prediction columns and auto-resolve."""
    labels = set(results_df["annotation"].unique()) | set(results_df["prediction"].unique())
    labels.discard("O")
    mapper = cls(list(labels), **kwargs)
    mapper._results_df = results_df  # store reference for get_mapped_results_dataframe
    return mapper

# New method on CanonicalMapper
def get_mapped_results_dataframe(self) -> pd.DataFrame:
    """Return DataFrame with annotation and prediction columns remapped to canonical entities."""
    mapping = self.get_mapping()
    df = self._results_df.copy()
    def _remap(val):
        if val == "O":
            return "O"
        mapped = mapping.get(val, val)
        return "O" if mapped is None else mapped
    df["annotation"] = df["annotation"].map(_remap)
    df["prediction"] = df["prediction"].map(_remap)
    return df
```

---

## Priority Order for Implementation

| Priority | Item | Rationale |
|---|---|---|
| **P0** | `BaseModel.predict_dataset()` | Foundation for the new pipeline; all other changes depend on this |
| **P0** | `CanonicalMapper.from_results_data_frame()` | Entry point for the mapper in the new pipeline; extracts labels from DataFrame |
| **P0** | `CanonicalMapper.get_mapped_results_dataframe()` | Produces the mapped DataFrame that feeds into `calculate_score_on_df()` |
| **P1** | `mapper.map_entities(results_df, hierarchy=N)` | Enables multi-hierarchy evaluation (PII/high-level/canonical); uses existing `get_branch()` |
| **P1** | `TokenEvaluator.calculate_score_on_df()` | Needed for parity with SpanEvaluator; completes the DataFrame-based interface |
| **P1** | Refactor `evaluate_all()` to use new primitives | Makes the old API delegate to the new one internally |
| **P2** | Deprecation warnings on per-sample `EvaluationResult` | Signal intent to remove; doesn't break anything |
| **P3** | Update notebooks & documentation | Show the new 5-step pipeline |
| **P3** | Verify `CanonicalMapper` HTML rendering & interactive mode | Complete ADR-002 feature coverage |

---

## Questions for Review

1. **Should `predict_dataset()` handle entity mapping?** ADR-001 says no (mapping is a property of evaluation, not the model). Confirm this is still the desired separation.

2. ~~**Should `map_entities()` live on `CanonicalMapper` or as a standalone function?**~~ **Resolved by updated ADR-002.** The mapper now owns both construction from the DataFrame (`from_results_data_frame`) and output of the mapped DataFrame (`get_mapped_results_dataframe`). No standalone function needed.

3. **Is `EvaluationResult` still the return type of `calculate_score_on_df()`?** Currently yes. Should it remain the aggregated result container, or should we move to a simpler return type (e.g., a dict or a new `ScoreResult` dataclass)?

4. **What happens to `get_results_dataframe()`?** Currently it's on `BaseEvaluator` and applies entity mapping. In the new pipeline, `predict_dataset()` produces the raw DataFrame and `get_mapped_results_dataframe()` applies mapping externally. Should `get_results_dataframe()` be deprecated or kept as a convenience?

5. **Backward compatibility**: The old `evaluate_all()` → `calculate_score()` path — should it remain fully functional (just delegate internally), or should we start issuing deprecation warnings?

6. **Should `get_mapped_results_dataframe()` handle `None` mappings (suppress entities)?** `CanonicalMapper.get_mapping()` can return `None` for suppressed entities. `get_mapped_results_dataframe()` should map those to `"O"`.

7. **What happens to `from_dataset()`?** The existing `CanonicalMapper.from_dataset()` works on `List[InputSample]`. Should it coexist with `from_results_data_frame()`, or be deprecated in favor of the DataFrame-based entry point?

8. **Should the mapper store the DataFrame?** `get_mapped_results_dataframe()` implies the mapper holds a reference to the original DataFrame. Confirm this stateful design is acceptable (vs. a `map(df)` method that takes the DataFrame as argument).

9. **How do `get_mapped_results_dataframe()` and `map_entities(df, hierarchy)` relate?** ADR-001 shows two patterns: the simple pipeline uses `get_mapped_results_dataframe()` (maps to canonical, level 3), while the multi-hierarchy pipeline uses `map_entities(results_df, hierarchy=N)`. Are these the same method with a default `hierarchy=3`, or two separate methods? Recommendation: `map_entities(results_df, hierarchy=3)` as a single method, with `get_mapped_results_dataframe()` as a convenience wrapper that calls `map_entities(self._results_df, hierarchy=3)`.

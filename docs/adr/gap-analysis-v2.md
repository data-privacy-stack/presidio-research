# Gap Analysis: ADR-001 & ADR-002 vs. Current Implementation (v2)

> Updated with design decisions from review.

## Summary

The codebase has **partially implemented ADR-002** (CanonicalMapper / EntityHierarchy) but has **not started ADR-001** (simplified DataFrame pipeline). The entity mapping infrastructure is largely in place; the evaluation pipeline refactoring and clean separation of concerns are the main remaining work.

### Key Design Decisions (from review)

1. **Three clean layers**: Model (predict) → Mapper (canonicalize) → Evaluator (score). No layer handles another's concerns.
2. **Evaluator owns zero mapping logic** — all entity mapping, including BIO prefix stripping (`B-PERSON` → `PERSON`), is removed from the evaluation package.
3. **Mapper is stateless w.r.t. DataFrames** — it stores only the mapping dict, not the DataFrame. `get_mapped_results_dataframe(results_df)` takes the DataFrame as an argument.
4. **`None` mappings preserve the entity** — `None` in the mapping dict means "keep as-is for evaluation" (will always be FP or FN since it won't match anything on the other side).
5. **`evaluate_all()` is deprecated** — raises `DeprecationError` with migration instructions.
6. **`EvaluationResult` remains** as return type of `calculate_score_on_df()` to support downstream metrics and plotting.

---

## Agreed CanonicalMapper API

### Construction

```python
# From a list/set of entities
mapper = CanonicalMapper(labels=["FIRSTNAME", "GPE", "US_SSN"])

# Hierarchy parameter controls mapping depth (default=3 = canonical)
mapper = CanonicalMapper(labels=[...], hierarchy=3)
```

Two valid inputs only:
1. A `list` or `set` of entity labels
2. A results DataFrame (via `get_mapped_results_dataframe()`, which extracts labels internally)

### Core Methods

```python
# Inspect mapping — two modes
mapper.get_mapping()                    # → dict[str, str | None]
mapper.get_mapping(mode='html')         # → HTML table (original, mapped@level3, category@level2)
mapper.get_mapping(mode='text')         # → text table (for CLI)

# Manually resolve pending labels
mapper.map({"GGE": "ORG", "CustID": "CLIENT_ID", "MY_CUSTOM_LABEL": None})

# Map a DataFrame — extracts new labels, updates internal mapping, returns mapped DataFrame
updated_df = mapper.get_mapped_results_dataframe(results_df, hierarchy=3)
```

### Removed / Changed

| Previous API | Decision |
|---|---|
| `CanonicalMapper.from_dataset(samples)` | **Remove.** Not a supported input type. |
| `CanonicalMapper.from_results_data_frame(results_df)` (classmethod) | **Remove.** Replaced by regular method `get_mapped_results_dataframe(results_df)` |
| `mapper.map_entities(results_df, hierarchy=N)` | **Merged into** `get_mapped_results_dataframe(results_df, hierarchy=N)` — single method |
| Mapper stores DataFrame reference | **No.** Mapper stores only the mapping. DataFrame is passed in. |

### `get_mapping()` display format (html/text)

Each row shows:

| Original Entity | Mapped Entity (Level 3) | Category (Level 2) | Resolution Tier |
|---|---|---|---|
| `FIRSTNAME` | `NAME` | `PERSON` | EXACT |
| `US_SSN` | `SSN` | `GOVERNMENT_ID` | COUNTRY |
| `GGE` | *(pending)* | — | PENDING |

---

## Agreed Evaluation Pipeline

### Target Pipeline

**Simple (single hierarchy level):**
```python
# 1. Load dataset
dataset = InputSample.read_dataset_json("data/dataset.json")

# 2. Choose model and run predictions → get DataFrame directly
model = PresidioAnalyzerWrapper(analyzer_engine=AnalyzerEngine())
results_df = model.predict_dataset(dataset)  # returns 5-column DataFrame

# 3. Map entities
mapper = CanonicalMapper()
results_df_mapped = mapper.get_mapped_results_dataframe(results_df)  # hierarchy=3 by default

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
    results_df_mapped = mapper.get_mapped_results_dataframe(results_df, hierarchy=hierarchy)
    results_per_hierarchy.append(evaluator.calculate_score_on_df(results_df=results_df_mapped))

# 5. Analyze/plot
plotter = Plotter(results=results_per_hierarchy[0])
plotter.plot_scores()
```

---

## ADR-002: CanonicalMapper — Gap Breakdown

### ✅ Implemented (keep as-is)

| Requirement | Location |
|---|---|
| `EntityHierarchy` with `canonicalize()`, `get_branch()`, aliases | `entity_mapping/mapper.py` |
| `HIERARCHY` dictionary with full taxonomy | `entity_mapping/definitions.py` |
| Resolution tiers: EXACT, COUNTRY, COUNTRY_FALLBACK, FUZZY, PENDING | `entity_mapping/mapper.py` |
| BIO/BIOES prefix stripping in mapper | `entity_mapping/mapper.py` |
| `map()` for manual resolution (atomic batch) | `entity_mapping/mapper.py` |
| `get_mapping()` → `dict[str, str \| None]` | `entity_mapping/mapper.py` |
| `IncompleteMapping` exception | `entity_mapping/mapper.py` |
| Logging (INFO/WARNING) | `entity_mapping/mapper.py` |
| `COUNTRIES` set | `entity_mapping/definitions.py` |
| Tests (mapper, hierarchy, label coverage) | `tests/` |

### 🔴 New / Changed

| # | Item | Status | Work Required |
|---|---|---|---|
| M1 | `__init__` accepts `hierarchy` param (default=3) | 🔴 New | Add `hierarchy` parameter to `CanonicalMapper.__init__()` controlling the depth of canonical resolution |
| M2 | `get_mapped_results_dataframe(results_df, hierarchy=3)` | 🔴 New | Regular method that: (a) extracts labels from `annotation` + `prediction` columns, (b) auto-resolves new labels and updates internal mapping, (c) returns a new DataFrame with both columns remapped at the requested hierarchy level. `None` mappings are kept as-is (the entity name passes through to the evaluator, which will count it as FP/FN). **Mixed-level warning**: if mapped predictions and annotations resolve to different granularities (e.g. model predicts `PERSON` but dataset annotates `FIRSTNAME`), the mapper logs a user-friendly warning naming the affected entities and suggesting: use a broader mapping level, or use `mapper.map()` to manually remap. No hierarchy/level jargon in the warning. |
| M3 | `get_mapping(mode='html')` and `get_mapping(mode='text')` | 🟡 Verify | Check if HTML/text rendering exists. If not, implement. Display columns: original entity, mapped entity (level 3), category (level 2), resolution tier. |
| M4 | Remove `from_dataset()` | 🔴 Breaking | Delete the classmethod. Mapper only accepts `labels` (list/set) or extracts labels via `get_mapped_results_dataframe()`. |
| M5 | Remove `from_results_data_frame()` classmethod if added | 🔴 Cleanup | Per design: no classmethod factory. Use `CanonicalMapper()` + `get_mapped_results_dataframe(df)`. |

---

## ADR-001: Evaluation Pipeline — Gap Breakdown

### 🔴 Strip all entity mapping from evaluation package

This is a **cross-cutting change** that affects multiple files:

| File | What to remove | Notes |
|---|---|---|
| `base_evaluator.py` | `entity_mapping` constructor param | Evaluator no longer handles mapping |
| `base_evaluator.py` | `_normalize_entity_for_comparison()` | All normalization moves to mapper |
| `base_evaluator.py` | `_to_io()` (BIO prefix stripping) | BIO stripping is a mapping concern, not evaluation |
| `base_evaluator.py` | `compare_by_io` constructor param | Related to BIO — removed with it |
| `base_evaluator.py` | Entity mapping logic in `compare()` | Evaluator receives already-mapped entities |
| `base_evaluator.py` | Entity mapping logic in `get_results_dataframe()` | Deprecated entirely (see E4) |
| `span_evaluator.py` | Any `apply_entity_mapping` flags in span processing | Mapping is done before evaluation |
| `token_evaluator.py` | Any mapping references | Clean separation |

**Implication**: The evaluator's `calculate_score_on_df()` receives a DataFrame where `annotation` and `prediction` columns are already in the same entity namespace. It does pure scoring — no remapping.

### Per-Item Gap Breakdown

| # | Item | Status | Work Required |
|---|---|---|---|
| E1 | `BaseModel.predict_dataset()` → 5-column DataFrame | 🔴 Not started | Add to `BaseModel`. Calls `batch_predict()`, assembles DataFrame with `(sentence_id, token, annotation, prediction, start_indices)`. No entity mapping. |
| E2 | Strip all entity mapping from evaluation package | 🔴 Not started | Remove `entity_mapping` param, `_normalize_entity_for_comparison()`, `_to_io()`, `compare_by_io` from `BaseEvaluator`. Update `SpanEvaluator` and `TokenEvaluator`. Evaluator receives pre-mapped DataFrame. |
| E3 | `TokenEvaluator.calculate_score_on_df()` | 🔴 Not started | Add to match SpanEvaluator's interface. Required for the DataFrame-based pipeline. |
| E4 | Deprecate `get_results_dataframe()` | 🔴 Not started | Add `DeprecationWarning` with migration message: "Use `model.predict_dataset()` + `mapper.get_mapped_results_dataframe()` instead." |
| E5 | Deprecate `evaluate_all()` | 🔴 Not started | Raise `DeprecationError` (hard stop) with migration message pointing to the new pipeline. |
| E6 | `EvaluationResult` retains all fields for metrics + error analysis + plotting | 🟡 Verify | Confirm `EvaluationResult` has everything needed. Per-sample carrier fields (`tokens`, `actual_tags`, `predicted_tags`) may still be needed for error analysis. Verify `Plotter` works after changes. |
| E7 | `SpanEvaluator` decoupled from model | 🟡 Partial | `` already works. Ensure `calculate_score_on_df()` is fully usable without a model instance (no `self.model` references). |
| E8 | `model` param becomes truly optional in `BaseEvaluator` | ✅ Done | Already handled. |
| E9 | Update notebooks & documentation | 🔴 Not started | Notebooks 4, 5, 6 → new pipeline. |

---

## Mixed-Level Entities: Behavior Clarification

A common scenario: the dataset annotates at a specific level (e.g. `FIRSTNAME`, `LASTNAME`) but the model predicts at a broader level (e.g. `PERSON`). After mapping, `PERSON` and `FIRSTNAME` remain different strings and produce spurious FP/FN during evaluation.

**Behavior**: `get_mapped_results_dataframe()` detects that mapped annotations and predictions resolve to different granularity levels and logs a user-friendly warning:

> *"Your model predicts `PERSON` but your dataset annotates `FIRSTNAME` and `LASTNAME`. After mapping, these are different entities and won't match during evaluation. You can either use a broader mapping level (e.g. `hierarchy=2`) so everything resolves to `PERSON`, or use `mapper.map({"PERSON": "FIRSTNAME"})` to tell the mapper what `PERSON` means in your context."*

The warning names the concrete entities — no jargon about hierarchy levels or internals. The mapper proceeds without error; the user decides whether to adjust.

---

## `None` Mappings: Behavior Clarification

When `CanonicalMapper.get_mapping()` returns `None` for a label, the entity name **passes through unchanged** to the evaluator. Since the evaluator compares `annotation` vs. `prediction` by exact string match:

- If `None`-mapped entity is on the **annotation side** → it won't match any prediction → counted as **FN**
- If `None`-mapped entity is on the **prediction side** → it won't match any annotation → counted as **FP**

This is the correct behavior: the user has explicitly said "I don't know what this entity is" — it should be evaluated but penalized.

---

## Priority Order for Implementation

| Priority | Item | ID | Rationale |
|---|---|---|---|
| **P0** | `BaseModel.predict_dataset()` | E1 | Foundation for the new pipeline |
| **P0** | `CanonicalMapper` `hierarchy` param + `get_mapped_results_dataframe()` | M1, M2 | Bridges model output → evaluator input |
| **P0** | Strip entity mapping from evaluation package | E2 | Clean separation of concerns; unblocks evaluator simplification |
| **P1** | `TokenEvaluator.calculate_score_on_df()` | E3 | Parity with SpanEvaluator |
| **P1** | Deprecate `evaluate_all()` (hard) + `get_results_dataframe()` (warning) | E4, E5 | Remove old code paths |
| **P1** | Remove `from_dataset()` from CanonicalMapper | M4 | Clean up old API |
| **P1** | `get_mapping(mode='html'/'text')` display | M3 | Transparency for users |
| **P2** | Verify `EvaluationResult` + `Plotter` work end-to-end | E6 | Ensure nothing breaks downstream |
| **P2** | Remove non-Presidio model wrappers (Flair, Spacy, Stanza, TextAnalytics) | R1 | Clean up models package; remove `[ner]` optional deps |
| **P2** | Update dependency versions to latest | R3 | `pandas 3.x`, `transformers 5.x` are major bumps — test carefully |
| **P2** | Migrate from Poetry to uv | R4 | Replace `poetry.lock` with `uv.lock`, convert `pyproject.toml` build system |
| **P2** | Add ruff config + pre-commit hooks | R5 | `ruff.toml` (line-length=88), pre-commit: format, lint, unit tests |
| **P2** | Separate unit vs. integration tests | R6 | Add `@pytest.mark.integration`, pre-commit runs unit only |
| **P3** | Update notebooks 4, 5, 6 to new pipeline | E9, R2 | Show the new 5-step pipeline |
| **P3** | Replace `notebooks/models/` with deprecation notice | R2 | Single-cell "no longer supported" message |

---

## Additional Requirements

### R1: Remove all non-Presidio model wrappers

Keep only `BaseModel`, `PresidioAnalyzerWrapper`, and `PresidioRecognizerWrapper`. Remove:

| File to remove | Class |
|---|---|
| `presidio_evaluator/models/flair_model.py` | `FlairModel` |
| `presidio_evaluator/models/spacy_model.py` | `SpacyModel` |
| `presidio_evaluator/models/stanza_model.py` | `StanzaModel` |
| `presidio_evaluator/models/text_analytics_wrapper.py` | `TextAnalyticsWrapper` |

Update `models/__init__.py` to only export `BaseModel`, `PresidioAnalyzerWrapper`, `PresidioRecognizerWrapper`.

Remove the `[ner]` optional dependency group from `pyproject.toml` (`flair`, `spacy_huggingface_pipelines`, `azure-ai-textanalytics`, `gliner`, `onnxruntime`).

Remove related tests:
- `test_spacy_model.py` (imports `SpacyModel`)
- `test_spacy_recognizer_generated_text.py` (imports `PresidioRecognizerWrapper` with spacy recognizer — **keep if R1 keeps recognizer wrapper, remove if spacy-specific**)

No other test files import the removed models (`FlairModel`, `StanzaModel`, `TextAnalyticsWrapper`).

### R2: Update notebooks

**Notebooks 4, 5, 6** — rewrite to use the new pipeline:
- `4_Evaluate_Presidio_Analyzer.ipynb` — new 5-step pipeline with `predict_dataset()` → `CanonicalMapper` → `calculate_score_on_df()`
- `5_Evaluate_Custom_Presidio_Analyzer.ipynb` — same, with custom recognizers
- `6_Interactive_Entity_Mapping.ipynb` — update to use new `CanonicalMapper` API (`get_mapped_results_dataframe()`, `get_mapping(mode='html')`)

**`notebooks/models/` directory** — replace each notebook's content with a single cell:

> "This notebook is no longer supported. Add models directly through Presidio to evaluate them."

Notebooks to replace:
- `Create datasets for Spacy training.ipynb`
- `Evaluate azure text analytics.ipynb`
- `Evaluate flair models.ipynb`
- `Evaluate spacy models.ipynb`
- `Evaluate stanza models.ipynb`

### R3: Update all dependency versions to latest

| Dependency | Current | Latest | New constraint |
|---|---|---|---|
| `spacy` | `^3.0.0` | `3.8.13` | `^3.8.0` |
| `numpy` | `^2.0.0` | `2.4.3` | `^2.4.0` |
| `pandas` | `^2.1.4` | `3.0.1` | `^3.0.0` |
| `presidio-analyzer` | `^2.2.360` | `2.2.362` | `^2.2.362` |
| `presidio-anonymizer` | `^2.2.360` | `2.2.362` | `^2.2.362` |
| `plotly` | `^6.1.1` | `6.6.0` | `^6.6.0` |
| `transformers` | `^4.57.6` | `5.3.0` | `^5.3.0` |
| `scikit-learn` | `^1.3.2` | `1.8.0` | `^1.8.0` |
| `faker` | `*` | `40.11.1` | `^40.0.0` |
| `tqdm` | `^4.60.0` | `4.67.3` | `^4.67.0` |
| `requests` | `^2.25` | `2.32.5` | `^2.32.0` |
| `python-dotenv` | `^1.0.0` | `1.2.2` | `^1.2.0` |
| `xmltodict` | `^0.12.0` | `1.0.4` | `^1.0.0` |

**Note**: `pandas ^3.0.0` and `transformers ^5.3.0` are major version bumps. These may introduce breaking changes and should be tested carefully.

### R4: Migrate from Poetry to uv

Replace Poetry with [uv](https://docs.astral.sh/uv/) as the project's package manager and build tool.

| Task | Details |
|---|---|
| Remove `poetry.lock` | Delete the Poetry lock file |
| Remove Poetry build system from `pyproject.toml` | Replace `[build-system] requires = ["poetry-core"]` / `build-backend = "poetry.core.masonry.api"` with `hatchling` or keep `setuptools` — uv is manager-agnostic |
| Convert Poetry-specific `pyproject.toml` sections | `[tool.poetry.dependencies]` → standard `[project.dependencies]` (PEP 621). `[tool.poetry.group.*.dependencies]` → `[project.optional-dependencies]` |
| Generate `uv.lock` | Run `uv lock` to create the new lock file |
| Update CI/CD | Replace `poetry install` / `poetry run` with `uv sync` / `uv run` in any GitHub Actions or CI configs |
| Update README | Replace Poetry install instructions with `uv sync` / `uv run` |
| Update contributor docs | Any `poetry add`, `poetry shell` references → `uv add`, `uv run` |

**Note**: The `pyproject.toml` conversion from Poetry to PEP 621 format should be done together with R3 (dependency version updates) to avoid duplicate edits.

### R5: Add ruff linter/formatter + pre-commit hooks

**Ruff config** — use `ruff.toml` based on the platform config with `line-length = 88`:

```toml
# ruff.toml (key settings — full file adapted from platform/ruff.toml)
line-length = 88           # changed from 120
target-version = "py312"
indent-width = 4

[format]
quote-style = "double"     # changed from single

[lint]
select = ["E", "F", "I", "UP", "C4", "B", "A", "COM", "N", "ANN", "ASYNC", "S",
          "PLC", "PLE", "PLR", "PLW"]
# (full ignore list from platform config, without tools/privacy per-file-ignores)
```

**Pre-commit hooks** — add `.pre-commit-config.yaml` with two hooks:

| Hook | What it does |
|---|---|
| `ruff format --check` | Ensures code is formatted |
| `ruff check` | Linting (errors, imports, security, naming) |
| `pytest` (unit tests only) | Runs unit tests (excludes integration tests) |

### R6: Reorganize tests into topic-based subdirectories

**Current state**: All test files are flat in `tests/`. Integration tests are mixed with unit tests. Some are marked `pytest.mark.slow`, but there's no clean separation.

**Target**: Group tests into subdirectories mirroring the package structure, with a dedicated `integration/` directory.

```
tests/
├── conftest.py                          # shared fixtures
├── data_generator/
│   ├── test_faker_sentences.py
│   ├── test_presidio_pseudonymize.py
│   ├── test_presidio_sentence_faker.py
│   ├── test_providers.py
│   ├── test_record_generator.py
│   └── test_span_generator.py
├── entity_mapping/
│   ├── test_canonical_mapper.py
│   ├── test_entity_hierarchy.py
│   └── test_label_coverage.py
├── evaluation/
│   ├── test_evaluation_result.py
│   ├── test_evaluator.py
│   ├── test_model_error.py
│   ├── test_plotter.py
│   ├── test_span_evaluator.py
│   └── test_token_evaluator.py
├── models/
│   └── test_base_model.py
├── integration/                          # loads real models — NOT run in pre-commit
│   ├── test_notebook.py                 (existing)
│   ├── test_presidio_analyzer_wrapper.py
│   ├── test_recognizers_generated_text.py
│   ├── test_recognizers_template_csv.py
│   └── test_data_objects.py             (loads spacy en_core_web_sm)
├── test_data_objects.py                  → move to integration/
├── test_span_to_tag.py                  (root-level utility)
└── test_validation.py                   (root-level utility)
```

**Files to remove** (R1 — obsolete models):
- `test_spacy_model.py`
- `test_spacy_recognizer_generated_text.py`

**No tests exist for** `dataset_formatters` or `experiment_tracking` (gaps, but not blocking).

**Changes needed**:

1. Create `__init__.py` in each test subdirectory.
2. Move test files into their respective subdirectories.
3. Update imports in moved files if they use relative paths.
4. Add `pytest.mark.integration` marker definition to `pyproject.toml`.
5. Mark all tests in `tests/integration/` with `@pytest.mark.integration` (or configure via `conftest.py` in the directory).
6. Pre-commit hook runs: `pytest -m "not integration"` (unit tests only).
7. CI runs both: `pytest` (all) or `pytest -m integration` (integration only) separately.
8. Replace the `--runslow` CLI flag with the standard `-m "not integration"` filter (or keep both).

---

## Resolved Questions

| # | Question | Decision |
|---|---|---|
| 1 | Should `predict_dataset()` handle entity mapping? | **No.** Neither model nor evaluator handles mapping. Mapping is its own layer. |
| 2 | Should `map_entities()` live on CanonicalMapper or standalone? | **On CanonicalMapper** as `get_mapped_results_dataframe()`. |
| 3 | Is `EvaluationResult` the return type of `calculate_score_on_df()`? | **Yes.** Required for downstream metrics and plotting. |
| 4 | What happens to `get_results_dataframe()`? | **Deprecated** with clear migration message. |
| 5 | Backward compat for `evaluate_all()` → `calculate_score()`? | **Deprecated (hard stop).** Raise `DeprecationError`. |
| 6 | Should `None` mappings suppress entities? | **No.** `None`-mapped entities pass through unchanged → always FP or FN. |
| 7 | What happens to `from_dataset()`? | **Remove.** Not a supported input. |
| 8 | Should the mapper store the DataFrame? | **No.** Mapper stores only the mapping. DataFrame passed as argument. |
| 9 | Relationship between `get_mapped_results_dataframe` and `map_entities`? | **Single method**: `get_mapped_results_dataframe(results_df, hierarchy=3)`. |
| 10 | What if the model predicts at a broader level than the dataset (e.g. model says `PERSON`, dataset says `FIRSTNAME`)? | **Warn, don't error.** The mapper detects that predictions and annotations resolve to different granularities, logs a user-friendly warning naming the affected entities, and suggests: use a broader mapping level, or use `mapper.map()` to manually remap. No hierarchy jargon. The evaluator stays pure string comparison — no hierarchy awareness. |

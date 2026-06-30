# PRD: Presidio Evaluator Redesign

## Introduction

Presidio Evaluator's current architecture tangles entity mapping, prediction, and scoring into a single call path (`evaluate_all()`). Users must understand internal class wiring to run a basic evaluation. Entity mapping is buried inside the evaluator, making it opaque and hard to customize.

This redesign separates concerns into three clean layers — **Model → Mapper → Evaluator** — each with a clear DataFrame-based interface. A user loads a dataset, calls the model, maps entities with `CanonicalMapper`, and evaluates — all in five lines of code. The project infrastructure is also modernized: obsolete model wrappers are removed, dependencies are updated, Poetry is replaced with uv, ruff is adopted, and tests are reorganized.

**Reference documents:**
- [ADR-001: Simplified Evaluation Pipeline](../docs/adr/ADR-001-simplified-evaluation-pipeline.md)
- [ADR-002: Entity Mapping](../docs/adr/ADR-002-entity-mapping.md)
- [Gap Analysis v2](../docs/adr/gap-analysis-v2.md) (authoritative source of truth for all requirements)

## Goals

- Provide a 5-step pipeline (load → predict → map → evaluate → plot) that fits in a single notebook cell
- Make entity mapping explicit and inspectable via `CanonicalMapper`, never hidden inside the evaluator
- Ensure every evaluator (`SpanEvaluator`, `TokenEvaluator`) accepts the same DataFrame interface
- Remove deprecated code paths with clear migration guidance
- Remove non-Presidio model wrappers that are no longer maintained
- Modernize project tooling (uv, ruff, test reorganization)

---

## User Stories

### US-001: Predict a dataset and get a DataFrame

**Description:** As a user, I want to call `model.predict_dataset(dataset)` and receive a DataFrame of predictions so that I have a single, standard object to pass to mapping and evaluation.

**Acceptance Criteria:**
- [ ] `BaseModel` has a `predict_dataset(dataset: list[InputSample]) -> pd.DataFrame` method
- [ ] The returned DataFrame has exactly 5 columns: `sentence_id`, `token`, `annotation`, `prediction`, `start_indices`
- [ ] `predict_dataset()` calls the existing `batch_predict()` internally and assembles the DataFrame
- [ ] `predict_dataset()` does NOT perform any entity mapping
- [ ] `PresidioAnalyzerWrapper.predict_dataset()` works end-to-end with real data
- [ ] Unit tests verify the DataFrame schema (column names, types) and that mapping is not applied
- [ ] Typecheck/lint passes

---

### US-002: Map entities using CanonicalMapper with hierarchy support

**Description:** As a user, I want to create a `CanonicalMapper` and map my results DataFrame so that model predictions are projected onto the dataset's entity vocabulary before evaluation.

**Acceptance Criteria:**
- [ ] `CanonicalMapper.__init__` accepts optional `canonical_depth` (default=None for auto-discovery) and `eval_entities` parameters
- [ ] `analyze(results_df)` discovers labels, auto-discovers evaluation depth via majority vote of annotation depths, identifies labels in hierarchy, and projects predictions onto dataset entity set
- [ ] `get_mapped_results_dataframe()` returns a new DataFrame with `annotation` and `prediction` columns remapped
- [ ] Trivial collisions (ancestor/descendant on same branch) are auto-fixed and flagged as COLLISION_TRIVIAL
- [ ] Ambiguous collisions (ancestor of multiple dataset entities) are flagged as COLLISION_AMBIGUOUS for user decision
- [ ] Cross-branch collisions are flagged as COLLISION_CROSS_BRANCH for user decision
- [ ] PREDICTION_ONLY entities default to removal (suppress from evaluation)
- [ ] DATASET_ONLY entities default to keep (show model gaps as false negatives)
- [ ] Issues sorted by severity then affected token count (most impactful first)
- [ ] Labels mapped to `None` become `"O"` in the output DataFrame
- [ ] Unit tests cover: auto-discovery of depth, projection rules, collision detection, DATASET_ONLY, PREDICTION_ONLY, frequency-based sorting
- [ ] Typecheck/lint passes

---

### US-003: Inspect entity mappings in table form

**Description:** As a user, I want to call `mapper.render_html()` or `mapper.get_mapping(mode='text')` so that I can see how each entity was resolved and what issues remain, sorted by impact.

**Acceptance Criteria:**
- [ ] `get_mapping()` with no arguments still returns `dict[str, str | None]` (backward compatible)
- [ ] `get_mapping(mode='html')` returns an HTML table string with columns: Raw Label, Identified As, Mapped To, Issue, Tokens Affected
- [ ] `get_mapping(mode='text')` returns a plain-text table suitable for terminal/CLI output
- [ ] Issues are sorted by severity then token count (most impactful first)
- [ ] Unit tests verify HTML contains expected `<table>` markup and text output is readable
- [ ] Typecheck/lint passes

---

### US-004: Remove old CanonicalMapper factory methods

**Description:** As a user, I want a single clear way to create and use a `CanonicalMapper` (constructor + `get_mapped_results_dataframe`) so that I am not confused by multiple overlapping factory methods.

**Acceptance Criteria:**
- [ ] `CanonicalMapper.from_dataset()` classmethod is deleted
- [ ] `CanonicalMapper.from_results_data_frame()` classmethod is deleted (if present)
- [ ] No references to removed methods remain in production code
- [ ] Existing tests that used `from_dataset()` are updated or removed
- [ ] Typecheck/lint passes

---

### US-005: Strip all entity mapping from the evaluation package

**Description:** As a user, I want the evaluator to do pure scoring on a pre-mapped DataFrame so that mapping concerns are never hidden inside evaluation logic.

**Acceptance Criteria:**
- [ ] `BaseEvaluator.__init__` no longer accepts an `entity_mapping` parameter
- [ ] `_normalize_entity_for_comparison()` is removed from `BaseEvaluator`
- [ ] `_to_io()` (BIO prefix stripping) is removed from `BaseEvaluator` — BIO stripping happens in the mapper
- [ ] `compare_by_io` constructor parameter is removed from `BaseEvaluator`
- [ ] Any entity mapping logic in `compare()` is removed
- [ ] `SpanEvaluator` and `TokenEvaluator` have no mapping references
- [ ] `calculate_score_on_df()` receives a DataFrame where `annotation` and `prediction` are already in the same namespace — it performs no remapping
- [ ] All existing evaluator tests are updated to pass pre-mapped data
- [ ] `EvaluationResult` returned by `SpanEvaluator.calculate_score_on_df()` still contains all fields needed by `Plotter` (regression check after the mapping strip)
- [ ] `Plotter(results=results).plot_scores()` runs without errors after the change
- [ ] Typecheck/lint passes

---

### US-006: Add calculate_score_on_df to TokenEvaluator

**Description:** As a user, I want `TokenEvaluator` to have the same `calculate_score_on_df()` interface as `SpanEvaluator` so that I can use either evaluator interchangeably in the new pipeline.

**Acceptance Criteria:**
- [ ] `TokenEvaluator.calculate_score_on_df(results_df) -> EvaluationResult` is implemented
- [ ] It accepts the same 5-column DataFrame schema as `SpanEvaluator.calculate_score_on_df()`
- [ ] It returns an `EvaluationResult` with per-entity and aggregate metrics
- [ ] Per-sample carrier fields (`tokens`, `actual_tags`, `predicted_tags`) are populated in `EvaluationResult` for error analysis
- [ ] Unit tests verify metrics match expected values on a known test DataFrame
- [ ] Integration test runs the full 5-step pipeline (`predict_dataset()` → `get_mapped_results_dataframe()` → `calculate_score_on_df()` → `Plotter`) end-to-end without errors
- [ ] Typecheck/lint passes

---

### US-007: Deprecate evaluate_all with a hard stop

**Description:** As a user who accidentally calls the old `evaluate_all()`, I want a clear error message telling me exactly what to use instead so that I can migrate quickly.

**Acceptance Criteria:**
- [ ] `evaluate_all()` raises `DeprecationError` immediately (hard stop — no fallback execution)
- [ ] The error message includes a migration snippet showing the new 5-step pipeline
- [ ] Unit test verifies that calling `evaluate_all()` raises `DeprecationError` with the expected message
- [ ] Typecheck/lint passes

---

### US-008: Deprecate get_results_dataframe with a warning

**Description:** As a user calling `get_results_dataframe()`, I want a deprecation warning telling me to use `model.predict_dataset()` + `mapper.get_mapped_results_dataframe()` instead so that I can migrate at my own pace.

**Acceptance Criteria:**
- [ ] `get_results_dataframe()` emits a `DeprecationWarning` with a clear migration message
- [ ] The method still executes (soft deprecation) to allow gradual migration
- [ ] The warning message references `model.predict_dataset()` and `mapper.get_mapped_results_dataframe()`
- [ ] Unit test captures the warning and verifies its content
- [ ] Typecheck/lint passes

---

### US-009: Decouple model from evaluators

**Description:** As a user, I want `SpanEvaluator` and `TokenEvaluator` to accept a pre-built results DataFrame instead of a model, so that I can separate prediction, mapping, and scoring into distinct steps and swap evaluators interchangeably.

**Acceptance Criteria:**
- [ ] `BaseEvaluator.__init__` raises `DeprecationError` when a non-`None` model is passed, directing users to `model.predict_dataset()` + `evaluator.calculate_score_on_df()`
- [ ] `SpanEvaluator()` and `TokenEvaluator()` are the standard construction patterns — no model required
- [ ] `calculate_score_on_df(results_df)` is the primary entry point for both evaluators and requires no model
- [ ] `evaluate_sample()` emits a `DeprecationWarning` with a migration message referencing `predict_dataset()` and `calculate_score_on_df()`
- [ ] `evaluate_sample()` still executes (soft deprecation)
- [ ] Unit test: constructing `SpanEvaluator(model=<model>)` raises `DeprecationError`
- [ ] Unit test: constructing `SpanEvaluator()` does not raise
- [ ] Unit test: `evaluate_sample()` emits `DeprecationWarning`
- [ ] All existing tests that pass `model=<mock>` to the evaluator constructor are updated to use `` and call `calculate_score_on_df()` directly
- [ ] Typecheck/lint passes

**Notes:** Core decoupling story per ADR-001. The evaluator becomes a pure scoring engine — it scores a pre-built DataFrame and has no knowledge of model inference. `evaluate_sample()` soft-deprecation is included here as part of the same decoupling effort.

---

### US-010: Remove non-Presidio model wrappers

**Description:** As a user, I want the models package to contain only Presidio-based wrappers so that I am not confused by unmaintained model integrations, and the dependency footprint is reduced.

**Acceptance Criteria:**
- [ ] `presidio_evaluator/models/flair_model.py` is deleted
- [ ] `presidio_evaluator/models/spacy_model.py` is deleted
- [ ] `presidio_evaluator/models/stanza_model.py` is deleted
- [ ] `presidio_evaluator/models/text_analytics_wrapper.py` is deleted
- [ ] `models/__init__.py` exports only `BaseModel`, `PresidioAnalyzerWrapper`, `PresidioRecognizerWrapper`
- [ ] The `[ner]` optional dependency group is removed from `pyproject.toml` (`flair`, `spacy_huggingface_pipelines`, `azure-ai-textanalytics`, `gliner`, `onnxruntime`)
- [ ] `tests/test_spacy_model.py` is deleted
- [ ] `tests/test_spacy_recognizer_generated_text.py` is deleted (spacy-specific recognizer test)
- [ ] No remaining imports of removed classes in production or test code
- [ ] Typecheck/lint passes

---

### US-011: Update notebooks to the new pipeline

**Description:** As a user, I want the evaluation notebooks to demonstrate the new 5-step pipeline so that I can follow working examples to evaluate my own models.

**Acceptance Criteria:**
- [ ] `notebooks/4_Evaluate_Presidio_Analyzer.ipynb` uses: load dataset → `predict_dataset()` → `CanonicalMapper().analyze()` → `get_mapped_results_dataframe()` → `calculate_score_on_df()` → `Plotter`
- [ ] `notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb` uses the same pipeline with custom recognizers
- [ ] `notebooks/6_Interactive_Entity_Mapping.ipynb` demonstrates `CanonicalMapper` API: `analyze()`, `render_html()` (with frequency-sorted issues), `map()` for manual resolution, multi-model comparison
- [ ] Each notebook runs end-to-end without errors (verified by running all cells)
- [ ] Typecheck/lint passes (for any `.py` cells)

---

### US-012: Replace obsolete model notebooks with deprecation notice

**Description:** As a user who finds an old model notebook, I want a clear message telling me the notebook is no longer supported and how to evaluate models through Presidio instead.

**Acceptance Criteria:**
- [ ] Each notebook in `notebooks/models/` is replaced with a single-cell notebook containing: *"This notebook is no longer supported. Add models directly through Presidio to evaluate them."*
- [ ] Affected notebooks: `Create datasets for Spacy training.ipynb`, `Evaluate azure text analytics.ipynb`, `Evaluate flair models.ipynb`, `Evaluate spacy models.ipynb`, `Evaluate stanza models.ipynb`
- [ ] No executable code remains in the replaced notebooks

---

### US-013: Update dependency versions

**Description:** As a user, I want dependencies to be up-to-date so that I benefit from bug fixes, performance improvements, and security patches.

**Acceptance Criteria:**
- [ ] `pyproject.toml` dependency constraints are updated to:
  - `spacy ^3.8.0`, `numpy ^2.4.0`, `pandas ^3.0.0`, `presidio-analyzer ^2.2.362`, `presidio-anonymizer ^2.2.362`
  - `plotly ^6.6.0`, `transformers ^5.3.0`, `scikit-learn ^1.8.0`, `faker ^40.0.0`, `tqdm ^4.67.0`
  - `requests ^2.32.0`, `python-dotenv ^1.2.0`, `xmltodict ^1.0.0`
- [ ] Lock file is regenerated and all dependencies resolve successfully
- [ ] `pandas ^3.0.0` and `transformers ^5.3.0` major version bumps are tested — all existing tests pass or are updated for breaking changes
- [ ] Typecheck/lint passes

**Note:** This story should be implemented together with US-014 (Poetry → uv migration) to avoid duplicate `pyproject.toml` edits.

---

### US-014: Migrate from Poetry to uv

**Description:** As a developer, I want the project to use uv instead of Poetry so that dependency installation is faster and the project follows modern Python packaging standards (PEP 621).

**Acceptance Criteria:**
- [ ] `poetry.lock` is deleted
- [ ] `[build-system]` in `pyproject.toml` no longer references `poetry-core`
- [ ] `[tool.poetry.dependencies]` is converted to PEP 621 `[project.dependencies]`
- [ ] `[tool.poetry.group.*.dependencies]` is converted to `[project.optional-dependencies]`
- [ ] `uv.lock` is generated by running `uv lock`
- [ ] `uv sync` installs all dependencies successfully
- [ ] `uv run pytest` runs the test suite successfully
- [ ] README install instructions are updated to use `uv sync` / `uv run`
- [ ] Any CI/GitHub Actions configs are updated to replace `poetry install`/`poetry run` with `uv sync`/`uv run`
- [ ] Typecheck/lint passes

---

### US-015: Add ruff linter and formatter

**Description:** As a developer, I want consistent code formatting and linting enforced by ruff so that code style is uniform and common issues are caught early.

**Acceptance Criteria:**
- [ ] `ruff.toml` is added to the project root with: `line-length = 88`, `target-version = "py312"`, `indent-width = 4`, `quote-style = "double"`
- [ ] Lint rules include at minimum: `E`, `F`, `I`, `UP`, `C4`, `B`, `A`, `COM`, `N`, `ANN`, `ASYNC`, `S`, `PLC`, `PLE`, `PLR`, `PLW`
- [ ] `ruff format` and `ruff check` run without errors on the entire codebase (fix existing violations or add targeted ignores)
- [ ] Typecheck/lint passes

---

### US-016: Add pre-commit hooks

**Description:** As a developer, I want pre-commit hooks that automatically check formatting, linting, and run unit tests before each commit so that broken code is caught early.

**Acceptance Criteria:**
- [ ] `.pre-commit-config.yaml` is added with hooks for: `ruff format --check`, `ruff check`, `pytest -m "not integration"`
- [ ] `pre-commit install` sets up the hooks successfully
- [ ] Pre-commit runs only unit tests (not integration tests) to keep commit time fast
- [ ] All hooks pass on the current codebase

---

### US-017: Reorganize tests into topic-based directories

**Description:** As a developer, I want tests grouped by package topic so that I can find and run related tests easily.

**Acceptance Criteria:**
- [ ] Tests are organized into subdirectories mirroring the package structure:
  - `tests/data_generator/` — `test_faker_sentences.py`, `test_presidio_pseudonymize.py`, `test_presidio_sentence_faker.py`, `test_providers.py`, `test_record_generator.py`, `test_span_generator.py`
  - `tests/entity_mapping/` — `test_canonical_mapper.py`, `test_entity_hierarchy.py`, `test_label_coverage.py`
  - `tests/evaluation/` — `test_evaluation_result.py`, `test_evaluator.py`, `test_model_error.py`, `test_plotter.py`, `test_span_evaluator.py`, `test_token_evaluator.py`
  - `tests/models/` — `test_base_model.py`
  - `tests/integration/` — `test_notebook.py`, `test_presidio_analyzer_wrapper.py`, `test_recognizers_generated_text.py`, `test_recognizers_template_csv.py`, `test_data_objects.py`
  - Root-level utilities remain: `test_span_to_tag.py`, `test_validation.py`
- [ ] Each subdirectory has an `__init__.py`
- [ ] All imports in moved test files are updated and working
- [ ] `pytest.mark.integration` marker is defined in `pyproject.toml`
- [ ] All tests in `tests/integration/` are marked with `@pytest.mark.integration` (via directory-level `conftest.py` or decorator)
- [ ] `pytest -m "not integration"` runs only unit tests
- [ ] `pytest` (no filter) still runs all tests
- [ ] Typecheck/lint passes

---

### US-018: Write migration guide and changelog

**Description:** As a user upgrading from the previous version, I want a concise migration guide and a detailed changelog so that I know exactly what changed, what broke, and how to update my code.

**Acceptance Criteria:**
- [ ] `docs/migration-guide.md` is created with:
  - A summary of what changed (three-layer architecture, entity mapping moved out of evaluator)
  - Before/after code examples for the most common use cases (evaluate with SpanEvaluator, evaluate with TokenEvaluator, map entities)
  - A table of removed/renamed methods and their replacements
  - Instructions for handling `DeprecationError` from `evaluate_all()`
  - Instructions for handling `DeprecationWarning` from `get_results_dataframe()`
- [ ] `CHANGELOG.md` is updated with a new version section containing:
  - **Breaking changes**: removed model wrappers, removed `evaluate_all()`, removed entity mapping from evaluator, removed `from_dataset()`, `pandas ^3.0.0` / `transformers ^5.3.0` bumps
  - **New features**: `predict_dataset()`, `get_mapped_results_dataframe()`, `get_mapping(mode='html'/'text')`, `TokenEvaluator.calculate_score_on_df()`, ruff, uv, test reorganization
  - **Deprecations**: `get_results_dataframe()` (soft, with warning)
- [ ] Migration guide is concise (fits on one screen for simple cases)
- [ ] Typecheck/lint passes (for any code snippets in docs)

---

## Functional Requirements

- FR-1: `BaseModel.predict_dataset(dataset)` returns a DataFrame with columns `(sentence_id, token, annotation, prediction, start_indices)`. No entity mapping is applied.
- FR-2: `CanonicalMapper.__init__(labels=None, hierarchy=3)` accepts an optional hierarchy depth parameter.
- FR-3: `CanonicalMapper.get_mapped_results_dataframe(results_df, hierarchy=3)` extracts labels from the DataFrame, auto-resolves new labels, and returns a remapped DataFrame. The mapper stores only the mapping dict, not the DataFrame.
- FR-3a: When `get_mapped_results_dataframe()` detects that mapped predictions and annotations resolve to different granularities (e.g. model predicts `PERSON`, dataset annotates `FIRSTNAME`), it logs a user-friendly warning naming the affected entities and suggesting to use a broader mapping level or `mapper.map()`. No hierarchy jargon.
- FR-4: `CanonicalMapper.get_mapping(mode=None)` returns a dict (default), HTML table (`mode='html'`), or text table (`mode='text'`).
- FR-5: Labels mapped to `None` pass through unchanged to the evaluator. They will always produce FP or FN since they won't match anything on the other side.
- FR-6: `CanonicalMapper.from_dataset()` and `CanonicalMapper.from_results_data_frame()` are removed.
- FR-7: `BaseEvaluator` has no `entity_mapping` parameter, no `_normalize_entity_for_comparison()`, no `_to_io()`, and no `compare_by_io`. All entity mapping is the mapper's responsibility.
- FR-7a: Passing a non-`None` `model` to `BaseEvaluator.__init__` raises `DeprecationError` (hard stop — mirrors `evaluate_all()`). The standard construction is `SpanEvaluator()` / `TokenEvaluator()` with no arguments. `calculate_score_on_df(results_df)` is the primary entry point. `evaluate_sample()` emits a `DeprecationWarning` (soft) but still executes.
- FR-8: `TokenEvaluator.calculate_score_on_df(results_df)` returns an `EvaluationResult`, matching `SpanEvaluator`'s interface.
- FR-9: `evaluate_all()` raises `DeprecationError` with migration instructions. `get_results_dataframe()` emits `DeprecationWarning`.
- FR-10: `EvaluationResult` remains the return type for all evaluators and is compatible with `Plotter`.
- FR-11: Only `BaseModel`, `PresidioAnalyzerWrapper`, and `PresidioRecognizerWrapper` remain in the models package. All other model wrappers and the `[ner]` dependency group are removed.
- FR-12: All dependencies are updated to their latest major versions. `pandas ^3.0.0` and `transformers ^5.3.0` are tested for breaking changes.
- FR-13: Poetry is replaced with uv. `pyproject.toml` uses PEP 621 format. `uv.lock` replaces `poetry.lock`.
- FR-14: `ruff.toml` enforces `line-length=88`, double quotes, and the agreed lint rule set. Pre-commit hooks run `ruff format`, `ruff check`, and unit tests.
- FR-15: Tests are organized into `data_generator/`, `entity_mapping/`, `evaluation/`, `models/`, and `integration/` subdirectories. Integration tests are excluded from pre-commit via `pytest -m "not integration"`.

---

## Non-Goals (Out of Scope)

- No new evaluator types beyond `SpanEvaluator` and `TokenEvaluator`
- No GUI or web UI for entity mapping — `CanonicalMapper` is a Python API (CLI/notebook)
- No automatic model downloading or training
- No changes to the `data_generator` or `dataset_formatters` packages
- No changes to the `experiment_tracking` package
- No new tests for `dataset_formatters` or `experiment_tracking` (known gaps, but not blocking)
- No changes to `EntityHierarchy` internals or the `HIERARCHY` taxonomy
- No support for `InputSample` as a direct input to `CanonicalMapper` (use DataFrame or label list)

---

## Technical Considerations

- **Three-layer architecture**: Model (predict) → Mapper (canonicalize) → Evaluator (score). Each layer communicates via DataFrames. No layer handles another's concerns.
- **DataFrame schema**: 5 columns — `sentence_id` (int), `token` (str), `annotation` (str), `prediction` (str), `start_indices` (int). This is the canonical interface between all layers.
- **BIO prefix stripping**: Moves from the evaluator to the mapper. The mapper already handles this (`entity_mapping/mapper.py`). The evaluator must stop doing it.
- **Major version bumps**: `pandas 3.x` drops deprecated APIs (e.g., `append`, `inplace` changes). `transformers 5.x` may rename or remove classes. Both require careful testing.
- **Poetry → uv migration**: Convert `[tool.poetry.*]` sections to PEP 621 `[project]` format. Do this in the same PR as dependency updates (US-013 + US-014) to minimize conflicts.
- **Test reorganization**: Use a directory-level `conftest.py` in `tests/integration/` to auto-apply the `integration` marker to all tests in that directory, rather than decorating each test individually.
- **Existing `CanonicalMapper` code**: Core resolution logic (EXACT, COUNTRY, COUNTRY_FALLBACK, FUZZY, PENDING tiers), `EntityHierarchy`, `map()`, and `get_mapping()` are already implemented. The main new work is `get_mapped_results_dataframe()` and `hierarchy` param.

---

## Success Metrics

- The 5-step pipeline (load → predict → map → evaluate → plot) can be executed in 5 lines of user code
- All evaluation notebooks (4, 5, 6) run end-to-end without errors using the new pipeline
- `pytest -m "not integration"` completes in under 30 seconds
- Zero entity mapping logic remains in the evaluation package
- `uv sync && uv run pytest` works from a clean checkout

---

## Open Questions

- Should `get_mapped_results_dataframe()` warn when it encounters labels that resolve to PENDING, or silently add them to `mapper.pending`?
- Should the migration guide live in `docs/` or in the repo root as `MIGRATION.md`?
- What version number should the redesigned release use? (semver major bump suggested given breaking changes)

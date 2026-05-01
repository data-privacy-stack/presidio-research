# AGENTS.md — Codebase Patterns

## Project Setup
- Python managed with `uv` — always use `uv run` for Python, never bare `python` or `pip`
- Virtualenv at `.venv/`
- Quality checks: `uv run ruff check --fix`, `uv run ruff format`, `uv run pytest`
- Fast entity-mapping tests: `pytest tests/entity_mapping/`

## CanonicalMapper
- Single-phase (Identify-only) — no projection phase
- `analyze(results_df, min_severity='WARNING')` — COLLISION_SAME_BRANCH (INFO) hidden unless `min_severity='INFO'`
- `get_mapped_results_dataframe()` returns `MappedResults` (frozen dataclass with `.original`, `.binary`, `.branch`, `.detailed`)
- `get_mapping()` returns `{label: resolved}` dict — UNRESOLVED labels excluded
- `get_issues()` filters by `_min_severity` — call after `analyze()` or `map()`

## Five Issue Types (IssueType enum)
- UNRESOLVED (ERROR, blocking)
- COLLISION_CROSS_BRANCH (WARNING, blocking)
- PREDICTION_ONLY (WARNING, blocking)
- DATASET_ONLY (WARNING, non-blocking)
- COLLISION_SAME_BRANCH (INFO, non-blocking)

## _Resolution dataclass fields
- `tier` — identification tier (EXACT, COUNTRY, COUNTRY_FALLBACK, FUZZY, UNRESOLVED)
- `resolved` — resolved canonical entity name (or None for UNRESOLVED)
- `score` — fuzzy match score (0.0–1.0, None for non-FUZZY tiers)
- NO `canonical`, `projected`, or `projection_type` fields

## IssueSeverity enum
- Values are lowercase: `'error'`, `'warning'`, `'info'`
- Use `.lower()` when converting from string; do NOT use `.upper()`

## Hierarchical Evaluation
- `calculate_hierarchical_scores(mapped_results: MappedResults)` returns `{"binary", "branch", "detailed"}`
- NOT `{"L0", "L1", "L2"}` — use the string level names
- `MappedResults` is in `presidio_evaluator/entity_mapping/data_objects.py`

## level_helpers
- `to_binary(label)` and `to_branch(label, hierarchy)` are pure functions
- `level_helpers.py` has no imports from `mapper.py` — no circular deps
- For branch lookups use `EntityHierarchy(canonical_depth=10)`

## Notebooks
- NB4: `4_Evaluate_Presidio_Analyzer.ipynb` — standard Presidio evaluation
- NB5: `5_Evaluate_Custom_Presidio_Analyzer.ipynb` — custom model evaluation
- NB6: `6_Interactive_Entity_Mapping.ipynb` — interactive mapping tutorial
- All notebooks use `mapped_results = mapper.get_mapped_results_dataframe()` (not `mapped_df`)
- `edit_notebook_file` requires VSC cell IDs (`#VSC-xxxx`) — use `copilot_getNotebookSummary` to get current IDs

## Git
- Pre-commit hooks: ruff-format, ruff-check, pytest (unit tests only)
- Branch: `ralph/canonical-mapper-single-phase`

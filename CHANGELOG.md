# CHANGELOG

## Unreleased

### Bug Fixes

- **`add_alias()` no longer rejects a valid alias on `LICENSE`** — the rollback guard added alongside branch aliases asked where the *target's name* resolves and compared that to where the new alias landed. Those are the same question for every node except one: the hierarchy declares `LICENSE` both as a canonical leaf under `EMPLOYMENT` and as an alias of `PROFESSIONAL_LICENSE`, so `canonicalize("LICENSE")` is `"PROFESSIONAL_LICENSE"` and a brand-new alias correctly attached to the `LICENSE` leaf looked like a conflict. `add_alias("LICENSE", "PROF_LIC")` raised `ValueError` claiming the alias "already resolves to 'LICENSE'" — describing the mapping the call had itself created one line earlier. The guard now derives the target from the node's path in the tree rather than from a name lookup, so it is unaffected by another node claiming the same name. Verified across all 126 canonical entities: previously 1 rejected a fresh alias, now none do.
- **`add_alias()` no longer silently steals an alias from another entity** — `add_alias("LOCATION", "EMAIL")` was accepted and quietly re-pointed `EMAIL` from `EMAIL_ADDRESS` to `LOCATION`, invalidating every corpus already annotated with that label. Whether the theft succeeded depended only on dictionary ordering. An alias already resolving somewhere other than the requested target is now rejected before the hierarchy is touched.
- **A rejected `add_alias()` no longer logs a spurious shadow warning** — the speculative rebuild ran the branch-alias shadow check while the doomed alias was still attached, so a failed call logged a warning about a hierarchy state that was discarded microseconds later, phrased as though `definitions.py` were at fault. Shadow warnings now come only from construction, which is what they are for.

### Removals

- **`_to_l1()` and `_to_l0()` removed from `base_evaluator`** — superseded by `EntityHierarchy.to_branch()` / `to_binary()`, and callerless. `_to_l1` was a copy of the branch projection that did not learn alias resolution when `to_branch()` did, leaving the two disagreeing on 434 of 570 labels — including `LOC`, which `_to_l1` bucketed under a `LOC` branch of its own rather than `LOCATION`, the exact mis-bucketing branch aliases exist to remove. Nothing called it, so nothing miscounted; it is deleted rather than fixed so the divergence cannot be reintroduced by a future import.

## Version 0.3.2

### Features

- **Branch-level aliases** — non-leaf hierarchy nodes can now declare raw aliases via a reserved `_aliases` key (e.g. `"LOCATION": {"_aliases": ["LOC"], ...}`), mirroring the alias lists that leaf nodes already have. `add_alias()` on a branch node now records the alias instead of creating a spurious child leaf. The reserved key is skipped by every tree-walk, so it never becomes a canonical entity.

### Breaking Changes

- **`LOC`, `ORG` and `PER` are no longer canonical entities** — they were empty leaf nodes under `LOCATION`/`ORGANIZATION`/`PERSON` > `NAME` and are now branch-level aliases of `LOCATION`/`ORGANIZATION`/`PERSON`. Coarse dataset labels like TAB's `LOC`/`ORG`/`PER` therefore match a model's `LOCATION`/`ORGANIZATION`/`PERSON` at the exact (leaf) level, not only at the branch level. Concretely:
  `LOC` and `ORG` were depth-3 canonical leaves, but `PER` sat one level deeper, under `PERSON > NAME`, so it does not follow the same before/after as the other two. Taking them separately:
  - `canonicalize("LOC")` returns `"LOCATION"` (was `"LOC"`), and likewise `canonicalize("ORG")`.
  - `canonicalize("PER")` returns `"PERSON"` — **was `"NAME"`**, not `"PER"`.
  - `LOC` and `ORG` no longer appear in `all_canonical_entities` or `canonical_to_branch`. `PER` never did, being below the canonical depth.
  - `get_depth("LOC")` returns `2` (was `3`), because `LOC` now denotes the depth-2 `LOCATION` branch. `get_depth("PER")` returns `2` — it previously **raised `EntityNotMappedError`**.
  - `CanonicalMapper.map()` no longer accepts `LOC`/`ORG`/`PER` as resolution *targets*, since targets must be canonical entities. Such mappings are also no longer needed — the labels resolve on their own.
  - `to_branch("LOC")` returns `"LOCATION"`, unchanged.
  - **`to_branch("PER")` returns `"PERSON"`, where it previously returned `"PER"`.** This one is a genuine behaviour change, not a no-op: anyone bucketing a `PER`-annotated corpus by branch gets a different answer after upgrading. `PER` used to fall through `to_branch` unresolved and form a bucket of its own; it now joins `PERSON`.

### Behavior Changes

- **`to_branch()` and `get_depth()` now resolve raw aliases**, not just canonical names. Previously a raw alias (e.g. `COMPANYNAME`, `QQ`) was passed through unchanged by `to_branch` and raised in `get_depth`; both now resolve it first. Unknown labels are still returned as-is by `to_branch`.
- **`add_alias()` accepts an alias as its subject**, so `add_alias("LOC", ...)` works as well as `add_alias("LOCATION", ...)`.
- **`add_alias()` now raises `ValueError` instead of silently no-opping** when the alias is already claimed by a descendant of the target (e.g. adding `CITY` to the `LOCATION` branch, where `CITY` already resolves to `ADDRESS`). The hierarchy is left unmodified — an alias the target already owns is preserved. It also raises `KeyError` if the reserved `_aliases` key is passed as the entity name.
- **A branch alias shadowed by one of its own descendants logs a warning at construction time**, so collisions declared statically in `definitions.py` are no longer silent.

### Bug Fixes

- **Span merging no longer depends on the DataFrame index** — `SpanEvaluator` mixed sentence-relative token positions with DataFrame index labels when checking whether two same-type spans are adjacent. With the global index produced by `predict_dataset()`, the between-tokens lookup read the wrong rows — or none at all — for every sentence except the one starting at row 0, and an empty lookup counts as "adjacent", silently merging same-type spans separated by regular words (e.g. the two PERSON spans in "John visited Berlin with Mary" became one). Span counts (`num_annotated`, `num_predicted`, `true_positives`) were deflated symmetrically for gold and predictions, so headline precision/recall could still look plausible. The evaluator now uses sentence-relative positions throughout and produces identical results for any DataFrame index.

## Version 0.3.1

### Bug Fixes

- **Token-based span IoU is now position-aware** — `SpanEvaluator` with `char_based=False` previously compared sets of token strings, so identical words at different positions were wrongly treated as overlapping (e.g. annotating the first "Michael" in "Michael met Michael" while predicting the second counted as a true positive, and "Human Rights" vs "Civil Rights" had a non-zero IoU through the shared word "Rights"). Token-level IoU now compares `(position, token)` pairs; `Span` gained an optional `normalized_start_indices` field to support this. Spans built without per-token indices fall back to string comparison guarded by a positional-overlap check.

## Version 0.3

> **Migration guide:** See [docs/migration-guide.md](docs/migration-guide.md) for step-by-step upgrade instructions.

### Breaking Changes

- **`evaluate_all()` removed** — raises `DeprecationError` at runtime. Replace with the three-step pipeline: `predict_dataset()` → `CanonicalMapper.get_mapped_results_dataframe()` → `calculate_score_on_df()`.
- **`entity_mapping` parameter removed** from `SpanEvaluator`, `TokenEvaluator`, and `BaseEvaluator` — entity mapping is now the responsibility of `CanonicalMapper`.
- **`compare_by_io` parameter removed** from evaluator constructors — BIO/BILUO prefix stripping is now performed by `CanonicalMapper`.
- **`BaseEvaluator.from_dataset()` removed** — use `model.predict_dataset(dataset)` directly.
- **Non-Presidio model wrappers removed**: `FlairModel`, `SpacyModel`, `StanzaModel`, `AzureAITextAnalyticsWrapper`. Add models directly through Presidio to evaluate them.
- **Minimum Python version raised to 3.11** (was 3.10) — required by `numpy >= 2.4.0`.
- **Package manager changed from Poetry to uv** — install with `uv sync`, run with `uv run`.

### New Features

- **`BaseModel.predict_dataset(dataset)`** — runs the model on a list of `InputSample` objects and returns a 5-column DataFrame (`sentence_id`, `token`, `annotation`, `prediction`, `start_indices`).
- **`CanonicalMapper`** — replaces `EntityMappingHelper` with an improved four-tier auto-resolution strategy (`EXACT`, `COUNTRY`, `FUZZY`, `PENDING`). Key methods:
  - `CanonicalMapper.from_dataset(dataset)` — builds a mapper from dataset labels.
  - `mapper.get_mapped_results_dataframe(results_df)` — applies entity mapping to a predictions DataFrame.
  - `mapper.get_mapping(mode='html' | 'text')` — returns the final `{raw_label: canonical | None}` dict.
  - `mapper.map({"LABEL": "CANONICAL"})` — manually resolve pending labels.
  - `mapper.render_html()` — display the resolution audit table in Jupyter.
- **`TokenEvaluator.calculate_score_on_df(results_df)`** — score token-level predictions from a DataFrame.
- **`SpanEvaluator.calculate_score_on_df(per_type, results_df)`** — score span-level predictions from a DataFrame.
- **Ruff** — added as the project linter and formatter (`ruff.toml` at project root).
- **Pre-commit hooks** — `ruff format`, `ruff check`, and `pytest` run automatically before every commit (`.pre-commit-config.yaml`).
- **Test reorganisation** — tests are now grouped by topic (`tests/data_generator/`, `tests/entity_mapping/`, `tests/evaluation/`, `tests/models/`, `tests/integration/`). Integration tests are tagged with `pytest.mark.integration`.

### Deprecations

- **`evaluator.get_results_dataframe()`** — soft `DeprecationWarning` emitted at runtime. Replace with `model.predict_dataset(dataset)`.

### CI/CD

- **PyPI release workflow** — added `.github/workflows/publish.yml`, which builds and publishes the package to PyPI via Trusted Publishing (OIDC) when a GitHub Release is published. Replaces the retired Azure DevOps `publish-to-pypi` pipeline.



## Version 0.2.5

### Improvements
 - Introduced a new evaluator, `SpanEvaluator` which compares full spans of annotations and predictions, instead of tokens. ([#141](https://github.com/data-privacy-stack/presidio-research/pull/141))
 - Make Azure SDK as an optional dependency ([#116](https://github.com/data-privacy-stack/presidio-research/pull/116))
 - Add a DF output to evaluation results ([#126](https://github.com/data-privacy-stack/presidio-research/pull/126))
### Bug Fixes
 - Fixed bugs around plotting and experiment tracking (#140) around configuring Presidio in the evaluation loop. ([#155](https://github.com/data-privacy-stack/presidio-research/pull/155))
 - Data generation bug fixes [#113](https://github.com/data-privacy-stack/presidio-research/pull/113)

## Version 0.2.0

### Breaking changes
- Removed notebooks (pseudonomyzation)
- Removed redundant classes `FakerSpan`, `FakerSpanResult` and updated code to use `Span` and `InputSample` respectively, changed `SentenceFaker` to inherit from Faker instead of using composition.
- Removed functions `from_faker_span`, `from_faker_spans_result` `convert_faker_spans` from `InputSample`, as faker spans are now `Span`s so there no need for translation.
- Removed `PresidioDataGenerator` to use `PresidioSentenceFaker` instead 
- Removed support for CRF models
- Removed the `FlairTrainer` class, please refer to the official Flair documentation for training Flair models
- Removed CRF as the package used is no longer maintained

### Improvements
- Improved evaluation notebooks: Notebook 4 shows a vanilla Presidio evaluation, notebook 5 shows a more customized Presidio with improved accuracy (#103)
- Removed the Pseudonomyzation notebook as there is a more advanced approach within Presidio (#103)
- Added the ability to use generic entities and skip words (#103)
- Added the ability to do faster batch predict (#103)
- Added sample_id to be able to reproduce the full sample (#103)
- Fixed issue with hospital provider networking (#103)

### Bug Fixes

- Fix translation of Input Sample tags (#88)
- Fix Presidio wrapper to call predict with a language parameter (#79)

### Other Changes
- Updates to all classes inheriting from BaseModel, as the predict signature has changed (now containing **kwargs) (#92)
- Added Poetry instead of setup.py (#91)
- Rename UsDriverLicenseProvider.driver_license to us_driver_license (#90)
- Removed redundant classes FakerSpan, FakerSpanResult and updated code to use Span and InputSample respectively instead (#72)
- Changed SentenceFaker to inherit from Faker instead of using composition (#72)
- Simplified the use of SentenceFaker in the default option (RecordGenerator is instantiated if records are passed, otherwise a SpanGenerator is instantiated) (#72)
- Updates to unit tests to support this change (#72)
- Updates to poetry to include the config in setup.cfg, setup.py, and pytest.ini (#72)

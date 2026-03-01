# CHANGELOG

## Unreleased

### Breaking changes
- **Entity mapping refactored**: `entity_mapping` parameter removed from `BaseModel` and now REQUIRED in evaluator constructors (`BaseEvaluator`, `SpanEvaluator`, `TokenEvaluator`). This provides better separation of concerns between model predictions and evaluation logic.
  - Old: `model = BaseModel(entity_mapping={...})`  
  - New: `evaluator = SpanEvaluator(model=model, entity_mapping={...})`
- Removed methods from `BaseModel`: `align_entity_types()`, `align_prediction_types()`, and `prepare_dataset()` - entity alignment is now handled during evaluation

### Improvements
- **EntityMappingHelper**: New interactive helper class for automatic entity mapping between dataset and model entities. Includes:
  - Automatic entity detection from both dataset and model
  - Suggested mappings with manual override capability
  - Interactive HTML-based review interface
  - Dataset filtering based on entity exclusions
- **Smart entity filtering**: `calculate_score()` in `SpanEvaluator` and `TokenEvaluator` now defaults the `entities` parameter to `entities_to_keep` (from evaluator constructor) when not explicitly provided, ensuring consistent filtering across token-level and span-level metrics
- Enhanced entity normalization in comparison logic with support for None values (identity mapping)
- New entity mapping utilities: `DictEntityMapper`, `SemanticEntityMapper`, `HybridEntityMapper`, and `create_presidio_mapper()` for flexible entity mapping strategies
- Updated notebooks (4 & 5) to demonstrate new `EntityMappingHelper` workflow
- Improved logging: replaced print statements with proper logger calls
- **Replaced example model**: New model yields better accuracy than before.

## Version 0.2.5

### Improvements
 - Introduced a new evaluator, `SpanEvaluator` which compares full spans of annotations and predictions, instead of tokens. ([#141](https://github.com/microsoft/presidio-research/pull/141))
 - Make Azure SDK as an optional dependency ([#116](https://github.com/microsoft/presidio-research/pull/116))
 - Add a DF output to evaluation results ([#126](https://github.com/microsoft/presidio-research/pull/126))
### Bug Fixes
 - Fixed bugs around plotting and experiment tracking (#140) around configuring Presidio in the evaluation loop. ([#155](https://github.com/microsoft/presidio-research/pull/155))
 - Data generation bug fixes [#113](https://github.com/microsoft/presidio-research/pull/113)

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

"""Script to update evaluation notebooks to use the new pipeline."""

import json
from pathlib import Path


def set_cell_source(nb, cell_id, new_source_str):
    """Update a cell by its exact ID, setting source as a list of lines."""
    lines = [line + "\n" for line in new_source_str.split("\n")]
    # Remove trailing \n from last line if it was empty
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    # Ensure last line doesn't end with newline (notebook convention)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    for cell in nb["cells"]:
        if cell["id"] == cell_id:
            cell["source"] = lines
            cell["outputs"] = []
            cell["execution_count"] = None
            return True
    print(f"WARNING: Cell {cell_id} not found!")
    return False


###############################################
# Notebook 4
###############################################
nb4_path = Path("notebooks/4_Evaluate_Presidio_Analyzer.ipynb")
with nb4_path.open() as f:
    nb4 = json.load(f)

# Verify cell IDs exist
print("Notebook 4 cell IDs:")
for c in nb4["cells"]:
    print(f"  {c['id']}: {''.join(c['source'])[:60]!r}")

# Cell ae85cae9: imports
set_cell_source(
    nb4,
    "ae85cae9",
    """\
import json
from collections import Counter
from pathlib import Path
from pprint import pprint

import pandas as pd
from presidio_analyzer import AnalyzerEngine

from presidio_evaluator import InputSample
from presidio_evaluator.entity_mapping import CanonicalMapper
from presidio_evaluator.evaluation import ModelError, Plotter, SpanEvaluator
from presidio_evaluator.experiment_tracking import get_experiment_tracker
from presidio_evaluator.models import PresidioAnalyzerWrapper

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

%reload_ext autoreload
%autoreload 2""",
)

# Cell aa44f70a7c7aa3f0: section 4 markdown
set_cell_source(
    nb4,
    "aa44f70a7c7aa3f0",
    """\
## 4. Entity Mapping with CanonicalMapper

**Entity mapping** translates the dataset's entity types to match the model's entity types. This is crucial because:
- Datasets and models often use different entity naming conventions
- For example: Dataset might have `STREET_ADDRESS` while Presidio uses `LOCATION`
- Proper mapping ensures accurate evaluation

`CanonicalMapper` auto-resolves labels through four tiers: `EXACT`, `COUNTRY`, `FUZZY`, `PENDING`.
Labels still `PENDING` after auto-resolution must be manually resolved with `.map()`.""",
)

# Cell ff2e676f44f72e4e: auto-resolve
set_cell_source(
    nb4,
    "ff2e676f44f72e4e",
    """\
# Auto-resolve entity labels from the dataset
mapper = CanonicalMapper.from_dataset(dataset)

# Review the auto-resolution (shows tier per label: EXACT / COUNTRY / FUZZY / PENDING)
mapper.render_html()""",
)

# Cell 88c6c806: manual mapping + get_mapping
set_cell_source(
    nb4,
    "88c6c806",
    """\
# Manually resolve labels the mapper couldn't auto-match
mapper.map({
    "AGE": "DATE_TIME",
    "TITLE": "PERSON",
    "ORGANIZATION": None,   # suppress — model doesn't detect it
})

# Review the final mapping
mapper.render_html()

# Get the final entity mapping dict
entity_mapping = mapper.get_mapping()
entities_to_keep = {v for v in entity_mapping.values() if v is not None}

print("\\n✓ Entity mapping complete")
print(f"Mapping: {len(entity_mapping)} entities")
print(f"Evaluating: {sorted(entities_to_keep)}")""",
)

# Cell 29d39ff1-4f14-4e32-ae84-ecc6c739f829: evaluator setup
set_cell_source(
    nb4,
    "29d39ff1-4f14-4e32-ae84-ecc6c739f829",
    """\
# Set up the experiment tracker to log the experiment for reproducibility
experiment = get_experiment_tracker()

# Wrap the analyzer
wrapped_analyzer = PresidioAnalyzerWrapper(analyzer_engine=analyzer_engine)

# Create the evaluator object (no entity_mapping — mapping is handled by CanonicalMapper)
evaluator = SpanEvaluator(
    model=wrapped_analyzer,
    entities_to_keep=list(entities_to_keep),
    iou_threshold=0.75,
)

# Track model and dataset params
params = {"dataset_name": dataset_name, "model_name": evaluator.model.name}
params.update(evaluator.model.to_log())
experiment.log_parameters(params)
experiment.log_dataset_hash(dataset)
experiment.log_parameter("entity_mappings", json.dumps(entity_mapping))""",
)

# Cell cf65af8f: run experiment
set_cell_source(
    nb4,
    "cf65af8f",
    """\
%%time
## Run experiment — three-step pipeline: predict → map → score

# Step 1: Get raw predictions from the model
results_df = wrapped_analyzer.predict_dataset(dataset)

# Step 2: Apply entity mapping (align dataset labels to model labels)
mapped_df = mapper.get_mapped_results_dataframe(results_df)

# Step 3: Score the mapped predictions
results = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)

# Track experiment results
experiment.log_metrics(results.to_log())
entities, confmatrix = results.to_confusion_matrix()
experiment.log_confusion_matrix(matrix=confmatrix, labels=entities)

# end experiment
experiment.end()""",
)

with nb4_path.open("w") as f:
    json.dump(nb4, f, indent=1, ensure_ascii=False)
print("Notebook 4 saved successfully")
print(
    "  EntityMappingHelper remaining:",
    sum(1 for c in nb4["cells"] if "EntityMappingHelper" in "".join(c["source"])),
)
print(
    "  CanonicalMapper present:",
    sum(1 for c in nb4["cells"] if "CanonicalMapper" in "".join(c["source"])),
)
print(
    "  evaluate_all remaining:",
    sum(1 for c in nb4["cells"] if "evaluate_all" in "".join(c["source"])),
)
print(
    "  predict_dataset present:",
    sum(1 for c in nb4["cells"] if "predict_dataset" in "".join(c["source"])),
)


###############################################
# Notebook 5
###############################################
nb5_path = Path("notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb")
with nb5_path.open() as f:
    nb5 = json.load(f)

# Cell 847acd88: Title markdown - update step list
set_cell_source(
    nb5,
    "847acd88",
    """\
## Evaluate a custom Presidio Analyzer using the Presidio Evaluator framework

This notebook demonstrates how to evaluate a Presidio instance using the presidio-evaluator framework. It builds upon [example 4](4_Evaluate_Presidio_Analyzer.ipynb), with changes to the `PresidioAnalyzer` instance to improve detection accuracy. For more information on customizing the Presidio Analyzer, see the [Presidio Analyzer documentation](https://microsoft.github.io/presidio/analyzer/) or this [tutorial](https://microsoft.github.io/presidio/tutorial/).

Steps:
1. Load dataset from file
2. Simple dataset statistics
3. Define the AnalyzerEngine object (and its parameters)
4. Entity mapping with CanonicalMapper
5. Set up the Evaluator object
6. Run experiment
7. Evaluate results
8. Error analysis""",
)

# Cell 574abb285875c503: imports - replace EntityMappingHelper with CanonicalMapper
set_cell_source(
    nb5,
    "574abb285875c503",
    """\
import json
import warnings
from collections import Counter
from pathlib import Path
from pprint import pprint

warnings.filterwarnings("ignore")

import pandas as pd
from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
)
from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine

from presidio_evaluator import InputSample
from presidio_evaluator.entity_mapping import CanonicalMapper
from presidio_evaluator.evaluation import ModelError, Plotter, SpanEvaluator
from presidio_evaluator.experiment_tracking import get_experiment_tracker
from presidio_evaluator.models import PresidioAnalyzerWrapper

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

%reload_ext autoreload
%autoreload 2
%matplotlib inline""",
)

# Cell 609369c1: section 4 markdown
set_cell_source(
    nb5,
    "609369c1",
    """\
## 4. Entity Mapping with CanonicalMapper

Use `CanonicalMapper` to align dataset entity labels with model entity labels.

**Why entity mapping?** The dataset may use entity labels like `STREET_ADDRESS` or `GPE` (geopolitical entity), while the model outputs `LOCATION`. `CanonicalMapper` auto-resolves labels through four tiers: `EXACT`, `COUNTRY`, `FUZZY`, `PENDING`. Labels still PENDING must be resolved manually with `.map()`.""",
)

# Cell c1ad6508: CanonicalMapper.from_dataset
set_cell_source(
    nb5,
    "c1ad6508",
    """\
# Auto-resolve entity labels from the dataset
mapper = CanonicalMapper.from_dataset(dataset)

# Review the auto-resolution (shows tier per label)
mapper.render_html()""",
)

# Cell d32db26f: manual mapping (ZIP_CODE)
set_cell_source(
    nb5,
    "d32db26f",
    """\
# Manually resolve any pending labels
# ZIP_CODE is detected by our custom recognizer — map it to itself
mapper.map({
    "ZIP_CODE": "ZIP_CODE",
})

# Get the final mapping dict
entity_mapping = mapper.get_mapping()
entities_to_keep = {v for v in entity_mapping.values() if v is not None}

print(f"Entity mapping ({len(entity_mapping)} labels):")
pprint(entity_mapping, compact=True)
print(f"\\nModel entities to evaluate ({len(entities_to_keep)}):")
pprint(sorted(entities_to_keep), compact=True)""",
)

# Cell 115cbba4: final mapping summary
set_cell_source(
    nb5,
    "115cbba4",
    """\
print("✓ Entity mapping complete")
print(f"Mapping: {len(entity_mapping)} entities")
print(f"Evaluating: {sorted(entities_to_keep)}")""",
)

# Cell 8076c4c80070f157: evaluator setup (was USE_FILTERED_ENTITIES + SpanEvaluator with entity_mapping)
set_cell_source(
    nb5,
    "8076c4c80070f157",
    """\
# Set up the experiment tracker
experiment = get_experiment_tracker()

# Wrap the analyzer
wrapped_analyzer = PresidioAnalyzerWrapper(
    analyzer_engine=analyzer_engine,
    score_threshold=analyzer_engine.default_score_threshold,
    language="en",
)

# Create the evaluator (no entity_mapping — handled by CanonicalMapper)
evaluator = SpanEvaluator(
    model=wrapped_analyzer,
    entities_to_keep=list(entities_to_keep),
    iou_threshold=0.75,
)

# Track experiment parameters
params = {"dataset_name": dataset_name, "model_name": evaluator.model.name}
params.update(evaluator.model.to_log())
experiment.log_parameters(params)
experiment.log_dataset_hash(dataset)
experiment.log_parameter("entity_mappings", json.dumps(entity_mapping))""",
)

# Cell 2abf0c96: run experiment
set_cell_source(
    nb5,
    "2abf0c96",
    """\
%%time

## Run experiment — three-step pipeline: predict → map → score

# Step 1: Get raw predictions from the model
results_df = wrapped_analyzer.predict_dataset(dataset)

# Step 2: Apply entity mapping (align dataset labels to model labels)
mapped_df = mapper.get_mapped_results_dataframe(results_df)

# Step 3: Score the mapped predictions
results = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)

# Track experiment results
experiment.log_metrics(results.to_log())
entities, confmatrix = results.to_confusion_matrix()
experiment.log_confusion_matrix(matrix=confmatrix, labels=entities)

# end experiment
experiment.end()

# Note that the experiment params and metrics are saved locally""",
)

with nb5_path.open("w") as f:
    json.dump(nb5, f, indent=1, ensure_ascii=False)
print("Notebook 5 saved successfully")
print(
    "  EntityMappingHelper remaining:",
    sum(1 for c in nb5["cells"] if "EntityMappingHelper" in "".join(c["source"])),
)
print(
    "  CanonicalMapper present:",
    sum(1 for c in nb5["cells"] if "CanonicalMapper" in "".join(c["source"])),
)
print(
    "  evaluate_all remaining:",
    sum(1 for c in nb5["cells"] if "evaluate_all" in "".join(c["source"])),
)
print(
    "  predict_dataset present:",
    sum(1 for c in nb5["cells"] if "predict_dataset" in "".join(c["source"])),
)

# Fix remaining EntityMappingHelper comment in NLP engine cell of nb5
for c in nb5["cells"]:
    if c["id"] == "313b508f-e901-40b9-b575-c7fb8a794652":
        src = "".join(c.get("source", []))
        new_src = src.replace(
            "# Create the NLP engine without entity mapping (will be handled by EntityMappingHelper)",
            "# Create the NLP engine without entity mapping (will be handled by CanonicalMapper)",
        )
        c["source"] = [new_src]
        c["outputs"] = []
        c["execution_count"] = None
        break

with nb5_path.open("w") as f:
    json.dump(nb5, f, indent=1, ensure_ascii=False)
print(
    "Notebook 5 final save: EntityMappingHelper remaining:",
    sum(1 for c in nb5["cells"] if "EntityMappingHelper" in "".join(c["source"])),
)

###############################################
# Notebook 6
###############################################
nb6_path = Path("notebooks/6_Interactive_Entity_Mapping.ipynb")
with nb6_path.open() as f:
    nb6 = json.load(f)

# Cell e4132a05: evaluator setup (remove entity_mapping from SpanEvaluator)
set_cell_source(
    nb6,
    "e4132a05",
    """\
wrapped_analyzer = PresidioAnalyzerWrapper(analyzer_engine=analyzer, language="en")

# Strategy 1: Filtered entities (recommended — avoids spurious FPs)
evaluator_filtered = SpanEvaluator(
    model=wrapped_analyzer,
    entities_to_keep=list(model_entities_to_use),
)

# Strategy 2: All model entities
evaluator_full = SpanEvaluator(
    model=wrapped_analyzer,
    entities_to_keep=None,
)

print("Ready to evaluate. Run the next cell to compare strategies.")""",
)

# Cell ae5f8d35: run experiment with new pipeline
set_cell_source(
    nb6,
    "ae5f8d35",
    """\
%%time

subset = dataset[:100]

# Step 1: Predict
results_df_filtered = wrapped_analyzer.predict_dataset(subset)
results_df_full = wrapped_analyzer.predict_dataset(subset)

# Step 2: Map
mapped_filtered = mapper.get_mapped_results_dataframe(results_df_filtered)
mapped_full = mapper.get_mapped_results_dataframe(results_df_full)

# Step 3: Score
scores_filtered = evaluator_filtered.calculate_score_on_df(per_type=True, results_df=mapped_filtered)
scores_full = evaluator_full.calculate_score_on_df(per_type=True, results_df=mapped_full)

print("\\n=== Strategy Comparison (100 samples) ===")
print(f"\\n{'Metric':<20} {'Filtered':>12} {'Full Model':>12}")
print("-" * 44)
print(
    f"{'PII Precision':<20} {scores_filtered.pii_precision:>12.3f} {scores_full.pii_precision:>12.3f}"
)
print(
    f"{'PII Recall':<20} {scores_filtered.pii_recall:>12.3f} {scores_full.pii_recall:>12.3f}"
)
print(f"{'PII F1':<20} {scores_filtered.pii_f:>12.3f} {scores_full.pii_f:>12.3f}")""",
)

# Cell 2e585409: quick reference - fix entity_mapping=entity_mapping in SpanEvaluator example
set_cell_source(
    nb6,
    "2e585409",
    '''\
print("""
=== CanonicalMapper Quick Reference ===

# Build from a dataset (InputSample list)
mapper = CanonicalMapper.from_dataset(dataset)
# Returns a dict directly when all labels resolve; otherwise returns a CanonicalMapper.

# Inspect auto-resolution
mapper.render_html()          # HTML audit table (Jupyter)
mapper._print_text()          # Plain-text fallback

# Check pending labels
mapper.pending                # list of unresolved labels

# Assign manually
mapper.map({
    "MY_LABEL": "PERSON",     # resolve to canonical entity
    "NOISE":    None,          # suppress from evaluation
})

# Interactive terminal resolution (shows ranked suggestions)
mapper.resolve_interactively()

# Get the final mapping dict (raises IncompleteMapping if pending is non-empty)
entity_mapping = mapper.get_mapping()  # {raw_label: canonical | None}

# New 3-step pipeline
results_df = model.predict_dataset(dataset)
mapped_df = mapper.get_mapped_results_dataframe(results_df)
scores = evaluator.calculate_score_on_df(per_type=True, results_df=mapped_df)

# Use with SpanEvaluator (no entity_mapping parameter)
entities_to_keep = {v for v in entity_mapping.values() if v is not None}
evaluator = SpanEvaluator(
    model=wrapped_model,
    entities_to_keep=list(entities_to_keep),
)

# Customise the taxonomy
from presidio_evaluator.entity_mapping import EntityHierarchy
h = EntityHierarchy()
h.add_alias("EMAIL_ADDRESS", "E_MAIL")
mapper = CanonicalMapper.from_dataset(dataset, hierarchy=h)
""")''',
)

with nb6_path.open("w") as f:
    json.dump(nb6, f, indent=1, ensure_ascii=False)
print("Notebook 6 saved successfully")
print(
    "  evaluate_all remaining:",
    sum(1 for c in nb6["cells"] if "evaluate_all" in "".join(c["source"])),
)
print(
    "  entity_mapping=entity_mapping remaining:",
    sum(
        1
        for c in nb6["cells"]
        if "entity_mapping=entity_mapping" in "".join(c["source"])
    ),
)
print(
    "  predict_dataset present:",
    sum(1 for c in nb6["cells"] if "predict_dataset" in "".join(c["source"])),
)

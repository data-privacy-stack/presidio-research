"""Update notebook 5 for the new two-phase CanonicalMapper API (US-019)."""

import json

nb = json.load(open("notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb"))

# Cell 2: imports — add IncompleteMapping
nb["cells"][2]["source"] = [
    "import json\n",
    "import warnings\n",
    "from collections import Counter\n",
    "from pathlib import Path\n",
    "from pprint import pprint\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import pandas as pd\n",
    "from presidio_analyzer import (\n",
    "    AnalyzerEngine,\n",
    "    Pattern,\n",
    "    PatternRecognizer,\n",
    "    RecognizerRegistry,\n",
    ")\n",
    "from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer\n",
    "from presidio_analyzer.nlp_engine import NerModelConfiguration, TransformersNlpEngine\n",
    "\n",
    "from presidio_evaluator import InputSample\n",
    "from presidio_evaluator.entity_mapping import CanonicalMapper, IncompleteMapping\n",
    "from presidio_evaluator.evaluation import ModelError, Plotter, SpanEvaluator\n",
    "from presidio_evaluator.experiment_tracking import get_experiment_tracker\n",
    "from presidio_evaluator.models import PresidioAnalyzerWrapper\n",
    "\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', None)\n",
    "pd.set_option('display.max_colwidth', None)\n",
    "\n",
    "%reload_ext autoreload\n",
    "%autoreload 2\n",
    "%matplotlib inline\n",
]
nb["cells"][2]["outputs"] = []
nb["cells"][2]["execution_count"] = None

# Cell 22: markdown — update workflow description, remove hierarchy=2 mention
nb["cells"][22]["source"] = [
    "## 5. Review entity mapping\n",
    "\n",
    "`CanonicalMapper` auto-resolves entity labels from the results DataFrame using a two-phase workflow:\n",
    "\n",
    "1. **Identify** - each label is matched to a canonical entity (exact alias, country-prefix, or fuzzy match)\n",
    "2. **Project** - labels are projected onto the *eval surface*, a set of entities at the majority-vote\n",
    "   depth computed from the dataset annotations\n",
    "\n",
    "After calling `analyze()`, inspect issues with `render_html()` or `get_issues()`. Issue types:\n",
    "\n",
    "| Type | Severity | Meaning |\n",
    "|------|----------|---------|\n",
    "| `UNRESOLVED` | ERROR | Label could not be matched - must map or suppress |\n",
    "| `COLLISION_AMBIGUOUS` | WARNING | Depth-2 label maps to multiple depth-3 entities - must pick one |\n",
    "| `COLLISION_CROSS_BRANCH` | WARNING | Label maps across hierarchy branches |\n",
    "| `PREDICTION_ONLY` | WARNING | Label only appears in predictions, not dataset - must map or suppress |\n",
    "| `COLLISION_TRIVIAL` | INFO | Auto-fixed: descendant collapsed to single ancestor |\n",
    "| `DATASET_ONLY` | INFO | Label only in dataset annotations (model never predicts it) |\n",
    "\n",
    "`DATASET_ONLY` is **non-blocking** (INFO severity) - you can call `get_mapped_results_dataframe()`\n",
    "even when `DATASET_ONLY` issues exist. They simply indicate entities the model never predicts.\n",
    "\n",
    "Use `map({label: canonical})` to remap, or `map({label: None})` to suppress a label.\n",
    "Pre-map labels you already know about *before* `analyze()` to avoid them appearing as issues.\n",
]

# Cell 23: code — remove hierarchy=2 from analyze(), keep pre-map pattern
nb["cells"][23]["source"] = [
    "# Pre-map labels we already know need overrides (before analyze)\n",
    "# This avoids them appearing as WARNING/ERROR issues after analysis.\n",
    "mapper = CanonicalMapper()\n",
    "mapper.map({\n",
    "    'EDUCATION_LEVEL': None,  # not in dataset\n",
    "    'OCCUPATION': None,       # not in dataset\n",
    "    'LICENSE_PLATE': None,    # not in dataset\n",
    "})\n",
    "\n",
    "# Analyze — no hierarchy parameter needed.\n",
    "# Eval depth is inferred automatically from the annotation labels in the DataFrame.\n",
    "mapper.analyze(results_df)\n",
    "mapper.render_html()\n",
]
nb["cells"][23]["outputs"] = []
nb["cells"][23]["execution_count"] = None

# Cell 24: code — update issue iteration to use new API (remove resolution_options.mapper_call)
nb["cells"][24]["source"] = [
    "# Review detected issues programmatically\n",
    "severity_icons = {'error': 'ERROR', 'warning': 'WARNING', 'info': 'INFO'}\n",
    "for issue in mapper.get_issues():\n",
    "    icon = severity_icons[issue.severity.value]\n",
    "    labels_str = ', '.join(issue.labels)\n",
    "    print(f'[{icon}] {issue.type.value}: {labels_str}')\n",
    "    if issue.annotation_count or issue.prediction_count:\n",
    "        print(f'  annotations={issue.annotation_count}, predictions={issue.prediction_count}')\n",
    "    print()\n",
]
nb["cells"][24]["outputs"] = []
nb["cells"][24]["execution_count"] = None

# Cell 25: code — remove hierarchy=2, use new issue type names, programmatic resolution
nb["cells"][25]["source"] = [
    "# Resolve all WARNING+ issues programmatically.\n",
    "# PREDICTION_ONLY: labels the model predicts but the dataset never annotates — suppress with None.\n",
    "# COLLISION_AMBIGUOUS: depth-2 labels that match multiple depth-3 entities — must pick one.\n",
    "# DATASET_ONLY is INFO (non-blocking) — no action needed.\n",
    "for issue in mapper.get_issues():\n",
    "    if issue.severity.value in ('warning', 'error'):\n",
    "        if issue.type.value == 'prediction_only':\n",
    "            mapper.map({lbl: None for lbl in issue.labels})\n",
    "        elif issue.type.value == 'collision_ambiguous':\n",
    "            # These are depth-2 Presidio labels — suppress or remap as needed\n",
    "            mapper.map({lbl: None for lbl in issue.labels})\n",
    "        elif issue.type.value == 'unresolved':\n",
    "            mapper.map({lbl: None for lbl in issue.labels})\n",
    "\n",
    "# Re-analyze to refresh issue detection after all map() calls\n",
    "mapper.analyze(results_df)\n",
    "\n",
    "# Get the remapped DataFrame\n",
    "mapped_df = mapper.get_mapped_results_dataframe()\n",
    "mapper.render_html()\n",
]
nb["cells"][25]["outputs"] = []
nb["cells"][25]["execution_count"] = None

with open("notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify no old API references remain
src_all = "\n".join("".join(c["source"]) for c in nb["cells"])
old_terms = [
    "canonical_depth",
    "eval_entities",
    "DEPTH_MISMATCH",
    "SUPPRESSED_WITH_ANNOTATIONS",
    "HIERARCHY_DEPTH_CHANGED",
    "hierarchy=2",
    "hierarchy=3",
    "analyze(results_df, hierarchy",
    "resolution_options[0].mapper_call",
]
print("Checking for old API references:")
for term in old_terms:
    if term in src_all:
        print(f"  FOUND: {term!r}")
    else:
        print(f"  OK (not found): {term!r}")

print(f"\nDone. Total cells: {len(nb['cells'])}")

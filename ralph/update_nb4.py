"""Update notebook 4 for the new two-phase CanonicalMapper API (US-018)."""

import json

nb = json.load(open("notebooks/4_Evaluate_Presidio_Analyzer.ipynb"))

# Cell 2: imports — add IncompleteMapping
nb["cells"][2]["source"] = [
    "import json\n",
    "from collections import Counter\n",
    "from pathlib import Path\n",
    "from pprint import pprint\n",
    "import warnings\n",
    "import pandas as pd\n",
    "from presidio_analyzer import AnalyzerEngine\n",
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
]
nb["cells"][2]["outputs"] = []
nb["cells"][2]["execution_count"] = None

# Cell 15: markdown — update workflow description
nb["cells"][15]["source"] = [
    "## 5. Review entity mapping\n",
    "\n",
    "The dataset and Presidio may not use the same entity labels. `CanonicalMapper` auto-resolves\n",
    "labels using a two-phase workflow:\n",
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
    "Call `map({label: canonical})` to remap, or `map({label: None})` to suppress a label.\n",
    "\n",
    "> **Note:** `get_mapped_results_dataframe()` raises `IncompleteMapping` if any ERROR or WARNING issues remain.\n",
    "\n",
    "For a deeper walkthrough see [Notebook 6 - Interactive Entity Mapping Tutorial](6_Interactive_Entity_Mapping.ipynb).\n",
]

# Cell 16: code — remove canonical_depth, new no-arg API
nb["cells"][16]["source"] = [
    "# Two-phase mapping: no constructor arguments needed.\n",
    "# Eval depth is inferred automatically from the annotation labels in the results DataFrame.\n",
    "mapper = CanonicalMapper()\n",
    "mapper.analyze(results_df)\n",
    "\n",
    "# Inspect all detected issues (render_html() falls back to text outside Jupyter)\n",
    "mapper.render_html()\n",
]
nb["cells"][16]["outputs"] = []
nb["cells"][16]["execution_count"] = None

# Cell 17: code — demonstrate IncompleteMapping before resolution
nb["cells"][17]["source"] = [
    "# get_mapped_results_dataframe() raises IncompleteMapping if WARNING or ERROR issues remain.\n",
    "# Demonstrate by calling it before resolving:\n",
    "try:\n",
    "    mapper.get_mapped_results_dataframe()\n",
    "except IncompleteMapping as e:\n",
    "    print('Blocked - resolve these issues first:')\n",
    "    for issue in mapper.get_issues():\n",
    "        if issue.severity.value in ('warning', 'error'):\n",
    "            print(f'  [{issue.severity.value}] {issue.type.value}: {issue.labels}')\n",
]
nb["cells"][17]["outputs"] = []
nb["cells"][17]["execution_count"] = None

# New cells to insert after cell 17
new_markdown_resolve = {
    "cell_type": "markdown",
    "id": "new-resolve-md",
    "metadata": {},
    "source": [
        "### 5a. Resolve issues\n",
        "\n",
        "Presidio's default model outputs depth-2 labels (`PERSON`, `DATE_TIME`, `LOCATION`, `ORGANIZATION`).\n",
        "When the eval surface is at depth 3, these trigger `COLLISION_AMBIGUOUS`.\n",
        "Use `map()` to pick a depth-3 canonical target.\n",
        "\n",
        "Labels that appear only in predictions (`PREDICTION_ONLY`) with no dataset equivalent should be\n",
        "suppressed with `None` to exclude them from evaluation.\n",
    ],
}

new_code_resolve = {
    "cell_type": "code",
    "execution_count": None,
    "id": "new-resolve-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Resolve COLLISION_AMBIGUOUS: pick a depth-3 canonical target for each depth-2 Presidio label\n",
        "mapper.map({\n",
        "    'PERSON': 'NAME',       # NAME is the primary depth-3 person-name entity\n",
        "    'DATE_TIME': 'DATE',    # DATE covers most date/time predictions\n",
        "    'LOCATION': 'LOC',      # LOC is the generic location entity\n",
        "    'ORGANIZATION': 'ORG', # ORG is the generic organization entity\n",
        "})\n",
        "\n",
        "# Suppress any remaining PREDICTION_ONLY or UNRESOLVED labels\n",
        "still_blocking = [\n",
        "    lbl\n",
        "    for issue in mapper.get_issues()\n",
        "    for lbl in issue.labels\n",
        "    if issue.severity.value in ('warning', 'error')\n",
        "]\n",
        "if still_blocking:\n",
        "    print('Suppressing:', still_blocking)\n",
        "    mapper.map({lbl: None for lbl in still_blocking})\n",
        "\n",
        "# Re-render to confirm no blocking issues remain\n",
        "mapper.render_html()\n",
    ],
}

new_code_get_df = {
    "cell_type": "code",
    "execution_count": None,
    "id": "new-get-df-code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Get the remapped DataFrame - succeeds now that no blocking issues remain.\n",
        "# The output includes original_annotation and original_prediction columns\n",
        "# that preserve the raw labels before remapping.\n",
        "mapped_df = mapper.get_mapped_results_dataframe()\n",
        "\n",
        "# Log entity mappings for experiment tracking\n",
        "experiment.log_parameter('entity_mappings', json.dumps(mapper.get_mapping()))\n",
        "\n",
        "mapped_df.head(50)\n",
    ],
}

# Insert new cells after index 17
nb["cells"].insert(18, new_markdown_resolve)
nb["cells"].insert(19, new_code_resolve)
nb["cells"].insert(20, new_code_get_df)

# Remove old cell 18 (mapped_df.head(50)) — now at index 21 after insertions
del nb["cells"][21]

# Remove old cell 19 (experiment.log_parameter) — now at index 21
del nb["cells"][21]

with open("notebooks/4_Evaluate_Presidio_Analyzer.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])[:90].replace("\n", " ")
    print(f"  {i}: [{cell['cell_type']}] {src}")

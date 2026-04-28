"""Rewrite notebook 6 for the new two-phase CanonicalMapper API (US-020)."""

import json

# Load existing notebook — keep setup/data cells, rewrite mapping cells
nb = json.load(open("notebooks/6_Interactive_Entity_Mapping.ipynb"))


def code_cell(source_lines, cell_id=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def md_cell(source_lines, cell_id=None):
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


# ── Cell 0: intro markdown ───────────────────────────────────────────────────
nb["cells"][0]["source"] = [
    "## Interactive Entity Mapping for PII Evaluation\n",
    "\n",
    "This notebook is a comprehensive tutorial on `CanonicalMapper` — the two-phase\n",
    "entity alignment tool for PII evaluation.\n",
    "\n",
    "**Topics covered:**\n",
    "1. Why entity mapping is needed\n",
    "2. Two-phase workflow: Identify (5 tiers) → Project (majority-vote depth)\n",
    "3. All six issue types with concrete examples\n",
    "4. Majority-vote depth auto-discovery\n",
    "5. Projection rules: exact, trivial, ambiguous, cross-branch, unresolved\n",
    "6. Programmatic resolution with `map()`\n",
    "7. Interactive resolution with `resolve_interactively()`\n",
    "8. Eval-surface locking for multi-model comparison\n",
    "9. EntityHierarchy customisation\n",
    "10. Getting the final mapped DataFrame\n",
]

# ── Cell 1: imports ──────────────────────────────────────────────────────────
nb["cells"][1]["source"] = [
    "from collections import Counter\n",
    "from pathlib import Path\n",
    "from pprint import pprint\n",
    "\n",
    "from presidio_analyzer import AnalyzerEngine, PatternRecognizer\n",
    "\n",
    "from presidio_evaluator import InputSample\n",
    "from presidio_evaluator.entity_mapping import (\n",
    "    CanonicalMapper,\n",
    "    EntityHierarchy,\n",
    "    IncompleteMapping,\n",
    "    IssueType,\n",
    "    IssueSeverity,\n",
    ")\n",
    "from presidio_evaluator.evaluation import SpanEvaluator\n",
    "from presidio_evaluator.models import PresidioAnalyzerWrapper\n",
]
nb["cells"][1]["outputs"] = []
nb["cells"][1]["execution_count"] = None

# ── Cell 2: why mapping ──────────────────────────────────────────────────────
nb["cells"][2]["source"] = [
    "## 1. Why Entity Mapping is Needed\n",
    "\n",
    "When evaluating a PII detection model, you are comparing:\n",
    "- **Dataset entities**: ground truth labels (e.g., `STREET_ADDRESS`, `FIRST_NAME`, `GPE`)\n",
    "- **Model entities**: what the model can detect (e.g., `LOCATION`, `PERSON`, `NRP`)\n",
    "\n",
    "These often use **different naming conventions and hierarchy depths**:\n",
    "\n",
    "| Dataset Entity | Model Entity | Notes |\n",
    "|----------------|--------------|-------|\n",
    "| `STREET_ADDRESS` | `LOCATION` | `STREET_ADDRESS` is depth-3 (`PII→LOCATION→ADDRESS`); Presidio predicts depth-2 `LOCATION` |\n",
    "| `FIRST_NAME` | `PERSON` | `FIRST_NAME` is depth-4; Presidio predicts depth-2 `PERSON` |\n",
    "| `GPE` | `LOCATION` | Both refer to geopolitical entities, different names |\n",
    "\n",
    "`CanonicalMapper` resolves these mismatches automatically in two phases:\n",
    "1. **Identify** — match each raw label to a canonical entity using 5 tiers\n",
    "2. **Project** — project canonical entities onto the eval surface at majority-vote depth\n",
]

# ── Cells 3, 4 (data/model setup) — keep as-is ──────────────────────────────

# ── Cell 5: workflow markdown ────────────────────────────────────────────────
nb["cells"][5]["source"] = [
    "## 2. Two-Phase Workflow: Identify → Project\n",
    "\n",
    "### Phase 1: Identification (5 tiers, in order)\n",
    "\n",
    "| Tier | Example | Notes |\n",
    "|------|---------|-------|\n",
    "| `EXACT` | `EMAIL_ADDRESS` → `EMAIL_ADDRESS` | Label is already a canonical name or known alias |\n",
    "| `COUNTRY` | `US_PASSPORT` → `PASSPORT` | Strip country prefix (`US_`, `AU_`, etc.) |\n",
    "| `COUNTRY_FALLBACK` | `GERMANY_PASSPORT_NUMBER` → `PASSPORT` | Strip country + `_NUMBER` suffix |\n",
    "| `FUZZY` | `EMAILADRES` → `EMAIL_ADDRESS` | Fuzzy string match above threshold (default 0.80) |\n",
    "| `UNRESOLVED` | `XYZZY_UNKNOWN` | No match found — must be manually mapped or suppressed |\n",
    "\n",
    "BIO tags (`B-PERSON`, `I-PERSON`) are automatically stripped before matching.\n",
    "\n",
    "### Phase 2: Projection\n",
    "\n",
    "After identification, each canonical entity is projected onto the **eval surface** —\n",
    "the set of entities at the **majority-vote depth** computed from the dataset annotations.\n",
    "\n",
    "Projection rules:\n",
    "| Rule | Issue Type | Notes |\n",
    "|------|------------|-------|\n",
    "| Canonical is already on the eval surface | `EXACT` projection | No change |\n",
    "| Canonical is a descendant — one sole ancestor on eval surface | `TRIVIAL` (auto-fixed, INFO) | Collapsed upward |\n",
    "| Canonical is a depth-2 ancestor with multiple depth-3 children on eval surface | `COLLISION_AMBIGUOUS` (WARNING) | Must pick one |\n",
    "| Canonical branch has no intersection with eval surface | `COLLISION_CROSS_BRANCH` (WARNING) | Must remap |\n",
    "| Identification failed | `UNRESOLVED` (ERROR) | Must map or suppress |\n",
]

# ── Cell 6: basic analyze ────────────────────────────────────────────────────
nb["cells"][6]["source"] = [
    "# Load and predict on a small subset\n",
    "subset = dataset[:100]\n",
    "wrapped_analyzer = PresidioAnalyzerWrapper(analyzer_engine=analyzer, language='en')\n",
    "results_df = wrapped_analyzer.predict_dataset(subset)\n",
    "\n",
    "# Create mapper — no constructor arguments needed\n",
    "mapper = CanonicalMapper()\n",
    "\n",
    "# analyze() runs both phases: Identify and Project\n",
    "mapper.analyze(results_df)\n",
    "\n",
    "# render_html() shows a color-coded audit table\n",
    "# (falls back to plain text outside Jupyter)\n",
    "mapper.render_html()\n",
]
nb["cells"][6]["outputs"] = []
nb["cells"][6]["execution_count"] = None

# ── Cell 7: issue types markdown ─────────────────────────────────────────────
nb["cells"][7]["source"] = [
    "## 3. All Six Issue Types\n",
    "\n",
    "After `analyze()`, `get_issues()` returns `MappingIssue` objects sorted by severity\n",
    "(ERROR first, then WARNING, then INFO), then by token count (most frequent first).\n",
    "\n",
    "| Issue Type | Severity | Blocking? | When it occurs |\n",
    "|------------|----------|-----------|----------------|\n",
    "| `UNRESOLVED` | ERROR | Yes | Label matched nothing in the hierarchy |\n",
    "| `COLLISION_AMBIGUOUS` | WARNING | Yes | Depth-2 label has multiple depth-3 eval surface matches |\n",
    "| `COLLISION_CROSS_BRANCH` | WARNING | Yes | Label maps to a branch absent from the eval surface |\n",
    "| `PREDICTION_ONLY` | WARNING | Yes | Label appears only in predictions, never in annotations |\n",
    "| `COLLISION_TRIVIAL` | INFO | No | Auto-fixed: label collapsed to its sole ancestor on eval surface |\n",
    "| `DATASET_ONLY` | INFO | No | Label appears only in annotations, never in predictions |\n",
    "\n",
    "**Blocking** means `get_mapped_results_dataframe()` raises `IncompleteMapping` until resolved.\n",
    "INFO issues are non-blocking — they are informational only.\n",
]

# ── Cell 8: view issues ───────────────────────────────────────────────────────
nb["cells"][8]["source"] = [
    "# View all detected issues with full detail\n",
    "print(f'Total issues: {len(mapper.get_issues())}')\n",
    "for issue in mapper.get_issues():\n",
    "    blocking = 'BLOCKING' if issue.severity != IssueSeverity.INFO else 'non-blocking'\n",
    "    print(f'  [{issue.severity.value.upper():7}] {issue.type.value:30} | {blocking} | {issue.labels}')\n",
    "    if issue.annotation_count or issue.prediction_count:\n",
    "        print(f'             annotations={issue.annotation_count}, predictions={issue.prediction_count}')\n",
]
nb["cells"][8]["outputs"] = []
nb["cells"][8]["execution_count"] = None

# ── Cell 9: applying fixes markdown ──────────────────────────────────────────
nb["cells"][9]["source"] = [
    "### Resolving Issues\n",
    "\n",
    "Use `map({label: canonical})` to remap a label to a canonical entity,\n",
    "or `map({label: None})` to suppress it (both annotation and prediction sides become `'O'`).\n",
    "\n",
    "**When to suppress (`None`)**:\n",
    "- The model predicts entity types your dataset does not cover (`PREDICTION_ONLY`)\n",
    "- A label is noise or irrelevant to your evaluation\n",
    "\n",
    "**When to remap**:\n",
    "- `COLLISION_AMBIGUOUS`: pick the depth-3 canonical entity that best matches the predictions\n",
    "- `UNRESOLVED`: manually specify what the label should map to\n",
    "- `COLLISION_CROSS_BRANCH`: remap to a canonical entity in the correct branch\n",
    "\n",
    "`map()` validates all targets atomically — if any target is invalid, none are applied.\n",
    "`map()` returns `self` so you can chain calls.\n",
]

# ── Cell 10: batch resolution ─────────────────────────────────────────────────
nb["cells"][10]["source"] = [
    "# Programmatic batch resolution — iterate over issues by type\n",
    "\n",
    "# PREDICTION_ONLY: suppress labels the model predicts but dataset never annotates\n",
    "prediction_only = [i for i in mapper.get_issues() if i.type == IssueType.PREDICTION_ONLY]\n",
    "for issue in prediction_only:\n",
    "    print(f'Suppressing PREDICTION_ONLY: {issue.labels}')\n",
    "    mapper.map({lbl: None for lbl in issue.labels})\n",
    "\n",
    "# COLLISION_AMBIGUOUS: Presidio depth-2 labels — pick the most appropriate depth-3 target\n",
    "ambiguous = [i for i in mapper.get_issues() if i.type == IssueType.COLLISION_AMBIGUOUS]\n",
    "for issue in ambiguous:\n",
    "    for lbl in issue.labels:\n",
    "        if lbl == 'PERSON':\n",
    "            mapper.map({'PERSON': 'NAME'})\n",
    "        elif lbl == 'DATE_TIME':\n",
    "            mapper.map({'DATE_TIME': 'DATE'})\n",
    "        elif lbl == 'LOCATION':\n",
    "            mapper.map({'LOCATION': 'LOC'})\n",
    "        elif lbl == 'ORGANIZATION':\n",
    "            mapper.map({'ORGANIZATION': 'ORG'})\n",
    "        else:\n",
    "            # Suppress unknown ambiguous labels\n",
    "            mapper.map({lbl: None})\n",
    "\n",
    "# Suppress other known noise\n",
    "mapper.map({'TITLE': None})\n",
    "mapper.map({'COMPOSER': None})\n",
    "\n",
    "# Re-analyze to refresh issue detection\n",
    "mapper.analyze(results_df)\n",
    "mapper.render_html()\n",
]
nb["cells"][10]["outputs"] = []
nb["cells"][10]["execution_count"] = None

# ── Cell 11: pending labels ───────────────────────────────────────────────────
nb["cells"][11]["source"] = [
    "# Check remaining blocking issues\n",
    "blocking = [i for i in mapper.get_issues() if i.severity != IssueSeverity.INFO]\n",
    "if blocking:\n",
    "    print(f'{len(blocking)} blocking issue(s) remain:')\n",
    "    for issue in blocking:\n",
    "        print(f'  [{issue.severity.value}] {issue.type.value}: {issue.labels}')\n",
    "else:\n",
    "    print('No blocking issues - ready to call get_mapped_results_dataframe()')\n",
    "\n",
    "# DATASET_ONLY issues are INFO (non-blocking) - they tell you what the model never predicts\n",
    "dataset_only = [i for i in mapper.get_issues() if i.type == IssueType.DATASET_ONLY]\n",
    "if dataset_only:\n",
    "    all_ds_only = [lbl for i in dataset_only for lbl in i.labels]\n",
    "    print(f'\\nDataset-only entities (INFO, non-blocking): {all_ds_only}')\n",
    "    print('  These entities never appear in model predictions — recall for them will be 0.')\n",
]
nb["cells"][11]["outputs"] = []
nb["cells"][11]["execution_count"] = None

# ── Cell 12: single lookup markdown ──────────────────────────────────────────
nb["cells"][12]["source"] = [
    "## 4. Single Entity Lookup\n",
    "\n",
    "Use `get_mapping(entity='...')` to look up the projected value for any label,\n",
    "including labels not yet analyzed. Raises `ValueError` for unresolvable labels.\n",
]

# ── Cell 13: single entity lookup ────────────────────────────────────────────
nb["cells"][13]["source"] = [
    "# Look up individual entities\n",
    "print('EMAIL_ADDRESS ->', mapper.get_mapping(entity='EMAIL_ADDRESS'))\n",
    "print('B-PERSON ->', mapper.get_mapping(entity='B-PERSON'))  # BIO prefix stripped\n",
    "print('GERMANY_PASSPORT_NUMBER ->', mapper.get_mapping(entity='GERMANY_PASSPORT_NUMBER'))  # COUNTRY tier\n",
    "print('US_SSN ->', mapper.get_mapping(entity='US_SSN'))\n",
    "\n",
    "# Full mapping dict for all labels seen so far\n",
    "full_mapping = mapper.get_mapping()\n",
    "print('\\nFull mapping:')\n",
    "pprint(full_mapping, compact=True)\n",
]
nb["cells"][13]["outputs"] = []
nb["cells"][13]["execution_count"] = None

# ── Cell 14: majority-vote depth section ──────────────────────────────────────
nb["cells"][14]["source"] = [
    "## 5. Majority-Vote Depth Auto-Discovery\n",
    "\n",
    "The eval surface is computed automatically from the **annotation** labels in the results DataFrame.\n",
    "Each annotation label is resolved to a canonical entity, its depth in the hierarchy is measured\n",
    "(capped at 3), and a weighted majority vote determines the eval depth.\n",
    "\n",
    "| Scenario | Majority depth | Eval surface |\n",
    "|----------|---------------|-------------|\n",
    "| Most annotations are `NAME`, `EMAIL_ADDRESS`, `SSN` (depth 3) | 3 | depth-3 entities |\n",
    "| Most annotations are `PERSON`, `LOCATION` (depth 2) | 2 | depth-2 entities |\n",
    "\n",
    "The eval surface is **locked** after the first `analyze()` call and does not change on subsequent\n",
    "calls. This ensures consistent evaluation across multiple models on the same dataset.\n",
]

# ── Cell 15: demonstrate majority-vote depth ──────────────────────────────────
nb["cells"][15]["source"] = [
    "# Demonstrate eval depth auto-discovery\n",
    "print(f'Eval depth (inferred from dataset): {mapper._eval_depth}')\n",
    "print(f'Eval surface size: {len(mapper.eval_surface)} entities')\n",
    "print(f'Sample eval surface entities: {sorted(mapper.eval_surface)[:10]}')\n",
    "\n",
    "# EntityHierarchy.get_depth() shows where each entity sits\n",
    "h = EntityHierarchy(canonical_depth=10)\n",
    "for entity in ['PERSON', 'NAME', 'FIRST_NAME', 'EMAIL_ADDRESS', 'SSN']:\n",
    "    try:\n",
    "        depth = h.get_depth(entity)\n",
    "        branch = h.canonical_to_branch.get(entity, [])\n",
    "        print(f'  {entity:25} depth={depth}  branch={\" → \".join(branch)}')\n",
    "    except Exception:\n",
    "        print(f'  {entity:25} not in hierarchy')\n",
]
nb["cells"][15]["outputs"] = []
nb["cells"][15]["execution_count"] = None

# ── Cell 16: projection rules demo ────────────────────────────────────────────
nb["cells"][16]["source"] = [
    "# Demonstrate projection rules\n",
    "# After analyze(), inspect _records to see tier and projection_type per label\n",
    'print(f\'{"Label":25} {"Tier":15} {"Canonical":20} {"Projected":20} {"Proj Type":20}\')\n',
    "print('-' * 100)\n",
    "for lbl, rec in sorted(mapper._records.items()):\n",
    "    print(f'{lbl:25} {rec.tier:15} {str(rec.canonical):20} {str(rec.projected):20} {str(rec.projection_type):20}')\n",
]
nb["cells"][16]["outputs"] = []
nb["cells"][16]["execution_count"] = None

# ── Cell 17: interactive resolution ───────────────────────────────────────────
nb["cells"][17]["source"] = [
    "# Adjusting the fuzzy threshold: lower = more aggressive auto-resolution\n",
    "loose_mapper = CanonicalMapper(fuzzy_threshold=0.65)\n",
    "loose_mapper.analyze(results_df)\n",
    "print(f'Pending with threshold=0.65: {loose_mapper.pending}')\n",
    "loose_mapper.render_html()\n",
]
nb["cells"][17]["outputs"] = []
nb["cells"][17]["execution_count"] = None

# ── Cell 18: final mapping / eval markdown ────────────────────────────────────
nb["cells"][18]["source"] = [
    "## 6. Interactive Resolution with resolve_interactively()\n",
    "\n",
    "`resolve_interactively()` prompts you for each WARNING+ issue one at a time.\n",
    "It accepts a `prompt_fn` argument for testing (default: `input`).\n",
    "The interactive session is most useful in a terminal or a Jupyter cell — it\n",
    "walks you through every WARNING/ERROR issue and lets you type a canonical name\n",
    "or `None` to suppress.\n",
    "\n",
    "```python\n",
    "# Interactive resolution in a terminal or Jupyter cell\n",
    "# (commented out here to avoid blocking automated notebook runs)\n",
    "# mapper.resolve_interactively()\n",
    "```\n",
    "\n",
    "For automated / batch resolution, use `map()` directly as shown in section 3.\n",
]

# ── Cell 19: eval surface locking ─────────────────────────────────────────────
nb["cells"][19]["source"] = [
    "# Get the full mapping dict once no blocking issues remain\n",
    "if mapper.pending:\n",
    "    print(f'Still {len(mapper.pending)} pending: {mapper.pending}')\n",
    "else:\n",
    "    entity_mapping = mapper.get_mapping()\n",
    "    print('Entity mapping ready:')\n",
    "    pprint(entity_mapping, compact=True)\n",
    "\n",
    "# Get the remapped DataFrame (includes original_annotation / original_prediction columns)\n",
    "try:\n",
    "    mapped_df = mapper.get_mapped_results_dataframe()\n",
    "    print(f'\\nMapped DataFrame: {mapped_df.shape[0]} rows, {mapped_df.shape[1]} columns')\n",
    "    print('Columns:', list(mapped_df.columns))\n",
    "except IncompleteMapping as e:\n",
    "    print(f'Blocked: {e}')\n",
]
nb["cells"][19]["outputs"] = []
nb["cells"][19]["execution_count"] = None

# ── Cell 20: multi-model eval surface locking ─────────────────────────────────
nb["cells"][20]["source"] = [
    "## 7. Eval Surface Locking for Multi-Model Comparison\n",
    "\n",
    "The eval surface is locked after the first `analyze()` call. When you call `analyze()`\n",
    "again (e.g., with a second model's results), the eval surface stays the same.\n",
    "This guarantees that both models are evaluated on identical entity types.\n",
]

# ── Cell 21: multi-model demo ─────────────────────────────────────────────────
nb["cells"][21]["source"] = [
    "# Demonstrate eval-surface locking across two models\n",
    "comparison_mapper = CanonicalMapper()\n",
    "\n",
    "# Model 1\n",
    "comparison_mapper.analyze(results_df)\n",
    "surface_after_model1 = frozenset(comparison_mapper.eval_surface)\n",
    "print(f'Eval surface after model 1: {len(surface_after_model1)} entities')\n",
    "\n",
    "# Model 2 (using same results_df for illustration; in practice this would be a different model)\n",
    "comparison_mapper.analyze(results_df)\n",
    "surface_after_model2 = frozenset(comparison_mapper.eval_surface)\n",
    "print(f'Eval surface after model 2: {len(surface_after_model2)} entities')\n",
    "\n",
    "assert surface_after_model1 == surface_after_model2, 'Eval surface changed - this should not happen!'\n",
    "print('Eval surface is identical across both models. Multi-model comparison is consistent.')\n",
]
nb["cells"][21]["outputs"] = []
nb["cells"][21]["execution_count"] = None

# ── Cell 22: custom hierarchy ──────────────────────────────────────────────────
nb["cells"][22]["source"] = [
    "## 8. EntityHierarchy Customisation\n",
    "\n",
    "Pass a custom hierarchy dict or `EntityHierarchy` instance to `CanonicalMapper(hierarchy=...)` to:\n",
    "- Add aliases for labels not in the default taxonomy\n",
    "- Extend the hierarchy with new entity types\n",
    "- Override the default mappings\n",
]

# ── Cell 23: summary ──────────────────────────────────────────────────────────
nb["cells"][23]["source"] = [
    "## Summary\n",
    "\n",
    "### Key Takeaways\n",
    "\n",
    "1. **Two-phase workflow**: `analyze()` runs Identify (5 tiers) then Project (majority-vote depth)\n",
    "\n",
    "2. **Eval depth is data-driven**: computed from annotation label depths; locked after first `analyze()`\n",
    "\n",
    "3. **Six issue types**:\n",
    "   - ERROR (blocking): `UNRESOLVED`\n",
    "   - WARNING (blocking): `COLLISION_AMBIGUOUS`, `COLLISION_CROSS_BRANCH`, `PREDICTION_ONLY`\n",
    "   - INFO (non-blocking): `COLLISION_TRIVIAL`, `DATASET_ONLY`\n",
    "\n",
    "4. **Resolution options**:\n",
    "   - `map({label: canonical})` — remap\n",
    "   - `map({label: None})` — suppress\n",
    "   - `resolve_interactively()` — guided interactive session\n",
    "\n",
    "5. **Eval surface locking** ensures consistent multi-model comparison\n",
    "\n",
    "6. **`get_mapped_results_dataframe()`** adds `original_annotation` and `original_prediction` columns\n",
    "   preserving raw labels before remapping\n",
    "\n",
    "### Best Practices\n",
    "\n",
    "1. Call `render_html()` after `analyze()` to visually audit all mappings\n",
    "2. Pre-map known overrides *before* `analyze()` to keep the issue list focused\n",
    "3. INFO issues (`DATASET_ONLY`, `COLLISION_TRIVIAL`) are non-blocking — review but no action required\n",
    "4. Use the same mapper instance across multiple models to ensure eval surface consistency\n",
]

# ── Cell 24: quick reference ──────────────────────────────────────────────────
nb["cells"][24]["source"] = [
    "print('''\n",
    "=== CanonicalMapper Quick Reference ===\n",
    "\n",
    "# 1. Create (no constructor args needed)\n",
    "mapper = CanonicalMapper()\n",
    "\n",
    "# 2. Pre-map known labels before analyze() (optional)\n",
    "mapper.map({'MY_LABEL': 'PERSON', 'NOISE': None})\n",
    "\n",
    "# 3. Analyze — Phase 1 (Identify) + Phase 2 (Project)\n",
    "mapper.analyze(results_df)   # eval depth inferred from annotations\n",
    "\n",
    "# 4. Review\n",
    "mapper.render_html()          # HTML audit table (Jupyter)\n",
    "mapper.get_issues()           # structured MappingIssue list\n",
    "mapper.pending                # UNRESOLVED-tier labels\n",
    "\n",
    "# 5. Resolve blocking issues (WARNING+)\n",
    "mapper.map({'PERSON': 'NAME'})             # remap COLLISION_AMBIGUOUS\n",
    "mapper.map({'PREDICTION_ONLY_LBL': None})  # suppress PREDICTION_ONLY\n",
    "# mapper.resolve_interactively()           # guided interactive session\n",
    "\n",
    "# 6. Single entity lookup\n",
    "mapper.get_mapping(entity='B-PERSON')   # strips BIO, returns projected value\n",
    "\n",
    "# 7. Get results\n",
    "entity_mapping = mapper.get_mapping()         # full {label: projected} dict\n",
    "mapped_df = mapper.get_mapped_results_dataframe()  # raises IncompleteMapping if blocking\n",
    "\n",
    "# 8. Evaluate\n",
    "evaluator = SpanEvaluator(iou_threshold=0.75)\n",
    "scores = evaluator.calculate_score_on_df(results_df=mapped_df)\n",
    "''')\n",
]
nb["cells"][24]["outputs"] = []
nb["cells"][24]["execution_count"] = None

with open("notebooks/6_Interactive_Entity_Mapping.ipynb", "w") as f:
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
    "hierarchy=N",
]
print("Checking for old API references:")
for term in old_terms:
    if term in src_all:
        print(f"  FOUND: {term!r}")
    else:
        print(f"  OK (not found): {term!r}")

print(f"\nDone. Total cells: {len(nb['cells'])}")

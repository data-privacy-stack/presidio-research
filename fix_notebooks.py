import json

REPLACEMENTS = [
    (
        "    model=wrapped_analyzer,\n",
        "    model=None,\n",
    ),
    (
        "# Create the evaluator object (no entity_mapping \u2014 mapping is handled by CanonicalMapper)\n",
        "# Create the evaluator object (pure scoring engine \u2014 no model)\n",
    ),
    (
        "# Create the evaluator (no entity_mapping \u2014 handled by CanonicalMapper)\n",
        "# Create the evaluator (pure scoring engine \u2014 no model)\n",
    ),
    (
        'params = {"dataset_name": dataset_name, "model_name": evaluator.model.name}\n',
        'params = {"dataset_name": dataset_name, "model_name": wrapped_analyzer.name}\n',
    ),
    (
        "params.update(evaluator.model.to_log())\n",
        "params.update(wrapped_analyzer.to_log())\n",
    ),
    (
        "model_name=evaluator.model.name,",
        "model_name=wrapped_analyzer.name,",
    ),
]

for nb_path in [
    "notebooks/4_Evaluate_Presidio_Analyzer.ipynb",
    "notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb",
]:
    with open(nb_path) as f:
        nb = json.load(f)

    changed = 0
    for cell in nb["cells"]:
        src = cell.get("source", [])
        new_src = []
        for line in src:
            new_line = line
            for old, new in REPLACEMENTS:
                if old in new_line:
                    new_line = new_line.replace(old, new)
                    changed += 1
            new_src.append(new_line)
        cell["source"] = new_src

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Fixed {nb_path}: {changed} replacements")

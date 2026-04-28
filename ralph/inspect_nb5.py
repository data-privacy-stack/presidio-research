"""Update notebook 5 for the new two-phase CanonicalMapper API (US-019)."""

import json

nb = json.load(open("notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb"))

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if any(
        t in src
        for t in [
            "CanonicalMapper",
            "mapper",
            "canonical",
            "entity_mapping",
            "hierarchy=",
            "get_issues",
            "IncompleteMapping",
        ]
    ):
        print(f"--- Cell {i} ({cell['cell_type']}) ---")
        print(src[:300])
        print()

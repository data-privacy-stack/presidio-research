"""Clean up duplicate cells in notebook 4 after US-018 edits."""

import json

nb = json.load(open("notebooks/4_Evaluate_Presidio_Analyzer.ipynb"))

# Cells 20, 21, 22 all look similar — keep only cell 20 (the clean version)
# Cell 20: "# Get the remapped DataFrame - succeeds now..." (the one we want)
# Cell 21: duplicate of 20 — DELETE
# Cell 22: old edit_notebook_file version — DELETE
# Cell 25: standalone "mapped_df" line — DELETE (it's in cell 20 now)

# Delete in reverse order to preserve indices
indices_to_delete = sorted([21, 22, 25], reverse=True)
for idx in indices_to_delete:
    src = "".join(nb["cells"][idx]["source"])[:60]
    print(f"Deleting cell {idx}: {src!r}")
    del nb["cells"][idx]

with open("notebooks/4_Evaluate_Presidio_Analyzer.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nDone. Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])[:90].replace("\n", " ")
    print(f"  {i}: [{cell['cell_type']}] {src}")

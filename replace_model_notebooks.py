"""Script to replace model notebooks with deprecation notice."""

import json

deprecation_cell = {
    "cell_type": "markdown",
    "id": "deprecated-notice",
    "metadata": {},
    "source": [
        "This notebook is no longer supported. Add models directly through Presidio to evaluate them."
    ],
}

notebooks = [
    "notebooks/models/Create datasets for Spacy training.ipynb",
    "notebooks/models/Evaluate azure text analytics.ipynb",
    "notebooks/models/Evaluate flair models.ipynb",
    "notebooks/models/Evaluate spacy models.ipynb",
    "notebooks/models/Evaluate stanza models.ipynb",
]

for nb_path in notebooks:
    with open(nb_path) as f:
        nb = json.load(f)

    # Replace all cells with a single deprecation markdown cell
    nb["cells"] = [dict(deprecation_cell)]
    # Clear widget state if present
    if "metadata" in nb:
        nb["metadata"].pop("widgets", None)

    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated: {nb_path}")

print("Done - all model notebooks replaced with deprecation notice")

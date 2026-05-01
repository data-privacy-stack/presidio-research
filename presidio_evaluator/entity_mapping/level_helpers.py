"""Pure label-projection helpers shared by the mapper and the evaluator.

These functions are kept in a dedicated module to avoid circular imports
between mapper.py and base_evaluator.py.
"""

from __future__ import annotations

from presidio_evaluator.entity_mapping.hierarchy import EntityHierarchy


def to_binary(label: str | None) -> str:
    """Project *label* to binary PII / O.

    ``None`` and ``"O"`` → ``"O"``; anything else → ``"PII"``.
    """
    if label is None or label == "O":
        return "O"
    return "PII"


def to_branch(label: str | None, hierarchy: EntityHierarchy) -> str:
    """Project *label* to its depth-2 branch ancestor.

    ``None`` and ``"O"`` → ``"O"``.  Any non-O label that can be found in
    the hierarchy is mapped to path element at index 1 (the depth-2 node,
    e.g. ``PERSON``, ``LOCATION``, …).  If the label is itself a depth-2
    node it is returned unchanged.  Labels not found in the hierarchy are
    returned as-is (they should not reach this function in normal usage).
    """
    if label is None or label == "O":
        return "O"
    branch_path = hierarchy.canonical_to_branch.get(label)
    if branch_path is None or len(branch_path) < 2:
        # depth-1 node (PII itself) or unknown — return as-is
        return label
    return branch_path[1]

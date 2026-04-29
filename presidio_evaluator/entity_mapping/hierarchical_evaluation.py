"""Hierarchical multi-level evaluation wrapper.

Scores model performance at three entity granularity levels simultaneously:

- **L0** — PII vs. non-PII (binary detection)
- **L1** — Branch-level (PERSON, LOCATION, …)
- **L2** — Canonical surface (NAME, STREET_ADDRESS, …)

See: docs/adr/ADR-003-hierarchical-evaluation.md
"""

from __future__ import annotations

import pandas as pd

from presidio_evaluator.entity_mapping.hierarchy import EntityHierarchy
from presidio_evaluator.evaluation import EvaluationResult, SpanEvaluator


def _to_l1(entity: str, hierarchy: EntityHierarchy) -> str:
    """Map an entity label to its depth-2 (branch) ancestor.

    If the entity is already depth-2 (e.g. PERSON) it is returned unchanged.
    Depth-1 (PII itself) is also returned unchanged.
    'O' is returned as 'O'.
    """
    if entity == "O":
        return "O"
    branch = hierarchy.canonical_to_branch.get(entity)
    if branch is None:
        return entity  # unknown entity — pass through unchanged
    if len(branch) >= 2:
        return branch[1]  # e.g. ['PII', 'PERSON', 'NAME'] -> 'PERSON'
    return branch[0]  # depth-1 node (PII itself)


def _to_l0(entity: str) -> str:
    """Map any non-O entity label to 'PII'."""
    return "O" if entity == "O" else "PII"


def calculate_hierarchical_scores(
    mapped_df: pd.DataFrame,
    hierarchy: EntityHierarchy | None = None,
    beta: float = 2.0,
) -> dict[str, EvaluationResult]:
    """Evaluate model performance at three granularity levels simultaneously.

    Accepts the output of :meth:`CanonicalMapper.get_mapped_results_dataframe`
    and produces scores at:

    - **L0**: PII vs. O — privacy-critical binary detection signal.
    - **L1**: Branch-level (PERSON, LOCATION, …) — category accuracy.
    - **L2**: Canonical surface (NAME, STREET_ADDRESS, …) — granularity accuracy.

    For each level the annotation and prediction columns are projected to the
    appropriate depth, then :class:`SpanEvaluator` computes standard
    precision / recall / F-beta.  No changes to :class:`SpanEvaluator` are
    required — the level-specific columns are renamed to ``annotation`` /
    ``prediction`` before scoring.

    :param mapped_df: DataFrame as returned by
        ``CanonicalMapper.get_mapped_results_dataframe()``.  Must contain
        ``annotation`` and ``prediction`` columns (plus ``sentence_id``,
        ``token``, and ``start_indices``).
    :param hierarchy: :class:`EntityHierarchy` instance used for branch
        lookups.  Defaults to the standard :func:`EntityHierarchy` built from
        :data:`HIERARCHY`.
    :param beta: F-beta parameter passed to :class:`SpanEvaluator`
        (default ``2.0``).
    :return: ``dict[str, EvaluationResult]`` with keys ``"L0"``, ``"L1"``,
        ``"L2"``.  Each value is an :class:`EvaluationResult` containing
        per-type and global PII metrics for that level.

    See Also
    --------
    docs/adr/ADR-003-hierarchical-evaluation.md
    """
    if hierarchy is None:
        hierarchy = EntityHierarchy()

    evaluator = SpanEvaluator()
    results: dict[str, EvaluationResult] = {}

    # --- L2: canonical surface — use mapped_df as-is ---
    results["L2"] = evaluator.calculate_score_on_df(mapped_df, beta=beta)

    # --- L1: branch-level (depth-2 ancestors) ---
    l1_df = mapped_df.copy()
    l1_df["annotation"] = l1_df["annotation"].apply(lambda e: _to_l1(e, hierarchy))
    l1_df["prediction"] = l1_df["prediction"].apply(lambda e: _to_l1(e, hierarchy))
    results["L1"] = evaluator.calculate_score_on_df(l1_df, beta=beta)

    # --- L0: PII vs. O (binary) ---
    l0_df = mapped_df.copy()
    l0_df["annotation"] = l0_df["annotation"].apply(_to_l0)
    l0_df["prediction"] = l0_df["prediction"].apply(_to_l0)
    results["L0"] = evaluator.calculate_score_on_df(l0_df, beta=beta)

    return results

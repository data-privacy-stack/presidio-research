# ---------------------------------------------------------------------------
# Issue types and structured issues
# ---------------------------------------------------------------------------


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


class IssueType(Enum):
    """Category of mapping issue detected during analysis.

    UNRESOLVED:             Label could not be matched to any canonical entity. Blocks extraction.
    COLLISION_CROSS_BRANCH: Label is on a different hierarchy branch from all annotation entities.
    PREDICTION_ONLY:        Label appears in predictions but not in dataset annotations.
    DATASET_ONLY:           Entity has annotations but no prediction maps to it.
    COLLISION_SAME_BRANCH:  Label and co-occurring annotation share the same hierarchy branch
                            at different depths (e.g. model predicts PERSON, dataset uses NAME).
                            Informational — handled automatically by hierarchical evaluation.
    """

    UNRESOLVED = "unresolved"
    COLLISION_CROSS_BRANCH = "collision_cross_branch"
    PREDICTION_ONLY = "prediction_only"
    DATASET_ONLY = "dataset_only"
    COLLISION_SAME_BRANCH = "collision_same_branch"


class IssueSeverity(Enum):
    """How serious the mapping issue is."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ResolutionOption:
    """A suggested fix for a mapping issue.

    Call ``mapper.map(option.mapper_call)`` to apply the fix.
    An empty ``mapper_call`` means re-running ``analyze()`` with different
    parameters (described in ``description``).
    """

    action: str
    description: str
    mapper_call: dict[str, str | None]
    auto_applicable: bool = False


@dataclass
class MappingIssue:
    """A structured mapping issue detected during analysis."""

    type: IssueType
    severity: IssueSeverity
    message: str
    labels: list[str]
    annotation_count: int | None = None
    prediction_count: int | None = None
    resolution_options: list[ResolutionOption] = field(default_factory=list)
    overlap_counts: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# MappedResults — output of CanonicalMapper.get_mapped_results_dataframe()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MappedResults:
    """Four DataFrames produced by the single-phase CanonicalMapper.

    Each DataFrame has ``annotation`` and ``prediction`` columns
    (plus all original non-label columns such as ``sentence_id``,
    ``token``, ``start_indices``) projected to the appropriate level.

    Attributes:
        original:  Raw input labels, unmodified.
        binary:    Labels resolved to ``"PII"`` (any non-O) or ``"O"``.
        branch:    Labels resolved to the depth-2 branch ancestor
                   (e.g. ``FIRST_NAME`` → ``PERSON``).
        detailed:  Labels resolved to the hierarchy node at native depth
                   (e.g. ``FIRST_NAME`` → ``NAME``).  Suppressed → ``"O"``.
    """

    original: pd.DataFrame
    binary: pd.DataFrame
    branch: pd.DataFrame
    detailed: pd.DataFrame

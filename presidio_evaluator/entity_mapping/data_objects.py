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
                            Each prediction is projected to the deepest annotated
                            ancestor of its own label, so this never blocks — including
                            when the annotations themselves mix depths on one branch.
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


#: Columns carrying the entity instance id of the span covering each token
#: (None for ``O`` tokens), one for the gold side and one for the prediction
#: side. ``BaseModel.predict_dataset`` produces both from the source spans, and
#: ``CanonicalMapper`` passes them through every level untouched, so span
#: evaluation reconstructs the same spans at every level: two adjacent
#: entities carry different ids even when their labels have been collapsed to
#: ``"PII"``, and even when they are of the same type. Ids are sequential per
#: sentence and independent between the two columns.
ANNOTATION_SPAN_ID = "annotation_span_id"
PREDICTION_SPAN_ID = "prediction_span_id"


@dataclass(frozen=True)
class MappedResults:
    """Four DataFrames produced by the single-phase CanonicalMapper.

    Each DataFrame has ``annotation`` and ``prediction`` columns
    (plus all original non-label columns such as ``sentence_id``,
    ``token``, ``start_indices``, ``annotation_span_id`` and
    ``prediction_span_id``) projected to the appropriate level. Only the two
    label columns differ between levels.

    Attributes:
        original:  Raw input columns and values.
        binary:    Labels resolved to ``"PII"`` (any non-O) or ``"O"``.
        branch:    Labels resolved to the depth-2 branch ancestor
                   (e.g. ``FIRST_NAME`` → ``PERSON``).
        detailed:  Labels resolved to the hierarchy node at native depth, with
                   each prediction projected upward to the deepest annotated
                   ancestor of its own label. Suppressed → ``"O"``.
    """

    original: pd.DataFrame
    binary: pd.DataFrame
    branch: pd.DataFrame
    detailed: pd.DataFrame

    def get_level(self, level: str) -> pd.DataFrame:
        """Get the DataFrame for the specified mapping level."""
        if level == "original":
            return self.original
        elif level == "binary":
            return self.binary
        elif level == "branch":
            return self.branch
        elif level == "detailed":
            return self.detailed
        else:
            raise ValueError(f"Invalid level: {level}")

    def get_prediction_entities_list(self, level: str) -> list[str]:
        """Get the list of unique entities present in the specified level."""
        df = self.get_level(level)
        entities = set(df["prediction"].unique())
        return sorted(e for e in entities if e != "O")

    def get_annotation_entities_list(self, level: str) -> list[str]:
        """Get the list of unique entities present in the specified level."""
        df = self.get_level(level)
        entities = set(df["annotation"].unique())
        return sorted(e for e in entities if e != "O")

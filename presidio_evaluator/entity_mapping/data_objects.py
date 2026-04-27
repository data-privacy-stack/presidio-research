# ---------------------------------------------------------------------------
# Issue types and structured issues
# ---------------------------------------------------------------------------


from dataclasses import dataclass, field
from enum import Enum


class IssueType(Enum):
    """Category of mapping issue detected during analysis."""

    UNRESOLVED = "unresolved"
    COLLISION_TRIVIAL = "collision_trivial"
    COLLISION_AMBIGUOUS = "collision_ambiguous"
    COLLISION_CROSS_BRANCH = "collision_cross_branch"
    PREDICTION_ONLY = "prediction_only"
    DATASET_ONLY = "dataset_only"


class IssueSeverity(Enum):
    """How serious the mapping issue is."""

    # TODO: add :params: to docstring

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

    # TODO: add :params: to docstring

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

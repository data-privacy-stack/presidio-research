"""Entity mapping: resolve raw entity labels to canonical entities."""

from presidio_evaluator.entity_mapping.data_objects import (
    IssueSeverity,
    IssueType,
    MappingIssue,
    ResolutionOption,
)
from presidio_evaluator.entity_mapping.definitions import (
    COUNTRIES,
    HIERARCHY,
    EntityNotMappedError,
)
from presidio_evaluator.entity_mapping.hierarchy import EntityHierarchy
from presidio_evaluator.entity_mapping.mapper import (
    CanonicalMapper,
    IncompleteMapping,
)

__all__ = [
    # Mapper
    "CanonicalMapper",
    "IncompleteMapping",
    # Issues
    "IssueType",
    "IssueSeverity",
    "MappingIssue",
    "ResolutionOption",
    # Entity hierarchy
    "COUNTRIES",
    "HIERARCHY",
    "EntityHierarchy",
    "EntityNotMappedError",
]

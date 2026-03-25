"""Entity mapping: resolve raw entity labels to canonical entities."""

from presidio_evaluator.entity_mapping.definitions import (
    COUNTRIES,
    HIERARCHY,
    EntityNotMappedError,
)
from presidio_evaluator.entity_mapping.mapper import (
    CanonicalMapper,
    EntityHierarchy,
    IncompleteMapping,
)

__all__ = [
    # Mapper
    "CanonicalMapper",
    "IncompleteMapping",
    # Entity hierarchy
    "COUNTRIES",
    "HIERARCHY",
    "EntityHierarchy",
    "EntityNotMappedError",
]

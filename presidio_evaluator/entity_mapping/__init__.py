"""Entity mapping: resolve raw entity labels to canonical entities."""

from presidio_evaluator.entity_mapping.definitions import (
    COUNTRIES,
    HIERARCHY,
    EntityNotMappedError,
)
from presidio_evaluator.entity_mapping.mapper import (
    ALL_CANONICAL_ENTITIES,
    CANONICAL_TO_BRANCH,
    RAW_TO_CANONICAL,
    CanonicalMapper,
    EntityHierarchy,
    IncompleteMapping,
    canonicalize,
    get_branch,
    print_hierarchy,
)

__all__ = [
    # Mapper
    "CanonicalMapper",
    "IncompleteMapping",
    # Entity hierarchy
    "ALL_CANONICAL_ENTITIES",
    "CANONICAL_TO_BRANCH",
    "COUNTRIES",
    "HIERARCHY",
    "RAW_TO_CANONICAL",
    "EntityHierarchy",
    "EntityNotMappedError",
    "canonicalize",
    "get_branch",
    "print_hierarchy",
]

"""Entity mapping: resolve raw entity labels to canonical EntityHierarchy entities."""

from presidio_evaluator.entity_mapping.mapper import (
    CanonicalMapper,
    EntityMapper,
    IncompleteMapping,
)

from presidio_evaluator.entity_mapping.hierarchy import (
    ALL_CANONICAL_ENTITIES,
    CANONICAL_TO_BRANCH,
    COUNTRIES,
    HIERARCHY,
    RAW_TO_CANONICAL,
    EntityHierarchy,
    EntityNotMappedError,
    canonicalize,
    get_branch,
    print_hierarchy,
)

__all__ = [
    # Mapper
    "CanonicalMapper",
    "EntityMapper",
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

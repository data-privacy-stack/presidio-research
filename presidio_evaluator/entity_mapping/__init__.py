"""Entity mapping utilities for aligning dataset entities to model entities.

This package provides tools for:
- Semantic similarity-based entity mapping
- Rule-based entity mapping (exact matches, patterns, substrings)
- Hybrid mapping strategies combining multiple approaches
- Interactive entity mapping sessions for experiment setup
- Automatic entity detection from models and datasets
"""

# Import from mapper module
from presidio_evaluator.entity_mapping.mapper import (
    EntityMapper,
    SemanticEntityMapper,
    create_hierarchical_mapper,
    create_presidio_mapper,
)

# Import from interactive module
from presidio_evaluator.entity_mapping.interactive import (
    EntityMappingHelper,
    get_model_entities,
    get_dataset_entities,
    suggest_mapping,
)

__all__ = [
    # Mapper classes
    "EntityMapper",
    "SemanticEntityMapper",
    # Mapper factory functions
    "create_hierarchical_mapper",
    "create_presidio_mapper",
    # Interactive mapping
    "EntityMappingHelper",
    "get_model_entities",
    "get_dataset_entities",
    "suggest_mapping",
]

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
    DictEntityMapper,
    SemanticEntityMapper,
    HybridEntityMapper,
    create_hierarchical_mapper,
    create_presidio_mapper,
)

# Import from interactive module
from presidio_evaluator.entity_mapping.interactive import (
    EntityMappingHelper,
    get_model_entities,
    get_dataset_entities,
    suggest_mapping,
    print_mapping_summary,
    save_mapping_to_file,
    load_mapping_from_file,
)

__all__ = [
    # Mapper classes
    "EntityMapper",
    "DictEntityMapper",
    "SemanticEntityMapper",
    "HybridEntityMapper",
    # Mapper factory functions
    "create_hierarchical_mapper",
    "create_presidio_mapper",
    # Interactive mapping
    "EntityMappingHelper",
    "get_model_entities",
    "get_dataset_entities",
    "suggest_mapping",
    "print_mapping_summary",
    "save_mapping_to_file",
    "load_mapping_from_file",
]

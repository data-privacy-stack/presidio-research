"""
Utility functions for two-tiered evaluation.

Provides functions to filter datasets by entity support and
calculate both coverage and quality scores.
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from presidio_evaluator import InputSample


def load_entity_support_mapping() -> Dict[str, Any]:
    """Load entity support mapping from JSON file."""
    mapping_file = Path(__file__).parent.parent.parent / "data" / "entity_support_mapping.json"
    with open(mapping_file) as f:
        return json.load(f)


def get_supported_entities(model_name: str) -> Set[str]:
    """
    Get the set of dataset entities that a model can detect.

    Args:
        model_name: Name of the model (e.g., "BERT-base-NER")

    Returns:
        Set of supported entity types
    """
    mapping = load_entity_support_mapping()
    if model_name not in mapping["models"]:
        raise ValueError(f"Model {model_name} not found in entity support mapping")

    return set(mapping["models"][model_name]["supported_dataset_entities"])


def filter_samples_by_entities(
    samples: List[InputSample],
    supported_entities: Set[str]
) -> List[InputSample]:
    """
    Filter dataset samples to only include supported entities.

    This removes any annotations for entities that the model cannot detect,
    allowing for a fair "quality" evaluation.

    Args:
        samples: List of InputSample objects with annotations
        supported_entities: Set of entity types to keep

    Returns:
        List of filtered InputSample objects with only supported entities
    """
    filtered_samples = []

    for sample in samples:
        # Filter spans to only include supported entities
        filtered_spans = [
            span for span in sample.spans
            if span.entity_type in supported_entities
        ]

        # Only include samples that have at least one supported entity
        # (Skip samples with only unsupported entities)
        if filtered_spans:
            filtered_sample = InputSample(
                full_text=sample.full_text,
                masked=sample.masked,
                spans=filtered_spans,
                template_id=sample.template_id if hasattr(sample, 'template_id') else None,
                create_tags_from_span=False,
                tokens=sample.tokens if hasattr(sample, 'tokens') else None,
                tags=None  # Will be recreated from filtered spans
            )
            filtered_samples.append(filtered_sample)

    return filtered_samples


def calculate_entity_counts(samples: List[InputSample]) -> Dict[str, int]:
    """
    Count occurrences of each entity type in samples.

    Args:
        samples: List of InputSample objects

    Returns:
        Dict mapping entity type to count
    """
    counts = {}
    for sample in samples:
        for span in sample.spans:
            entity_type = span.entity_type
            counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts


def get_coverage_metrics(model_name: str) -> Dict[str, Any]:
    """
    Get coverage metrics for a model.

    Args:
        model_name: Name of the model

    Returns:
        Dict with coverage information
    """
    mapping = load_entity_support_mapping()
    model_info = mapping["models"][model_name]

    return {
        "supported_count": model_info["supported_count"],
        "unsupported_count": model_info["unsupported_count"],
        "coverage_percentage": model_info["coverage_percentage"],
        "supported_entities": model_info["supported_dataset_entities"],
        "unsupported_entities": model_info["unsupported_dataset_entities"],
    }


def print_filtering_summary(
    model_name: str,
    original_samples: List[InputSample],
    filtered_samples: List[InputSample],
    supported_entities: Set[str]
):
    """
    Print summary of dataset filtering.

    Args:
        model_name: Name of the model
        original_samples: Original dataset samples
        filtered_samples: Filtered dataset samples
        supported_entities: Set of supported entities
    """
    orig_counts = calculate_entity_counts(original_samples)
    filt_counts = calculate_entity_counts(filtered_samples)

    orig_total = sum(orig_counts.values())
    filt_total = sum(filt_counts.values())

    print("=" * 100)
    print(f"FILTERING DATASET FOR: {model_name}")
    print("=" * 100)
    print(f"Original samples: {len(original_samples)}")
    print(f"Filtered samples: {len(filtered_samples)} ({len(filtered_samples)/len(original_samples)*100:.1f}%)")
    print(f"\nOriginal entities: {orig_total}")
    print(f"Filtered entities: {filt_total} ({filt_total/orig_total*100:.1f}%)")
    print(f"\nSupported entity types ({len(supported_entities)}): {sorted(supported_entities)}")

    # Show removed entities
    removed_entities = set(orig_counts.keys()) - supported_entities
    if removed_entities:
        print(f"\n❌ Removed entity types ({len(removed_entities)}): {sorted(removed_entities)}")
        removed_count = sum(orig_counts[e] for e in removed_entities)
        print(f"   Removed {removed_count} entity instances")

    print("=" * 100)


def main():
    """Test the filtering utilities."""
    print("Testing entity support utilities...\n")

    # Test loading
    mapping = load_entity_support_mapping()
    print(f"✓ Loaded entity support mapping")
    print(f"  Dataset entities: {len(mapping['dataset_entities'])}")
    print(f"  Models: {len(mapping['models'])}\n")

    # Test for each model
    for model_name in mapping["models"].keys():
        supported = get_supported_entities(model_name)
        coverage = get_coverage_metrics(model_name)

        print(f"\n{model_name}:")
        print(f"  Supported: {coverage['supported_count']}/{coverage['supported_count'] + coverage['unsupported_count']} "
              f"({coverage['coverage_percentage']:.1f}%)")
        print(f"  Entities: {sorted(supported)}")


if __name__ == "__main__":
    main()


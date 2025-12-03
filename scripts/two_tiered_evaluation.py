"""Two-Tiered Model Evaluation Script

Evaluates models using two approaches:
1. Coverage Score: Evaluate on full dataset (production reality)
2. Quality Score: Evaluate on filtered dataset (fair comparison of model capabilities)

This script:
- Loads existing Tier 1 (coverage) scores from corrected evaluation
- Runs Tier 2 (quality) evaluation with filtered datasets
- Generates comprehensive comparison report
"""
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation.entity_filtering import (
    get_supported_entities,
    filter_samples_by_entities,
    print_filtering_summary,
    calculate_entity_counts,
    get_coverage_metrics
)


def load_dataset(dataset_path: Path) -> List[InputSample]:
    """Load dataset from JSON file."""
    with open(dataset_path) as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        sample = InputSample.from_json(item)
        samples.append(sample)
    
    return samples


def load_tier1_scores() -> Dict[str, Any]:
    """Load existing Tier 1 (coverage) scores from corrected evaluation."""
    data_dir = Path(__file__).parent.parent / "data"
    results_file = data_dir / "corrected_all_results.json"
    
    with open(results_file) as f:
        results = json.load(f)
    
    tier1_scores = {}
    for model_name, model_results in results.items():
        tier1_scores[model_name] = {
            "f_score": model_results["pii_f"],
            "precision": model_results["pii_precision"],
            "recall": model_results["pii_recall"],
            "per_entity": model_results.get("per_entity", {})
        }
    
    return tier1_scores


def evaluate_on_filtered_dataset(
    model_name: str,
    dataset: List[InputSample]
) -> Dict[str, float]:
    """
    Evaluate a model on a filtered dataset (only supported entities).

    For now, this is a placeholder that simulates evaluation.
    In a real implementation, you would:
    1. Load the model
    2. Run predictions on filtered dataset
    3. Calculate metrics

    Args:
        model_name: Name of the model
        dataset: Filtered dataset with only supported entities

    Returns:
        Dict with f_score, precision, recall
    """
    # TODO: Implement actual model evaluation
    # This is where you would:
    # 1. Load the model
    # 2. Run model.predict(dataset)
    # 3. Run evaluator.evaluate(predictions, dataset)
    # 4. Return metrics
    
    # For now, return placeholder scores
    # These will need to be replaced with actual evaluation
    print(f"  [PLACEHOLDER] Would evaluate {model_name} on {len(dataset)} filtered samples")
    print(f"  [TODO] Implement actual model loading and prediction here")
    
    return {
        "f_score": 0.0,  # Placeholder
        "precision": 0.0,  # Placeholder  
        "recall": 0.0,  # Placeholder
        "note": "PLACEHOLDER - needs actual model evaluation implementation"
    }


def run_two_tiered_evaluation(dataset_path: Path):
    """Run two-tiered evaluation for all models."""

    print("=" * 100)
    print("TWO-TIERED MODEL EVALUATION")
    print("=" * 100)
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    print(f"✓ Loaded {len(dataset)} samples")

    # Count entities in full dataset
    full_entity_counts = calculate_entity_counts(dataset)
    print(f"✓ Dataset contains {sum(full_entity_counts.values())} total entities")
    print(f"  Entity types: {sorted(full_entity_counts.keys())}")
    print()

    # Load Tier 1 (coverage) scores
    print("=" * 100)
    print("TIER 1: COVERAGE SCORES (Full Dataset)")
    print("=" * 100)
    tier1_scores = load_tier1_scores()

    print("\nExisting coverage scores from corrected evaluation:")
    for model_name, scores in tier1_scores.items():
        print(f"  {model_name:20s}: F={scores['f_score']:.4f}, "
              f"P={scores['precision']:.4f}, R={scores['recall']:.4f}")
    print()

    # Run Tier 2 (quality) evaluation
    print("=" * 100)
    print("TIER 2: QUALITY SCORES (Filtered Dataset)")
    print("=" * 100)
    print()

    tier2_scores = {}

    for model_name in tier1_scores.keys():
        print(f"\n{'=' * 100}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 100}")

        # Get supported entities
        supported_entities = get_supported_entities(model_name)
        coverage_metrics = get_coverage_metrics(model_name)

        print(f"\nModel supports {len(supported_entities)}/17 dataset entities "
              f"({coverage_metrics['coverage_percentage']:.1f}%)")
        print(f"Supported: {sorted(supported_entities)}")
        print(f"Unsupported: {sorted(coverage_metrics['unsupported_entities'])}")
        print()

        # Filter dataset
        filtered_dataset = filter_samples_by_entities(dataset, supported_entities)
        print_filtering_summary(model_name, dataset, filtered_dataset, supported_entities)

        # Evaluate on filtered dataset
        print(f"\nRunning quality evaluation...")
        quality_scores = evaluate_on_filtered_dataset(model_name, filtered_dataset)

        tier2_scores[model_name] = {
            **quality_scores,
            "filtered_samples": len(filtered_dataset),
            "supported_entity_count": len(supported_entities),
            "supported_entities": sorted(supported_entities)
        }

        print(f"\n✓ Quality scores: F={quality_scores['f_score']:.4f}, "
              f"P={quality_scores['precision']:.4f}, R={quality_scores['recall']:.4f}")
        print()

    # Combine results
    print("\n" + "=" * 100)
    print("COMBINED TWO-TIERED RESULTS")
    print("=" * 100)

    combined_results = {}
    for model_name in tier1_scores.keys():
        combined_results[model_name] = {
            "tier1_coverage": tier1_scores[model_name],
            "tier2_quality": tier2_scores[model_name],
            "entity_support": get_coverage_metrics(model_name)
        }

    # Save results
    output_file = Path(__file__).parent.parent / "data" / "two_tiered_results.json"
    with open(output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n✓ Saved results to: {output_file}")

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print()
    print("| Model           | Coverage F | Quality F | Entities | Interpretation |")
    print("|-----------------|------------|-----------|----------|----------------|")

    for model_name, results in combined_results.items():
        coverage_f = results["tier1_coverage"]["f_score"]
        quality_f = results["tier2_quality"]["f_score"]
        entity_count = results["entity_support"]["supported_count"]
        coverage_pct = results["entity_support"]["coverage_percentage"]

        print(f"| {model_name:15s} | {coverage_f:10.4f} | {quality_f:9.4f} | "
              f"{entity_count:2d}/17 ({coverage_pct:4.1f}%) | ... |")

    print()
    print("=" * 100)
    print("NEXT STEPS")
    print("=" * 100)
    print()
    print("⚠️  NOTE: Tier 2 (quality) scores are currently PLACEHOLDERS")
    print()
    print("To complete the evaluation, you need to:")
    print("1. Implement actual model loading in evaluate_on_filtered_dataset()")
    print("2. Run model predictions on filtered datasets")
    print("3. Calculate real quality metrics")
    print()
    print("The framework is ready - just needs model integration!")
    print("=" * 100)

    return combined_results


def main():
    """Main entry point."""
    # Use the synthetic dataset
    dataset_path = Path(__file__).parent.parent / "data" / "synth_dataset_v2.json"

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please specify correct dataset path")
        return

    results = run_two_tiered_evaluation(dataset_path)

    # Generate markdown report
    generate_markdown_report(results)


def generate_markdown_report(results: Dict[str, Any]):
    """Generate a markdown report of the two-tiered evaluation."""
    report = []
    report.append("# Two-Tiered Evaluation Report")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append("This report presents a **two-tiered evaluation** of PII NER models:")
    report.append("")
    report.append("1. **Tier 1 (Coverage)**: Evaluates all models on the full dataset")
    report.append("   - Answers: 'Which model works best for MY data?'")
    report.append("   - Includes penalty for unsupported entities")
    report.append("")
    report.append("2. **Tier 2 (Quality)**: Evaluates each model on only its supported entities")
    report.append("   - Answers: 'How good is this model at what it does?'")
    report.append("   - Fair comparison of model capabilities")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## Results")
    report.append("")
    report.append("### Comparison Table")
    report.append("")
    report.append("| Model | Coverage (Tier 1) | Quality (Tier 2) | Entities | Coverage % |")
    report.append("|-------|-------------------|------------------|----------|------------|")

    # Sort by coverage score
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]["tier1_coverage"]["f_score"],
        reverse=True
    )

    for model_name, model_results in sorted_models:
        coverage_f = model_results["tier1_coverage"]["f_score"]
        quality_f = model_results["tier2_quality"]["f_score"]
        entity_count = model_results["entity_support"]["supported_count"]
        coverage_pct = model_results["entity_support"]["coverage_percentage"]

        report.append(f"| {model_name} | {coverage_f:.4f} | {quality_f:.4f} | "
                     f"{entity_count}/17 | {coverage_pct:.1f}% |")

    report.append("")
    report.append("### Entity Support Details")
    report.append("")

    for model_name, model_results in sorted_models:
        report.append(f"#### {model_name}")
        report.append("")
        report.append(f"**Coverage**: {model_results['entity_support']['coverage_percentage']:.1f}% "
                     f"({model_results['entity_support']['supported_count']}/17 entities)")
        report.append("")
        report.append(f"**Supported entities**: {', '.join(model_results['entity_support']['supported_entities'])}")
        report.append("")
        report.append(f"**Unsupported entities**: {', '.join(model_results['entity_support']['unsupported_entities'])}")
        report.append("")
        report.append(f"**Tier 1 (Coverage)**:")
        report.append(f"- F-Score: {model_results['tier1_coverage']['f_score']:.4f}")
        report.append(f"- Precision: {model_results['tier1_coverage']['precision']:.4f}")
        report.append(f"- Recall: {model_results['tier1_coverage']['recall']:.4f}")
        report.append("")
        report.append(f"**Tier 2 (Quality)**:")
        report.append(f"- F-Score: {model_results['tier2_quality']['f_score']:.4f}")
        report.append(f"- Precision: {model_results['tier2_quality']['precision']:.4f}")
        report.append(f"- Recall: {model_results['tier2_quality']['recall']:.4f}")

        if "note" in model_results['tier2_quality']:
            report.append(f"- ⚠️ {model_results['tier2_quality']['note']}")

        report.append("")

    report.append("---")
    report.append("")
    report.append("## Interpretation")
    report.append("")
    report.append("### Model Selection Guide")
    report.append("")
    report.append("Choose a model based on your needs:")
    report.append("")
    report.append("1. **Need comprehensive PII detection** (credit cards, IBANs, IPs, etc.)")
    report.append("   - Choose the model with highest entity coverage")
    report.append("   - Prioritize Tier 1 (Coverage) score")
    report.append("")
    report.append("2. **Need high accuracy on specific entity types**")
    report.append("   - Choose the model with highest Tier 2 (Quality) score")
    report.append("   - Check that it supports your required entities")
    report.append("")
    report.append("3. **Medical/Healthcare domain**")
    report.append("   - Choose medical-focused models (RoBERTa-i2b2, StanfordAIMI)")
    report.append("   - Look for AGE, DATE, ID, PATIENT support")
    report.append("")
    report.append("### Coverage vs Quality Tradeoff")
    report.append("")
    report.append("The two-tiered evaluation reveals the tradeoff between:")
    report.append("- **Specialist models**: High quality, limited coverage")
    report.append("- **Generalist models**: Broad coverage, potentially lower quality")
    report.append("")
    report.append("Both metrics are important for making an informed decision!")
    report.append("")

    # Save report
    output_file = Path(__file__).parent.parent / "TWO_TIERED_EVALUATION_REPORT.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"\n✓ Generated markdown report: {output_file}")


if __name__ == "__main__":
    main()


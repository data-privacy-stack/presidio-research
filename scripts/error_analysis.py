"""
Comprehensive error analysis for all evaluated models
Uses presidio-evaluator's ModelError analysis tools
"""
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import SpanEvaluator, ModelError
from presidio_evaluator.models import PresidioAnalyzerWrapper
import pandas as pd
from collections import Counter, defaultdict

def load_experiment_results(exp_file):
    """Load experiment file and extract evaluation results"""
    with open(exp_file, 'r') as f:
        exp_data = json.load(f)
    return exp_data

def analyze_model_errors(model_name, exp_file, dataset):
    """Perform detailed error analysis for a model"""
    print("\n" + "="*100)
    print(f"ERROR ANALYSIS: {model_name}")
    print("="*100)

    exp_data = load_experiment_results(exp_file)
    metrics = exp_data['metrics']

    # Get per-entity performance
    print("\n📊 PER-ENTITY PERFORMANCE:")
    print("-" * 100)

    entity_metrics = defaultdict(dict)
    for key, value in metrics.items():
        if '_precision' in key or '_recall' in key or '_f' in key:
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                entity, metric = parts
                if entity not in ['pii', 'n']:
                    entity_metrics[entity][metric] = value

    # Sort by F-score
    sorted_entities = sorted(
        entity_metrics.items(),
        key=lambda x: x[1].get('f', 0) if x[1].get('f') else 0,
        reverse=True
    )

    for entity, metrics_dict in sorted_entities:
        f = metrics_dict.get('f', 0)
        p = metrics_dict.get('precision', 0)
        r = metrics_dict.get('recall', 0)

        if f > 0:
            status = "✅" if f > 0.8 else "⚠️" if f > 0.5 else "❌"
            print(f"{status} {entity:20s}: F={f:.3f}, P={p:.3f}, R={r:.3f}")

    # Identify problem entities
    print("\n🔍 PROBLEM ENTITIES (F-Score < 0.5):")
    print("-" * 100)

    problem_entities = []
    for entity, metrics_dict in entity_metrics.items():
        f = metrics_dict.get('f', 0)
        if f < 0.5 and f > 0:
            p = metrics_dict.get('precision', 0)
            r = metrics_dict.get('recall', 0)

            # Diagnose the problem
            if p < 0.5 and r < 0.5:
                issue = "Both precision and recall poor - model struggles with this entity"
            elif p < r:
                issue = "Low precision - many false positives"
            elif r < p:
                issue = "Low recall - missing many instances"
            else:
                issue = "Moderate performance issues"

            problem_entities.append({
                'entity': entity,
                'f': f,
                'p': p,
                'r': r,
                'issue': issue
            })
            print(f"❌ {entity:20s}: F={f:.3f}, P={p:.3f}, R={r:.3f}")
            print(f"   → {issue}")

    # Analyze confusion patterns from confusion matrix
    if 'confusion_matrix' in exp_data:
        print("\n🔀 CONFUSION MATRIX ANALYSIS:")
        print("-" * 100)

        matrix = exp_data['confusion_matrix']
        labels = exp_data['labels']

        # Find most confused pairs
        confusions = []
        for i, true_label in enumerate(labels):
            if true_label == 'O':
                continue
            for j, pred_label in enumerate(labels):
                if i != j and matrix[i][j] > 0:
                    confusions.append({
                        'true': true_label,
                        'predicted': pred_label,
                        'count': matrix[i][j]
                    })

        # Sort by count
        confusions.sort(key=lambda x: x['count'], reverse=True)

        print("\nTop 10 Confusion Pairs:")
        for conf in confusions[:10]:
            print(f"  • {conf['true']:20s} → {conf['predicted']:20s}: {conf['count']:3d} times")

        # Calculate false negatives (predicted as O)
        print("\n⚠️  FALSE NEGATIVES (Missed Detections):")
        fn_by_entity = {}
        o_index = labels.index('O') if 'O' in labels else None

        if o_index is not None:
            for i, true_label in enumerate(labels):
                if true_label != 'O' and matrix[i][o_index] > 0:
                    fn_by_entity[true_label] = matrix[i][o_index]

        if fn_by_entity:
            for entity, count in sorted(fn_by_entity.items(), key=lambda x: x[1], reverse=True)[:10]:
                total = sum(matrix[labels.index(entity)])
                miss_rate = count / total if total > 0 else 0
                print(f"  • {entity:20s}: {count:3d} missed ({miss_rate*100:.1f}% miss rate)")

        # Calculate false positives (O predicted as entity)
        print("\n⚠️  FALSE POSITIVES (Wrong Detections):")
        fp_by_entity = {}

        if o_index is not None:
            for j, pred_label in enumerate(labels):
                if pred_label != 'O' and matrix[o_index][j] > 0:
                    fp_by_entity[pred_label] = matrix[o_index][j]

        if fp_by_entity:
            for entity, count in sorted(fp_by_entity.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  • {entity:20s}: {count:3d} false positives")

    # Summary
    print("\n📋 SUMMARY:")
    print("-" * 100)
    overall_f = metrics.get('pii_f', 0)
    overall_p = metrics.get('pii_precision', 0)
    overall_r = metrics.get('pii_recall', 0)

    print(f"Overall: F={overall_f:.4f}, P={overall_p:.4f}, R={overall_r:.4f}")

    if overall_p > overall_r:
        print("• Model is CONSERVATIVE - prefers precision over recall")
        print("• Missing some PII (false negatives) but few false alarms")
    elif overall_r > overall_p:
        print("• Model is AGGRESSIVE - prefers recall over precision")
        print("• Catches more PII but has more false alarms")
    else:
        print("• Model is BALANCED - similar precision and recall")

    print(f"• {len(problem_entities)} entities with F-Score < 0.5")
    print(f"• {len([e for e, m in entity_metrics.items() if m.get('f', 0) > 0.8])} entities with F-Score > 0.8")

    return {
        'model': model_name,
        'overall_f': overall_f,
        'overall_p': overall_p,
        'overall_r': overall_r,
        'problem_entities': problem_entities,
        'strong_entities': [(e, m['f']) for e, m in entity_metrics.items() if m.get('f', 0) > 0.8]
    }

def main():
    data_dir = Path(__file__).parent.parent

    # Load dataset
    dataset = InputSample.read_dataset_json(data_dir / "data" / "synth_dataset_v2.json")

    models = {
        "BERT-base-NER": data_dir / "experiment_20251111-094238.json",
        "RoBERTa-i2b2": data_dir / "experiment_20251111-100735.json",
        "DeBERTa-PII": data_dir / "experiment_20251111-101707.json",
        "StanfordAIMI": data_dir / "experiment_20251111-094001.json"
    }

    print("="*100)
    print("COMPREHENSIVE ERROR ANALYSIS - ALL MODELS")
    print("="*100)
    print(f"\nAnalyzing {len(models)} models on {len(dataset)} samples")

    all_analyses = []

    for model_name, exp_file in models.items():
        if exp_file.exists():
            analysis = analyze_model_errors(model_name, exp_file, dataset)
            all_analyses.append(analysis)
        else:
            print(f"\n⚠️  Experiment file not found: {exp_file}")

    # Comparative analysis
    print("\n\n" + "="*100)
    print("COMPARATIVE ANALYSIS")
    print("="*100)

    print("\n🏆 STRONGEST ENTITIES BY MODEL:")
    print("-" * 100)
    for analysis in all_analyses:
        print(f"\n{analysis['model']}:")
        for entity, f_score in analysis['strong_entities'][:5]:
            print(f"  ✅ {entity:20s}: {f_score:.3f}")

    print("\n⚠️  WEAKEST ENTITIES BY MODEL:")
    print("-" * 100)
    for analysis in all_analyses:
        print(f"\n{analysis['model']}:")
        if analysis['problem_entities']:
            for prob in analysis['problem_entities'][:5]:
                print(f"  ❌ {prob['entity']:20s}: F={prob['f']:.3f} - {prob['issue']}")
        else:
            print("  ✓ No major problem entities")

    print("\n📊 PRECISION vs RECALL TRADE-OFFS:")
    print("-" * 100)
    for analysis in sorted(all_analyses, key=lambda x: x['overall_f'], reverse=True):
        p = analysis['overall_p']
        r = analysis['overall_r']
        diff = abs(p - r)

        if diff < 0.05:
            balance = "⚖️  BALANCED"
        elif p > r:
            balance = "🎯 PRECISION-FOCUSED"
        else:
            balance = "🔍 RECALL-FOCUSED"

        print(f"{analysis['model']:20s}: P={p:.3f}, R={r:.3f} ({balance})")

    # Save analysis
    output_file = data_dir / "data" / "error_analysis_summary.json"
    with open(output_file, 'w') as f:
        json.dump(all_analyses, f, indent=2, default=str)
    print(f"\n✓ Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()


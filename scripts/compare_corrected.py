"""
Generate comparison report for corrected evaluations
"""
from pathlib import Path
import json
import pandas as pd

def main():
    data_dir = Path(__file__).parent.parent / "data"

    result_files = {
        "StanfordAIMI": "stanford_corrected_results.json",
        "BERT-base-NER": "bert_ner_corrected_results.json",
        "RoBERTa-i2b2": "roberta_i2b2_corrected_results.json",
        "DeBERTa-PII": "deberta_pii_corrected_results.json"
    }

    print("="*100)
    print("CORRECTED MODEL COMPARISON - WITH PROPER ENTITY MAPPINGS")
    print("="*100)

    all_results = {}
    for model_name, filename in result_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                all_results[model_name] = json.load(f)
            print(f"✓ Loaded {model_name}")

    comparison = []
    for model_name, results in all_results.items():
        comparison.append({
            "Model": model_name,
            "F-Score": results["pii_f"],
            "Precision": results["pii_precision"],
            "Recall": results["pii_recall"]
        })

    df = pd.DataFrame(comparison)
    df = df.sort_values("F-Score", ascending=False)

    print("\n" + "="*100)
    print("OVERALL PERFORMANCE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    # Load old results for comparison
    print("\n" + "="*100)
    print("IMPROVEMENT AFTER CORRECTING MAPPINGS")
    print("="*100)

    old_scores = {
        "StanfordAIMI": 0.8115,
        "BERT-base-NER": 0.7800,
        "RoBERTa-i2b2": 0.5098,
        "DeBERTa-PII": 0.2556
    }

    for model_name, new_results in all_results.items():
        old_score = old_scores.get(model_name, 0)
        new_score = new_results["pii_f"]
        change = new_score - old_score
        pct_change = (change / old_score * 100) if old_score > 0 else 0

        symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"{symbol} {model_name:20s}: {old_score:.4f} → {new_score:.4f} ({change:+.4f}, {pct_change:+.1f}%)")

    # Save results
    df.to_csv(data_dir / "corrected_comparison.csv", index=False)

    with open(data_dir / "corrected_all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Saved to data/corrected_comparison.csv")

    # Best model
    print("\n" + "="*100)
    print("WINNER")
    print("="*100)
    best = df.iloc[0]
    print(f"🏆 {best['Model']}")
    print(f"   F-Score:   {best['F-Score']:.4f}")
    print(f"   Precision: {best['Precision']:.4f}")
    print(f"   Recall:    {best['Recall']:.4f}")
    print("="*100)

if __name__ == "__main__":
    main()


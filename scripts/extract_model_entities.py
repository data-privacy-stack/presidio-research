"""
Extract entity labels from Hugging Face model configs
"""
from transformers import AutoConfig
import json

def get_model_entities(model_name):
    """Extract entity labels from model config"""
    try:
        config = AutoConfig.from_pretrained(model_name)

        # Try different config attributes where labels might be stored
        labels = None
        if hasattr(config, 'id2label'):
            labels = list(config.id2label.values())
        elif hasattr(config, 'label2id'):
            labels = list(config.label2id.keys())

        return labels
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def main():
    models = {
        "StanfordAIMI": "StanfordAIMI/stanford-deidentifier-base",
        "BERT-base-NER": "dslim/bert-base-NER",
        "RoBERTa-i2b2": "obi/deid_roberta_i2b2",
        "DeBERTa-PII": "lakshyakh93/deberta_finetuned_pii"
    }

    print("="*100)
    print("EXTRACTING MODEL ENTITY LABELS")
    print("="*100)

    all_entities = {}

    for name, model_id in models.items():
        print(f"\n{'='*100}")
        print(f"Model: {name} ({model_id})")
        print("="*100)

        entities = get_model_entities(model_id)

        if entities:
            all_entities[name] = entities
            print(f"Found {len(entities)} entity labels:")

            # Remove 'O' and 'B-'/'I-' prefixes if present
            unique_entities = set()
            for label in entities:
                if label == 'O':
                    continue
                # Remove BIO prefixes
                clean_label = label.replace('B-', '').replace('I-', '')
                unique_entities.add(clean_label)

            print(f"\nUnique entities (after removing BIO prefixes): {len(unique_entities)}")
            print(sorted(unique_entities))
        else:
            print("Could not extract entity labels")

    # Save to JSON
    from pathlib import Path
    output_file = Path(__file__).parent.parent / "data" / "model_entities.json"
    with open(output_file, 'w') as f:
        json.dump(all_entities, f, indent=2)
    print(f"\n\n✓ Saved entity labels to: {output_file}")

if __name__ == "__main__":
    main()


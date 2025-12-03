"""
Create entity support mapping for fair model comparison.

This script analyzes which dataset entities each model can actually detect,
based on the entity mappings defined in the evaluation scripts.
"""
import json
from pathlib import Path
from typing import Dict, Set, List


def normalize_entity(entity: str) -> str:
    """Remove BIO/BILOU prefixes from entity labels."""
    prefixes = ['B-', 'I-', 'L-', 'U-', 'O-']
    for prefix in prefixes:
        if entity.startswith(prefix):
            return entity[len(prefix):]
    return entity


def get_model_supported_entities() -> Dict[str, Dict[str, Set[str]]]:
    """
    Define which dataset entities each model can detect.

    Returns:
        Dict mapping model name to:
            - 'raw_entities': Set of entities the model natively outputs
            - 'mapped_entities': Set of dataset entities the model can detect
    """

    # Load model entities
    model_entities_file = Path(__file__).parent.parent / "data" / "model_entities.json"
    with open(model_entities_file) as f:
        model_entities = json.load(f)

    support_mapping = {}

    # StanfordAIMI/stanford-deidentifier-base
    stanford_raw = {normalize_entity(e) for e in model_entities["StanfordAIMI"] if e != "O"}
    support_mapping["StanfordAIMI"] = {
        "raw_entities": stanford_raw,
        "mapped_entities": {
            "DATE_TIME",      # DATE → DATE_TIME
            "PHONE_NUMBER",   # PHONE → PHONE_NUMBER
            "PERSON",         # PATIENT, HCW → PERSON
            "ORGANIZATION",   # HOSPITAL, VENDOR → ORGANIZATION
            "US_DRIVER_LICENSE",  # ID → US_DRIVER_LICENSE (generic ID)
            "US_SSN",         # ID → US_SSN (generic ID)
        }
    }

    # dslim/bert-base-NER
    bert_raw = {normalize_entity(e) for e in model_entities["BERT-base-NER"] if e != "O"}
    support_mapping["BERT-base-NER"] = {
        "raw_entities": bert_raw,
        "mapped_entities": {
            "PERSON",         # PER → PERSON
            "ORGANIZATION",   # ORG → ORGANIZATION
            "GPE",            # LOC → GPE
            "STREET_ADDRESS", # LOC → STREET_ADDRESS
            # Note: MISC is ignored, doesn't map to dataset entities
        }
    }

    # obi/deid_roberta_i2b2
    roberta_raw = {normalize_entity(e) for e in model_entities["RoBERTa-i2b2"] if e != "O"}
    support_mapping["RoBERTa-i2b2"] = {
        "raw_entities": roberta_raw,
        "mapped_entities": {
            "AGE",            # AGE → AGE
            "DATE_TIME",      # DATE → DATE_TIME
            "EMAIL_ADDRESS",  # EMAIL → EMAIL_ADDRESS
            "PHONE_NUMBER",   # PHONE → PHONE_NUMBER
            "PERSON",         # PATIENT, STAFF → PERSON
            "ORGANIZATION",   # HOSP, PATORG → ORGANIZATION
            "GPE",            # LOC → GPE
            "STREET_ADDRESS", # LOC → STREET_ADDRESS
            "US_DRIVER_LICENSE",  # ID → US_DRIVER_LICENSE
            "US_SSN",         # ID → US_SSN
            # Note: OTHERPHI is ignored
        }
    }

    # lakshyakh93/deberta_finetuned_pii
    deberta_raw = {normalize_entity(e) for e in model_entities["DeBERTa-PII"] if e != "O"}
    support_mapping["DeBERTa-PII"] = {
        "raw_entities": deberta_raw,
        "mapped_entities": {
            "PERSON",         # PREFIX, FIRSTNAME, MIDDLENAME, LASTNAME, FULLNAME, NAME → PERSON
            "TITLE",          # PREFIX, SUFFIX → TITLE
            "ORGANIZATION",   # COMPANY_NAME → ORGANIZATION
            "EMAIL_ADDRESS",  # EMAIL → EMAIL_ADDRESS
            "DATE_TIME",      # DATE, TIME → DATE_TIME
            "DOMAIN_NAME",    # URL → DOMAIN_NAME
            "IP_ADDRESS",     # IPV4, IPV6, IP → IP_ADDRESS
            "STREET_ADDRESS", # STREETADDRESS, BUILDINGNUMBER, SECONDARYADDRESS, STREET → STREET_ADDRESS
            "GPE",            # CITY, STATE, COUNTY → GPE
            "ZIP_CODE",       # ZIPCODE → ZIP_CODE
            "PHONE_NUMBER",   # PHONE_NUMBER → PHONE_NUMBER
            "CREDIT_CARD",    # CREDITCARDNUMBER → CREDIT_CARD
            "IBAN_CODE",      # IBAN → IBAN_CODE
            "US_SSN",         # SSN → US_SSN
            # Has many more entities but these are the ones that map to dataset
        }
    }

    return support_mapping


def get_dataset_entities() -> Set[str]:
    """Get all entities present in the dataset."""
    # These are from the synthetic dataset
    return {
        "AGE",
        "CREDIT_CARD",
        "DATE_TIME",
        "DOMAIN_NAME",
        "EMAIL_ADDRESS",
        "GPE",
        "IBAN_CODE",
        "IP_ADDRESS",
        "NRP",
        "ORGANIZATION",
        "PERSON",
        "PHONE_NUMBER",
        "STREET_ADDRESS",
        "TITLE",
        "US_DRIVER_LICENSE",
        "US_SSN",
        "ZIP_CODE",
    }


def main():
    print("=" * 100)
    print("CREATING ENTITY SUPPORT MAPPING")
    print("=" * 100)

    support_mapping = get_model_supported_entities()
    dataset_entities = get_dataset_entities()

    print(f"\n📊 Dataset has {len(dataset_entities)} entity types")
    print(f"   {sorted(dataset_entities)}\n")

    # Create coverage report
    coverage_report = {}

    for model_name, mapping in support_mapping.items():
        mapped = mapping["mapped_entities"]
        raw = mapping["raw_entities"]

        # Calculate coverage
        supported_count = len(mapped)
        coverage_pct = (supported_count / len(dataset_entities)) * 100

        # Find unsupported entities
        unsupported = dataset_entities - mapped

        coverage_report[model_name] = {
            "raw_entity_count": len(raw),
            "raw_entities": sorted(raw),
            "supported_dataset_entities": sorted(mapped),
            "supported_count": supported_count,
            "coverage_percentage": round(coverage_pct, 1),
            "unsupported_dataset_entities": sorted(unsupported),
            "unsupported_count": len(unsupported),
        }

        print("=" * 100)
        print(f"Model: {model_name}")
        print("=" * 100)
        print(f"  Raw model entities: {len(raw)}")
        print(f"  → Maps to {supported_count}/{len(dataset_entities)} dataset entities ({coverage_pct:.1f}% coverage)")
        print(f"\n  ✅ SUPPORTED: {sorted(mapped)}")
        print(f"\n  ❌ UNSUPPORTED: {sorted(unsupported)}")
        print()

    # Save to JSON
    output_file = Path(__file__).parent.parent / "data" / "entity_support_mapping.json"
    with open(output_file, 'w') as f:
        json.dump({
            "dataset_entities": sorted(dataset_entities),
            "models": coverage_report,
        }, f, indent=2)

    print("=" * 100)
    print(f"✓ Saved entity support mapping to: {output_file}")
    print("=" * 100)

    # Print summary table
    print("\n📊 COVERAGE SUMMARY\n")
    print("| Model | Raw Entities | Supported Dataset Entities | Coverage |")
    print("|-------|--------------|---------------------------|----------|")
    for model_name, report in coverage_report.items():
        print(f"| {model_name:20s} | {report['raw_entity_count']:12d} | "
              f"{report['supported_count']:25d} | {report['coverage_percentage']:7.1f}% |")


if __name__ == "__main__":
    main()


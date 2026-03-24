import ast
import json
import re
import urllib.request
from collections import Counter
from huggingface_hub import HfApi
from transformers import AutoConfig


# ---------------------------------------------------------------------------
# Vendor / reference entity lists
# Sources:
#   Presidio:           https://microsoft.github.io/presidio/supported_entities/
#   Azure Deident.:     https://learn.microsoft.com/en-us/dotnet/api/azure.health.deidentification.phicategory
#   Private.ai:         https://docs.private-ai.com/entities/
#   Amazon Macie:       https://docs.aws.amazon.com/macie/latest/user/mdis-reference-pii.html
# ---------------------------------------------------------------------------

PRESIDIO_ENTITIES = [
    # Global
    "CREDIT_CARD", "CRYPTO", "DATE_TIME", "EMAIL_ADDRESS", "IBAN_CODE",
    "IP_ADDRESS", "MAC_ADDRESS", "NRP", "LOCATION", "PERSON", "PHONE_NUMBER",
    "MEDICAL_LICENSE", "URL",
    # USA
    "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_MBI", "US_NPI",
    "US_PASSPORT", "US_SSN",
    # UK
    "UK_NHS", "UK_NINO", "UK_PASSPORT", "UK_POSTCODE", "UK_VEHICLE_REGISTRATION",
    # Spain
    "ES_NIF", "ES_NIE",
    # Italy
    "IT_FISCAL_CODE", "IT_DRIVER_LICENSE", "IT_VAT_CODE", "IT_PASSPORT",
    "IT_IDENTITY_CARD",
    # Poland
    "PL_PESEL",
    # Singapore
    "SG_NRIC_FIN", "SG_UEN",
    # Australia
    "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE",
    # India
    "IN_PAN", "IN_AADHAAR", "IN_VEHICLE_REGISTRATION", "IN_VOTER",
    "IN_PASSPORT", "IN_GSTIN",
    # Finland
    "FI_PERSONAL_IDENTITY_CODE",
    # Korea
    "KR_DRIVER_LICENSE", "KR_FRN", "KR_PASSPORT", "KR_BRN", "KR_RRN",
    # Nigeria
    "NG_NIN", "NG_VEHICLE_REGISTRATION",
    # Thailand
    "TH_TNIN",
    # Medical / Clinical (via MedicalNERRecognizer)
    "MEDICAL_DISEASE_DISORDER", "MEDICAL_MEDICATION",
    "MEDICAL_THERAPEUTIC_PROCEDURE", "MEDICAL_CLINICAL_EVENT",
    "MEDICAL_BIOLOGICAL_ATTRIBUTE", "MEDICAL_BIOLOGICAL_STRUCTURE",
    "MEDICAL_FAMILY_HISTORY", "MEDICAL_HISTORY",
]

# Azure Health Deidentification service — PhiCategory enum values
AZURE_DEID_ENTITIES = [
    "ACCOUNT", "AGE", "BIOID", "CITY", "COUNTRY_OR_REGION", "DATE", "DEVICE",
    "DOCTOR", "EMAIL", "FAX", "HEALTH_PLAN", "HOSPITAL", "ID_NUM", "IP_ADDRESS",
    "LICENSE", "LOCATION_OTHER", "MEDICAL_RECORD", "ORGANIZATION", "PATIENT",
    "PHONE", "PROFESSION", "SOCIAL_SECURITY", "STATE", "STREET", "URL",
    "USERNAME", "VEHICLE", "ZIP",
]

# Private.ai — PII + Health Information + PCI entities
PRIVATE_AI_ENTITIES = [
    # PII
    "ACCOUNT_NUMBER", "AGE", "DATE", "DATE_INTERVAL", "DOB", "DRIVER_LICENSE",
    "DURATION", "EMAIL_ADDRESS", "EVENT", "FILENAME", "GENDER",
    "HEALTHCARE_NUMBER", "IP_ADDRESS", "LANGUAGE", "LOCATION",
    "LOCATION_ADDRESS", "LOCATION_ADDRESS_STREET", "LOCATION_CITY",
    "LOCATION_COORDINATE", "LOCATION_COUNTRY", "LOCATION_STATE", "LOCATION_ZIP",
    "MARITAL_STATUS", "MONEY", "NAME", "NAME_FAMILY", "NAME_GIVEN",
    "NAME_MEDICAL_PROFESSIONAL", "NUMERICAL_PII", "OCCUPATION", "ORGANIZATION",
    "ORGANIZATION_MEDICAL_FACILITY", "ORIGIN", "PASSPORT_NUMBER", "PASSWORD",
    "PHONE_NUMBER", "PHYSICAL_ATTRIBUTE", "POLITICAL_AFFILIATION", "RELIGION",
    "SEXUALITY", "SSN", "TIME", "URL", "USERNAME", "VEHICLE_ID", "ZODIAC_SIGN",
    # Health Information
    "BLOOD_TYPE", "CONDITION", "DOSE", "DRUG", "INJURY", "MEDICAL_PROCESS",
    "STATISTICS",
    # PCI
    "BANK_ACCOUNT", "CREDIT_CARD", "CREDIT_CARD_EXPIRATION", "CVV",
    "ROUTING_NUMBER",
]

# Amazon Macie — Managed Data Identifiers for PII
# Source: https://docs.aws.amazon.com/macie/latest/user/mdis-reference-pii.html
# Managed data identifier IDs as listed in the official documentation.
AMAZON_MACIE_ENTITIES = [
    # Birth date
    "DATE_OF_BIRTH",
    # Driver's license (per country)
    "DRIVERS_LICENSE",              # US
    "AUSTRALIA_DRIVERS_LICENSE",
    "AUSTRIA_DRIVERS_LICENSE",
    "BELGIUM_DRIVERS_LICENSE",
    "BULGARIA_DRIVERS_LICENSE",
    "CANADA_DRIVERS_LICENSE",
    "CROATIA_DRIVERS_LICENSE",
    "CYPRUS_DRIVERS_LICENSE",
    "CZECHIA_DRIVERS_LICENSE",
    "DENMARK_DRIVERS_LICENSE",
    "ESTONIA_DRIVERS_LICENSE",
    "FINLAND_DRIVERS_LICENSE",
    "FRANCE_DRIVERS_LICENSE",
    "GERMANY_DRIVERS_LICENSE",
    "GREECE_DRIVERS_LICENSE",
    "HUNGARY_DRIVERS_LICENSE",
    "INDIA_DRIVERS_LICENSE",
    "IRELAND_DRIVERS_LICENSE",
    "ITALY_DRIVERS_LICENSE",
    "LATVIA_DRIVERS_LICENSE",
    "LITHUANIA_DRIVERS_LICENSE",
    "LUXEMBOURG_DRIVERS_LICENSE",
    "MALTA_DRIVERS_LICENSE",
    "NETHERLANDS_DRIVERS_LICENSE",
    "POLAND_DRIVERS_LICENSE",
    "PORTUGAL_DRIVERS_LICENSE",
    "ROMANIA_DRIVERS_LICENSE",
    "SLOVAKIA_DRIVERS_LICENSE",
    "SLOVENIA_DRIVERS_LICENSE",
    "SPAIN_DRIVERS_LICENSE",
    "SWEDEN_DRIVERS_LICENSE",
    "UK_DRIVERS_LICENSE",
    # Electoral roll number
    "UK_ELECTORAL_ROLL_NUMBER",
    # Full name
    "NAME",
    # GPS coordinates
    "LATITUDE_LONGITUDE",
    # HTTP cookie
    "HTTP_COOKIE",
    # Mailing address
    "ADDRESS",
    "BRAZIL_CEP_CODE",
    # National identification number (per country)
    "ARGENTINA_DNI_NUMBER",
    "BRAZIL_RG_NUMBER",
    "CHILE_RUT_NUMBER",
    "COLOMBIA_CITIZENSHIP_CARD_NUMBER",
    "FRANCE_NATIONAL_IDENTIFICATION_NUMBER",
    "GERMANY_NATIONAL_IDENTIFICATION_NUMBER",
    "INDIA_AADHAAR_NUMBER",
    "ITALY_NATIONAL_IDENTIFICATION_NUMBER",
    "MEXICO_CURP_NUMBER",
    "SPAIN_DNI_NUMBER",
    # National Insurance Number
    "UK_NATIONAL_INSURANCE_NUMBER",
    # Passport number (per country)
    "CANADA_PASSPORT_NUMBER",
    "FRANCE_PASSPORT_NUMBER",
    "GERMANY_PASSPORT_NUMBER",
    "ITALY_PASSPORT_NUMBER",
    "SPAIN_PASSPORT_NUMBER",
    "UK_PASSPORT_NUMBER",
    "USA_PASSPORT_NUMBER",
    # Permanent residence number
    "CANADA_NATIONAL_IDENTIFICATION_NUMBER",
    # Phone number (per country)
    "PHONE_NUMBER",                 # Canada and US
    "BRAZIL_PHONE_NUMBER",
    "FRANCE_PHONE_NUMBER",
    "GERMANY_PHONE_NUMBER",
    "ITALY_PHONE_NUMBER",
    "SPAIN_PHONE_NUMBER",
    "UK_PHONE_NUMBER",
    # Public transportation card number
    "ARGENTINA_TARJETA_SUBE",
    # Social Insurance Number
    "CANADA_SOCIAL_INSURANCE_NUMBER",
    # Social Security number
    "USA_SOCIAL_SECURITY_NUMBER",
    "SPAIN_SOCIAL_SECURITY_NUMBER",
    # Taxpayer identification / reference number (per country)
    "ARGENTINA_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER",
    "ARGENTINA_ORGANIZATION_TAX_IDENTIFICATION_NUMBER",
    "AUSTRALIA_TAX_FILE_NUMBER",
    "BRAZIL_CNPJ_NUMBER",
    "BRAZIL_CPF_NUMBER",
    "CHILE_RUT_NUMBER",
    "COLOMBIA_INDIVIDUAL_NIT_NUMBER",
    "COLOMBIA_ORGANIZATION_NIT_NUMBER",
    "FRANCE_TAX_IDENTIFICATION_NUMBER",
    "GERMANY_TAX_IDENTIFICATION_NUMBER",
    "INDIA_PERMANENT_ACCOUNT_NUMBER",
    "ITALY_NATIONAL_IDENTIFICATION_NUMBER",
    "MEXICO_INDIVIDUAL_RFC_NUMBER",
    "MEXICO_ORGANIZATION_RFC_NUMBER",
    "SPAIN_NIE_NUMBER",
    "SPAIN_NIF_NUMBER",
    "SPAIN_TAX_IDENTIFICATION_NUMBER",
    "UK_TAX_IDENTIFICATION_NUMBER",
    "USA_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER",
    # Vehicle identification number
    "VEHICLE_IDENTIFICATION_NUMBER",
]

VENDOR_LISTS = {
    "Presidio": PRESIDIO_ENTITIES,
    "Azure Health Deidentification": AZURE_DEID_ENTITIES,
    "Private.ai": PRIVATE_AI_ENTITIES,
    "Amazon Macie": AMAZON_MACIE_ENTITIES,
}


# ---------------------------------------------------------------------------
# HuggingFace model scraping
# ---------------------------------------------------------------------------

def strip_bio_tags(label: str) -> str:
    """Remove BIO/BIOLU prefix or postfix tags (e.g. B-LOC, LOC-B, I-LOC, LOC-U)."""
    label = str(label)
    label = re.sub(r"^[BILUS]-", "", label)   # prefix: B-LOC  → LOC
    label = re.sub(r"-[BILUS]$", "", label)   # postfix: LOC-B → LOC
    return label


def extract_entities_from_query(search_terms, limit_per_term=50):
    """
    Returns a dict keyed by search term. Each value is a dict with:
      - "entity_counts":  Counter(entity -> # models that return it)
      - "model_entities": dict(model_id -> sorted list of entities)
    """
    api = HfApi()
    all_results = {}

    for term in search_terms:
        print(f"--- Processing search term: '{term}' ---")

        try:
            models = api.list_models(
                task="token-classification",
                search=term,
                sort="downloads",
                limit=limit_per_term,
            )
        except Exception as e:
            print(f"Error fetching models for '{term}': {e}")
            continue

        entity_counts = Counter()
        model_entities = {}

        for model in models:
            model_id = model.id
            try:
                config = AutoConfig.from_pretrained(model_id)
                labels = getattr(config, "id2label", {})
                if not isinstance(labels, dict):
                    continue

                model_clean_labels = set()
                for label_name in labels.values():
                    clean = strip_bio_tags(label_name)
                    if clean.upper() not in ("O", "OTHER") and "LABEL_" not in clean.upper():
                        model_clean_labels.add(clean)

                if model_clean_labels:
                    model_entities[model_id] = sorted(model_clean_labels)
                    entity_counts.update(model_clean_labels)

            except Exception:
                continue

        all_results[term] = {
            "entity_counts": entity_counts,
            "model_entities": model_entities,
        }
        print(f"Found {len(entity_counts)} unique entities across {len(model_entities)} models.\n")

    return all_results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_summary(results, title):
    print(f"\n{'='*20} {title} {'='*20}")
    for term, data in results.items():
        entity_counts = data["entity_counts"]
        model_entities = data["model_entities"]

        print(f"\nKeyword: {term.upper()}")

        print("\n  Entities by popularity (# models):")
        if entity_counts:
            for entity, count in entity_counts.most_common():
                print(f"    {entity}: {count} model(s)")
        else:
            print("    No specific entities found.")

        print("\n  Entities per model:")
        if model_entities:
            for model_id, entities in model_entities.items():
                print(f"    {model_id}: {', '.join(entities)}")
        else:
            print("    No models with labeled entities found.")


def print_combined_totals(results, vendor_lists=None):
    """Aggregate entity counts across all search terms and vendor lists, then print ranked."""
    combined = Counter()
    for data in results.values():
        combined.update(data["entity_counts"])

    if vendor_lists:
        for entities in vendor_lists.values():
            combined.update(entities)

    print(f"\n{'='*20} COMBINED TOTALS (HuggingFace models + vendors) {'='*20}")
    if combined:
        for entity, count in combined.most_common():
            print(f"  {entity}: {count}")
    else:
        print("  No entities found.")


def print_vendor_lists(vendor_lists):
    print(f"\n{'='*20} VENDOR / REFERENCE ENTITY LISTS {'='*20}")
    for vendor, entities in vendor_lists.items():
        print(f"\n  {vendor} ({len(entities)} entities):")
        print(f"    {', '.join(sorted(entities))}")


# ---------------------------------------------------------------------------
# HuggingFace dataset scraping
# ---------------------------------------------------------------------------

_SPAN_ENTITY_FIELDS = frozenset({
    "label", "entity_type", "tag", "entity", "type",
    "ner_tag", "pii_type", "class", "category",
})

_HF_DS_SERVER = "https://datasets-server.huggingface.co"


def _fetch_ds_json(path, timeout=15):
    """GET a path on the HF datasets-server and return parsed JSON, or None on failure."""
    try:
        with urllib.request.urlopen(f"{_HF_DS_SERVER}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _classlabels_from_feature_list(features):
    """
    Recursively parse a HuggingFace dataset-card features list and return all
    ClassLabel names (handles both dict and list ``names`` mappings).

    Columns whose names hint at POS/syntax tagging (``pos``, ``chunk``, ``dep``,
    ``morph``) are skipped so that only NER-relevant labels are included.
    """
    _SYNTAX_COL_HINTS = ("pos", "chunk", "dep", "morph", "xpos", "upos")
    entities = set()
    if not isinstance(features, list):
        return entities
    for f in features:
        if not isinstance(f, dict):
            continue
        col_lower = f.get("name", "").lower()
        if any(hint in col_lower for hint in _SYNTAX_COL_HINTS):
            continue
        for container_key in ("sequence", "dtype"):
            container = f.get(container_key)
            if isinstance(container, dict) and "class_label" in container:
                names = container["class_label"].get("names", {})
                name_iter = names.values() if isinstance(names, dict) else names
                for name in name_iter:
                    clean = strip_bio_tags(str(name))
                    if clean.upper() not in ("O", "OTHER") and "LABEL_" not in clean.upper():
                        entities.add(clean)
        # Recurse into nested list subfields (span-based schemas)
        sub_list = f.get("list")
        if isinstance(sub_list, list):
            entities |= _classlabels_from_feature_list(sub_list)
    return entities


def _classlabels_from_card(dataset_id):
    """
    Extract entity types from a dataset README card's ClassLabel features.
    Returns an empty set when the dataset has no ClassLabel features or card data.
    """
    try:
        info = HfApi().dataset_info(dataset_id)
        if not info.card_data:
            return set()
        d = info.card_data.to_dict()
    except Exception:
        return set()

    ds_info = d.get("dataset_info", {})
    configs = ds_info if isinstance(ds_info, list) else [ds_info]
    entities = set()
    for cfg in configs:
        if isinstance(cfg, dict):
            entities |= _classlabels_from_feature_list(cfg.get("features", []))
    return entities


def _scan_row(obj, entities):
    """
    Recursively scan a dataset row object for span-dict entity-type values.

    Handles three column formats:

    * **Native list** – the column is already a list of span dicts, returned
      as a Python list by the datasets-server.
    * **JSON string** – the column is a ``dtype: string`` field whose value is
      a JSON-encoded list of span dicts (e.g. ``gretelai/synthetic_pii_…``).
    * **Python-repr string** – the column stores a Python ``repr()`` of a list
      of span dicts with single-quoted keys (e.g. ``ai4privacy/pii-masking-…``).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in _SPAN_ENTITY_FIELDS and isinstance(v, str) and v:
                clean = strip_bio_tags(v)
                if clean.upper() not in ("O", "OTHER"):
                    entities.add(clean)
            else:
                _scan_row(v, entities)
    elif isinstance(obj, list):
        for item in obj:
            _scan_row(item, entities)
    elif isinstance(obj, str):
        s = obj.strip()
        if s.startswith("[") or s.startswith("{"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    _scan_row(parser(s), entities)
                    break
                except Exception:
                    pass


def _entities_for_one_dataset(dataset_id):
    """
    Extract entity types from a HuggingFace dataset using two strategies:

    1. **ClassLabel names** from the dataset card README features (fast,
       no row fetching needed).  Works for token-classification BIO datasets.
    2. **Row sampling** via the datasets-server ``/first-rows`` endpoint,
       scanning span-object fields.  Works for span-based datasets where entity
       types are stored as string values in nested dicts.
    """
    # Strategy 1 – ClassLabel features from card metadata
    entities = _classlabels_from_card(dataset_id)
    if entities:
        return entities

    # Strategy 2 – sample first rows; discover config+split via /splits
    splits_data = _fetch_ds_json(f"/splits?dataset={dataset_id}")
    if not splits_data:
        return set()

    for split_info in splits_data.get("splits", []):
        config = split_info.get("config", "default")
        split = split_info.get("split", "train")
        rows_data = _fetch_ds_json(
            f"/first-rows?dataset={dataset_id}&config={config}&split={split}"
        )
        if rows_data and rows_data.get("rows"):
            entities = set()
            for row_wrapper in rows_data["rows"]:
                _scan_row(row_wrapper.get("row", {}), entities)
            if entities:
                return entities

    return set()


def extract_entities_from_datasets(search_terms, limit_per_term=30):
    """
    Search HuggingFace *datasets* for NER/PII entity labels.

    Supports both token-classification datasets (ClassLabel features in the
    dataset card README) and span-based datasets (JSON or Python-repr list
    fields containing entity-type strings).

    Args:
        search_terms:   List of keyword strings to search HuggingFace with.
        limit_per_term: Maximum number of datasets to inspect per search term.

    Returns:
        Dict keyed by search term.  Each value is a dict with:

        * ``entity_counts``    – Counter(entity -> # datasets it appears in)
        * ``dataset_entities`` – dict(dataset_id -> sorted list of entities)
    """
    api = HfApi()
    all_results = {}

    for term in search_terms:
        print(f"--- Processing dataset search term: '{term}' ---")
        try:
            datasets = list(api.list_datasets(
                search=term,
                sort="downloads",
                limit=limit_per_term,
            ))
        except Exception as e:
            print(f"Error fetching datasets for '{term}': {e}")
            all_results[term] = {"entity_counts": Counter(), "dataset_entities": {}}
            continue

        entity_counts = Counter()
        dataset_entities = {}

        for ds in datasets:
            ds_id = ds.id
            try:
                entities = _entities_for_one_dataset(ds_id)
                if entities:
                    dataset_entities[ds_id] = sorted(entities)
                    entity_counts.update(entities)
            except Exception:
                continue

        all_results[term] = {
            "entity_counts": entity_counts,
            "dataset_entities": dataset_entities,
        }
        print(
            f"Found {len(entity_counts)} unique entities "
            f"across {len(dataset_entities)} datasets.\n"
        )

    return all_results


# ---------------------------------------------------------------------------
# Dataset display helpers
# ---------------------------------------------------------------------------

def print_dataset_summary(results, title):
    print(f"\n{'='*20} {title} {'='*20}")
    for term, data in results.items():
        entity_counts = data["entity_counts"]
        dataset_entities = data["dataset_entities"]

        print(f"\nKeyword: {term.upper()}")

        print("\n  Entities by popularity (# datasets):")
        if entity_counts:
            for entity, count in entity_counts.most_common():
                print(f"    {entity}: {count} dataset(s)")
        else:
            print("    No specific entities found.")

        print("\n  Entities per dataset:")
        if dataset_entities:
            for ds_id, entities in dataset_entities.items():
                print(f"    {ds_id}: {', '.join(entities)}")
        else:
            print("    No datasets with labeled entities found.")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    privacy_results = extract_entities_from_query(["pii", "phi", "privacy"])

    print_summary(privacy_results, "PRIVACY/PII ENTITIES FROM MODELS (per search term)")
    print_combined_totals(privacy_results, VENDOR_LISTS)
    print_vendor_lists(VENDOR_LISTS)

    dataset_results = extract_entities_from_datasets(["pii", "phi", "privacy"])

    print_dataset_summary(dataset_results, "PRIVACY/PII ENTITIES FROM DATASETS (per search term)")
    print_combined_totals(dataset_results)


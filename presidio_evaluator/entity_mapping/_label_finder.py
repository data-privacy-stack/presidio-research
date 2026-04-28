"""
label_finder.py — discover PII/NER entity labels from HuggingFace and reference services.

Two discovery strategies:
  • Models  — queries HuggingFace token-classification models, reads id2label from
              each model config, strips BIO prefixes, and aggregates by frequency.
  • Datasets — queries HuggingFace datasets, extracts ClassLabel feature names from
               dataset card metadata or by sampling first rows via the datasets-server API.

A curated reference list (THIRD_PARTY_SERVICE_ENTITIES) captures entity names used by
major PII detection services and is included in the combined frequency totals.

Entry point: run as a module (`python -m presidio_evaluator.entity_mapping.label_finder`) to print ranked summaries for
the search terms "pii", "phi", and "privacy".
"""

import ast
import json
import urllib.request
from collections import Counter

from huggingface_hub import HfApi
from transformers import AutoConfig

from presidio_evaluator.entity_mapping import EntityHierarchy

# ---------------------------------------------------------------------------
# Third-party service entity reference list
# Aggregated from multiple PII detection services and NER datasets.
# ---------------------------------------------------------------------------

THIRD_PARTY_SERVICE_ENTITIES = [
    # Global
    "CREDIT_CARD",
    "CRYPTO",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "NRP",
    "LOCATION",
    "PERSON",
    "PHONE_NUMBER",
    "MEDICAL_LICENSE",
    "URL",
    # USA
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_MBI",
    "US_NPI",
    "US_PASSPORT",
    "US_SSN",
    # UK
    "UK_NHS",
    "UK_NINO",
    "UK_PASSPORT",
    "UK_POSTCODE",
    "UK_VEHICLE_REGISTRATION",
    # Spain
    "ES_NIF",
    "ES_NIE",
    # Italy
    "IT_FISCAL_CODE",
    "IT_DRIVER_LICENSE",
    "IT_VAT_CODE",
    "IT_PASSPORT",
    "IT_IDENTITY_CARD",
    # Poland
    "PL_PESEL",
    # Singapore
    "SG_NRIC_FIN",
    "SG_UEN",
    # Australia
    "AU_ABN",
    "AU_ACN",
    "AU_TFN",
    "AU_MEDICARE",
    # India
    "IN_PAN",
    "IN_AADHAAR",
    "IN_VEHICLE_REGISTRATION",
    "IN_VOTER",
    "IN_PASSPORT",
    "IN_GSTIN",
    # Finland
    "FI_PERSONAL_IDENTITY_CODE",
    # Korea
    "KR_DRIVER_LICENSE",
    "KR_FRN",
    "KR_PASSPORT",
    "KR_BRN",
    "KR_RRN",
    # Nigeria
    "NG_NIN",
    "NG_VEHICLE_REGISTRATION",
    # Thailand
    "TH_TNIN",
    # Medical / Clinical (via MedicalNERRecognizer)
    "MEDICAL_DISEASE_DISORDER",
    "MEDICAL_MEDICATION",
    "MEDICAL_THERAPEUTIC_PROCEDURE",
    "MEDICAL_CLINICAL_EVENT",
    "MEDICAL_BIOLOGICAL_ATTRIBUTE",
    "MEDICAL_BIOLOGICAL_STRUCTURE",
    "MEDICAL_FAMILY_HISTORY",
    "MEDICAL_HISTORY",
    "ACCOUNT",
    "AGE",
    "BIOID",
    "CITY",
    "COUNTRY_OR_REGION",
    "DATE",
    "DEVICE",
    "DOCTOR",
    "EMAIL",
    "FAX",
    "HEALTH_PLAN",
    "HOSPITAL",
    "ID_NUM",
    "LICENSE",
    "LOCATION_OTHER",
    "MEDICAL_RECORD",
    "ORGANIZATION",
    "PATIENT",
    "PHONE",
    "PROFESSION",
    "SOCIAL_SECURITY",
    "STATE",
    "STREET",
    "VEHICLE",
    "ZIP",
    "ACCOUNT_NUMBER",
    "DATE_INTERVAL",
    "DOB",
    "DRIVER_LICENSE",
    "DURATION",
    "EVENT",
    "FILENAME",
    "GENDER",
    "HEALTHCARE_NUMBER",
    "LANGUAGE",
    "LOCATION_ADDRESS",
    "LOCATION_ADDRESS_STREET",
    "LOCATION_CITY",
    "LOCATION_COORDINATE",
    "LOCATION_COUNTRY",
    "LOCATION_STATE",
    "LOCATION_ZIP",
    "MARITAL_STATUS",
    "MONEY",
    "NAME",
    "NAME_FAMILY",
    "NAME_GIVEN",
    "NAME_MEDICAL_PROFESSIONAL",
    "NUMERICAL_PII",
    "OCCUPATION",
    "ORGANIZATION_MEDICAL_FACILITY",
    "ORIGIN",
    "PASSPORT_NUMBER",
    "PASSWORD",
    "PHYSICAL_ATTRIBUTE",
    "POLITICAL_AFFILIATION",
    "RELIGION",
    "SEXUALITY",
    "SSN",
    "TIME",
    "VEHICLE_ID",
    "ZODIAC_SIGN",
    # Health Information
    "BLOOD_TYPE",
    "CONDITION",
    "DOSE",
    "DRUG",
    "INJURY",
    "MEDICAL_PROCESS",
    "STATISTICS",
    # PCI
    "BANK_ACCOUNT",
    "CREDIT_CARD_EXPIRATION",
    "CVV",
    "ROUTING_NUMBER",
    # Birth date
    "DATE_OF_BIRTH",
    # Driver's license (per country)
    "DRIVERS_LICENSE",  # US
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
    "COLOMBIA_INDIVIDUAL_NIT_NUMBER",
    "COLOMBIA_ORGANIZATION_NIT_NUMBER",
    "FRANCE_TAX_IDENTIFICATION_NUMBER",
    "GERMANY_TAX_IDENTIFICATION_NUMBER",
    "INDIA_PERMANENT_ACCOUNT_NUMBER",
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

# ---------------------------------------------------------------------------
# HuggingFace model scraping
# ---------------------------------------------------------------------------


def _normalize(label: str) -> str:
    """Normalize a label to UPPER_SNAKE_CASE for deduplication and comparison."""
    return (
        EntityHierarchy._strip_bio(str(label))
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _clean_label(raw: str) -> str | None:
    """
    Strip BIO prefix/suffix and return None for labels that should be discarded
    (outside token, generic LABEL_ placeholders).
    """
    clean = EntityHierarchy._strip_bio(str(raw))
    if _normalize(clean) in ("O", "OTHER") or "LABEL" in _normalize(clean):
        return None
    return clean


def _add_to_counter(counter: "Counter[str]", label: str) -> None:
    """
    Add *label* to *counter* using its normalized form as the key.

    The stored key is the UPPER_SNAKE_CASE form: if the normalized key already
    exists in the counter we keep whatever spelling is already there; otherwise
    we record the label as-is (preserving the first-seen spelling).
    """
    key = _normalize(label)
    # Find the existing display name for this key, or use the new label
    for existing in list(counter):
        if _normalize(existing) == key:
            counter[existing] += 1
            return
    counter[label] += 1


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
                pipeline_tag="token-classification",
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
                    clean = _clean_label(label_name)
                    if clean is not None:
                        model_clean_labels.add(clean)

                if model_clean_labels:
                    model_entities[model_id] = sorted(model_clean_labels)
                    for label in model_clean_labels:
                        _add_to_counter(entity_counts, label)

            except Exception:  # noqa: S112 — intentional: skip datasets that fail to load
                continue

        all_results[term] = {
            "entity_counts": entity_counts,
            "model_entities": model_entities,
        }
        print(
            f"Found {len(entity_counts)} unique entities across {len(model_entities)} models.\n",
        )

    return all_results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _merge_results(results):
    """
    Merge per-search-term result dicts into one combined entry.

    For model results (keys: ``entity_counts``, ``model_entities``) and dataset
    results (keys: ``entity_counts``, ``dataset_entities``), counters are summed
    and item dicts are union-merged.

    Returns a dict with the same structure as a single per-term result.
    """
    merged_counts: Counter = Counter()
    merged_items: dict = {}  # model_entities or dataset_entities
    items_key = None

    for data in results.values():
        merged_counts.update(data["entity_counts"])
        for key in data:
            if key != "entity_counts":
                items_key = key
                merged_items.update(data[key])

    out = {"entity_counts": merged_counts}
    if items_key:
        out[items_key] = merged_items
    return out


def print_summary(results, title, top_n=20) -> None:
    merged = _merge_results(results)
    entity_counts = merged["entity_counts"]
    model_entities = merged.get("model_entities", {})

    print(
        f"\n{'=' * 20} {title} {'=' * 20}\n"
        f"  {len(model_entities)} models · {len(entity_counts)} unique entities"
    )

    if not entity_counts:
        print("  No specific entities found.")
        return

    ranked = entity_counts.most_common()
    top = ranked[:top_n]
    rest = len(ranked) - top_n
    print(f"  Top {min(top_n, len(ranked))} entities by model count:")
    max_len = max(len(e) for e, _ in top)
    for entity, count in top:
        bar = "█" * count
        print(f"    {entity:<{max_len}}  {count:>3}  {bar}")
    if rest > 0:
        print(f"    ... and {rest} more")


def print_combined_totals(results, reference_entities=None, top_n=30) -> None:
    """Aggregate entity counts across all search terms and optional reference entity list, then print ranked."""
    combined = Counter()
    for data in results.values():
        combined.update(data["entity_counts"])

    if reference_entities:
        combined.update(reference_entities)

    print(
        f"\n{'=' * 20} COMBINED TOTALS (HuggingFace + third-party services) {'=' * 20}"
    )
    if not combined:
        print("  No entities found.")
        return

    ranked = combined.most_common()
    top = ranked[:top_n]
    rest = len(ranked) - top_n
    max_len = max(len(e) for e, _ in top)
    max_count = top[0][1] if top else 1
    bar_width = 30
    for entity, count in top:
        bar = "█" * int(count / max_count * bar_width)
        print(f"  {entity:<{max_len}}  {count:>4}  {bar}")
    if rest > 0:
        print(f"  ... and {rest} more")


# ---------------------------------------------------------------------------
# HuggingFace dataset scraping
# ---------------------------------------------------------------------------

_SPAN_ENTITY_FIELDS = frozenset(
    {
        "label",
        "entity_type",
        "tag",
        "entity",
        "type",
        "ner_tag",
        "pii_type",
        "class",
        "category",
    },
)

_HF_DS_SERVER = "https://datasets-server.huggingface.co"


def _fetch_ds_json(path, timeout=15):
    """GET a path on the HF datasets-server and return parsed JSON, or None on failure."""
    try:
        with urllib.request.urlopen(f"{_HF_DS_SERVER}{path}", timeout=timeout) as r:  # noqa: S310
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
    _syntax_col_hints = ("pos", "chunk", "dep", "morph", "xpos", "upos")
    entities = set()
    if not isinstance(features, list):
        return entities
    for f in features:
        if not isinstance(f, dict):
            continue
        col_lower = f.get("name", "").lower()
        if any(hint in col_lower for hint in _syntax_col_hints):
            continue
        for container_key in ("sequence", "dtype"):
            container = f.get(container_key)
            if isinstance(container, dict) and "class_label" in container:
                names = container["class_label"].get("names", {})
                name_iter = names.values() if isinstance(names, dict) else names
                for name in name_iter:
                    clean = _clean_label(str(name))
                    if clean is not None:
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


def _scan_row(obj, entities) -> None:
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
                clean = _clean_label(v)
                if clean is not None:
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
                except Exception:  # noqa: S110 — intentional: parse failure is expected for non-JSON strings
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
            f"/first-rows?dataset={dataset_id}&config={config}&split={split}",
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
            datasets = list(
                api.list_datasets(
                    search=term,
                    sort="downloads",
                    limit=limit_per_term,
                ),
            )
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
                    for label in entities:
                        _add_to_counter(entity_counts, label)
            except Exception:  # noqa: S112
                continue

        all_results[term] = {
            "entity_counts": entity_counts,
            "dataset_entities": dataset_entities,
        }
        print(
            f"Found {len(entity_counts)} unique entities "
            f"across {len(dataset_entities)} datasets.\n",
        )

    return all_results


# ---------------------------------------------------------------------------
# Dataset display helpers
# ---------------------------------------------------------------------------


def print_dataset_summary(results, title, top_n=50) -> None:
    merged = _merge_results(results)
    entity_counts = merged["entity_counts"]
    dataset_entities = merged.get("dataset_entities", {})

    print(
        f"\n{'=' * 20} {title} {'=' * 20}\n"
        f"  {len(dataset_entities)} datasets · {len(entity_counts)} unique entities"
    )

    if not entity_counts:
        print("  No specific entities found.")
        return

    ranked = entity_counts.most_common()
    top = ranked[:top_n]
    rest = len(ranked) - top_n
    print(f"  Top {min(top_n, len(ranked))} entities by dataset count:")
    max_len = max(len(e) for e, _ in top)
    for entity, count in top:
        bar = "█" * count
        print(f"    {entity:<{max_len}}  {count:>3}  {bar}")
    if rest > 0:
        print(f"    ... and {rest} more")


# ---------------------------------------------------------------------------
# Hierarchy mapping and histograms
# ---------------------------------------------------------------------------


def _build_combined_counter(results, reference_entities=None):
    """Build a single Counter from HuggingFace results and an optional reference list."""
    combined = Counter()
    for data in results.values():
        combined.update(data["entity_counts"])
    if reference_entities:
        for label in reference_entities:
            _add_to_counter(combined, label)
    return combined


def _map_to_hierarchy_buckets(entity_counts):
    """
    Map entity labels to their 2nd- and 3rd-level hierarchy nodes.

    Attempts to canonicalize each entity via EntityHierarchy (exact →
    country-prefix → fuzzy). Entities that cannot be resolved are tracked
    separately.

    Returns:
        depth2_counts  Counter(2nd-level node  -> total count)
        depth3_counts  Counter(3rd-level canonical -> total count)
        unresolved     dict(entity -> count) for labels not in the hierarchy
    """

    h = EntityHierarchy()
    depth2_counts: Counter = Counter()
    depth3_counts: Counter = Counter()
    unresolved: dict[str, int] = {}

    for entity, count in entity_counts.items():
        try:
            branch = h.get_branch(entity)
            # branch: ['PII', '<2nd-level>', '<canonical>']
            if len(branch) >= 2:
                depth2_counts[branch[1]] += count
            depth3_counts[branch[-1]] += count
        except Exception:  # noqa: S112 — entity not in hierarchy; skip gracefully
            unresolved[entity] = count

    return depth2_counts, depth3_counts, unresolved


def print_hierarchy_histograms(
    entity_counts,
    title="HIERARCHY DISTRIBUTION",
    bar_width=40,
) -> None:
    """
    Print two histograms showing how entity labels distribute across the PII taxonomy.

    * **2nd-level** — broad semantic categories (PERSON, LOCATION, GOVERNMENT_ID, …)
    * **3rd-level canonical** — specific canonical entity types (PASSPORT, SSN, …)

    Entities that cannot be resolved by the hierarchy are listed separately.

    Args:
        entity_counts: Counter or dict mapping entity name → occurrence count.
        title:         Section heading.
        bar_width:     Maximum bar width in characters (bars are normalised to the peak).
    """
    depth2_counts, depth3_counts, unresolved = _map_to_hierarchy_buckets(entity_counts)

    print(f"\n{'=' * 20} {title} {'=' * 20}")

    def _print_histogram(counts, heading) -> None:
        ranked = counts.most_common()
        print(f"\n  {heading} ({len(ranked)} buckets):")
        if not ranked:
            print("    (empty)")
            return
        peak = ranked[0][1]
        max_label_len = max(len(e) for e, _ in ranked)
        for entity, count in ranked:
            bar = "█" * (int(count / peak * bar_width) if peak else 0)
            print(f"    {entity:<{max_label_len}}  {count:>4}  {bar}")

    _print_histogram(depth2_counts, "By 2nd-level category")
    _print_histogram(depth3_counts, "By 3rd-level canonical entity")

    if unresolved:
        top = sorted(unresolved.items(), key=lambda x: -x[1])
        print(f"\n  Unresolved ({len(unresolved)} labels not in hierarchy):")
        for entity, count in top[:15]:
            print(f"    {entity}: {count}")
        if len(unresolved) > 15:
            print(f"    ... and {len(unresolved) - 15} more")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    privacy_results = extract_entities_from_query(["pii", "privacy"])

    combined_model_counts = _build_combined_counter(
        privacy_results, THIRD_PARTY_SERVICE_ENTITIES
    )
    print_combined_totals(privacy_results, THIRD_PARTY_SERVICE_ENTITIES)
    print_hierarchy_histograms(
        combined_model_counts, "HIERARCHY DISTRIBUTION (models + reference)"
    )

    dataset_results = extract_entities_from_datasets(["pii", "privacy"])

    combined_dataset_counts = _build_combined_counter(dataset_results)
    print_combined_totals(dataset_results)
    print_hierarchy_histograms(
        combined_dataset_counts, "HIERARCHY DISTRIBUTION (datasets)"
    )

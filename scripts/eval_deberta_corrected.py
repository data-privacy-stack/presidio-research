"""
Corrected evaluation script for DeBERTa-PII with proper 60-entity mappings
"""
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import SpanEvaluator
from presidio_evaluator.models import PresidioAnalyzerWrapper
from presidio_evaluator.experiment_tracking import get_experiment_tracker

def main():
    print("="*80)
    print("EVALUATING: lakshyakh93/deberta_finetuned_pii (CORRECTED MAPPING)")
    print("="*80)

    dataset_name = "synth_dataset_v2.json"
    data_dir = Path(__file__).parent.parent / "data"
    dataset = InputSample.read_dataset_json(data_dir / dataset_name)
    print(f"\nDataset size: {len(dataset)}")

    from presidio_analyzer.nlp_engine import TransformersNlpEngine, NerModelConfiguration
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, Pattern
    from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer

    # CORRECTED MODEL ENTITY MAPPING - 60 entities!
    model_to_presidio_entity_mapping = {
        # Person names
        "FIRSTNAME": "PERSON",
        "MIDDLENAME": "PERSON",
        "LASTNAME": "PERSON",
        "FULLNAME": "PERSON",
        "NAME": "PERSON",
        "DISPLAYNAME": "PERSON",
        "USERNAME": "PERSON",

        # Titles/Prefixes
        "PREFIX": "TITLE",
        "SUFFIX": "TITLE",

        # Organization
        "COMPANY_NAME": "ORGANIZATION",
        "JOBDESCRIPTOR": "ORGANIZATION",
        "JOBTITLE": "ORGANIZATION",
        "JOBAREA": "ORGANIZATION",

        # Location/Address
        "STREETADDRESS": "STREET_ADDRESS",
        "STREET": "STREET_ADDRESS",
        "CITY": "GPE",
        "STATE": "GPE",
        "COUNTY": "GPE",
        "ZIPCODE": "ZIP_CODE",
        "BUILDINGNUMBER": "STREET_ADDRESS",
        "SECONDARYADDRESS": "STREET_ADDRESS",

        # Contact
        "EMAIL": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "PHONEIMEI": "PHONE_NUMBER",

        # Internet
        "URL": "DOMAIN_NAME",
        "IP": "IP_ADDRESS",
        "IPV4": "IP_ADDRESS",
        "IPV6": "IP_ADDRESS",

        # Financial
        "CREDITCARDNUMBER": "CREDIT_CARD",
        "IBAN": "IBAN_CODE",
        "BIC": "IBAN_CODE",
        "ACCOUNTNUMBER": "IBAN_CODE",
        "SSN": "US_SSN",

        # Date/Time
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",

        # Ignore others
        "BITCOINADDRESS": "O",
        "ETHEREUMADDRESS": "O",
        "LITECOINADDRESS": "O",
        "VEHICLEVIN": "O",
        "VEHICLEVRM": "O",
        "PASSWORD": "O",
        "PIN": "O",
        "MAC": "O",
        "USERAGENT": "O",
        "GENDER": "O",
        "SEX": "O",
        "SEXTYPE": "O",
        "AMOUNT": "O",
        "CURRENCY": "O",
        "CURRENCYCODE": "O",
        "CURRENCYNAME": "O",
        "CURRENCYSYMBOL": "O",
        "NUMBER": "O",
        "MASKEDNUMBER": "O",
        "NEARBYGPSCOORDINATE": "O",
        "ORDINALDIRECTION": "O",
        "JOBTYPE": "O",
        "ACCOUNTNAME": "O",
        "CREDITCARDCVV": "O",
        "CREDITCARDISSUER": "O",
    }

    model_config = [{"lang_code": "en", "model_name": {
        "spacy": "en_core_web_sm",
        "transformers": "lakshyakh93/deberta_finetuned_pii"
    }}]

    ner_model_configuration = NerModelConfiguration(
        labels_to_ignore=["O"],
        model_to_presidio_entity_mapping=model_to_presidio_entity_mapping
    )

    print("\nLoading model...")
    nlp_engine = TransformersNlpEngine(
        models=model_config, ner_model_configuration=ner_model_configuration
    )
    nlp_engine.load()
    print("Model loaded!")

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    unnecessary = ['NhsRecognizer', 'UkNinoRecognizer', 'SgFinRecognizer',
                  'AuAbnRecognizer', 'AuAcnRecognizer','AuTfnRecognizer',
                  'AuMedicareRecognizer', 'InPanRecognizer',
                  'InAadhaarRecognizer', 'InVehicleRegistrationRecognizer',
                  'InPassportRecognizer', 'InVoterRecognizer']
    [registry.remove_recognizer(rec) for rec in unnecessary]

    context_enhancer = LemmaContextAwareEnhancer(context_prefix_count=10, context_suffix_count=10)
    analyzer_engine = AnalyzerEngine(
        nlp_engine=nlp_engine, context_aware_enhancer=context_enhancer,
        registry=registry, default_score_threshold=0.3
    )

    # Dataset entity alignment
    # Supported entities → mapped, Unsupported → "O" (filtered out)
    entities_mapping = {
        # Supported by model
        "PERSON": "PERSON",
        "FIRSTNAME": "PERSON",
        "LASTNAME": "PERSON",
        "MIDDLENAME": "PERSON",
        "FULLNAME": "PERSON",
        "NAME": "PERSON",
        "USERNAME": "PERSON",
        "ORGANIZATION": "ORGANIZATION",
        "COMPANY_NAME": "ORGANIZATION",
        "GPE": "GPE",
        "CITY": "GPE",
        "STATE": "GPE",
        "COUNTY": "GPE",
        "STREET_ADDRESS": "STREET_ADDRESS",
        "STREETADDRESS": "STREET_ADDRESS",
        "STREET": "STREET_ADDRESS",
        "ZIP_CODE": "ZIP_CODE",
        "ZIPCODE": "ZIP_CODE",
        "TITLE": "TITLE",
        "PREFIX": "TITLE",
        "SUFFIX": "TITLE",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "EMAIL": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "CREDIT_CARD": "CREDIT_CARD",
        "CREDITCARDNUMBER": "CREDIT_CARD",
        "US_SSN": "US_SSN",
        "SSN": "US_SSN",
        "IBAN_CODE": "IBAN_CODE",
        "IBAN": "IBAN_CODE",
        "IP_ADDRESS": "IP_ADDRESS",
        "IP": "IP_ADDRESS",
        "IPV4": "IP_ADDRESS",
        "IPV6": "IP_ADDRESS",
        "DATE_TIME": "DATE_TIME",
        "DATE": "DATE_TIME",
        "TIME": "DATE_TIME",
        "DOMAIN_NAME": "DOMAIN_NAME",
        "URL": "DOMAIN_NAME",
        "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
        # Unsupported - filtered out
        "AGE": "O",
        "NRP": "O",
        "O": "O"
    }
    
    dataset = SpanEvaluator.align_entity_types(
        dataset, entities_mapping=entities_mapping, allow_missing_mappings=False
    )

    experiment = get_experiment_tracker()
    evaluator = SpanEvaluator(model=analyzer_engine, iou_threshold=0.7)

    params = {
        "dataset_name": dataset_name,
        "model_name": "lakshyakh93/deberta_finetuned_pii",
        "model_size": "750MB",
        "entity_mapping": "CORRECTED_60_ENTITIES",
        "model_entities": "60 entities including FIRSTNAME, LASTNAME, COMPANY_NAME, STREETADDRESS, etc."
    }
    params.update(evaluator.model.to_log())
    experiment.log_parameters(params)
    experiment.log_dataset_hash(dataset)
    experiment.log_parameter("entity_mappings", json.dumps(model_to_presidio_entity_mapping))

    print("\nRunning evaluation...")
    evaluation_results = evaluator.evaluate_all(dataset)
    results = evaluator.calculate_score(evaluation_results)

    experiment.log_metrics(results.to_log())
    entities, confmatrix = results.to_confusion_matrix()
    experiment.log_confusion_matrix(matrix=confmatrix, labels=entities)
    experiment.end()

    print(f"\n{'='*80}")
    print("RESULTS - DeBERTa-PII (CORRECTED)")
    print(f"{'='*80}")
    print(f"F-Score:   {results.pii_f:.4f}")
    print(f"Precision: {results.pii_precision:.4f}")
    print(f"Recall:    {results.pii_recall:.4f}")
    print(f"{'='*80}")

    results_dict = {
        "model": "lakshyakh93/deberta_finetuned_pii",
        "pii_f": results.pii_f,
        "pii_precision": results.pii_precision,
        "pii_recall": results.pii_recall,
        "per_entity": {}
    }

    metrics_dict = results.to_log()
    for key, value in metrics_dict.items():
        if '_' in key and key.split('_')[0] not in ['pii', 'n']:
            results_dict["per_entity"][key] = value

    output_file = data_dir / "deberta_pii_corrected_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    main()


"""
Corrected evaluation script for RoBERTa-i2b2 with proper entity mappings including BIOLU tags
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
    print("EVALUATING: obi/deid_roberta_i2b2 (CORRECTED MAPPING)")
    print("="*80)

    dataset_name = "synth_dataset_v2.json"
    data_dir = Path(__file__).parent.parent / "data"
    dataset = InputSample.read_dataset_json(data_dir / dataset_name)
    print(f"\nDataset size: {len(dataset)}")

    from presidio_analyzer.nlp_engine import TransformersNlpEngine, NerModelConfiguration
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, Pattern
    from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer

    # CORRECTED MODEL ENTITY MAPPING
    # Model uses BIOLU tagging: B-, I-, L-, U- prefixes
    # Base entities: AGE, DATE, EMAIL, HOSP, ID, LOC, OTHERPHI, PATIENT, PATORG, PHONE, STAFF
    model_to_presidio_entity_mapping = {
        # Person entities (all variants)
        "PATIENT": "PERSON",
        "STAFF": "PERSON",
        "L-PATIENT": "PERSON",
        "U-PATIENT": "PERSON",
        "L-STAFF": "PERSON",
        "U-STAFF": "PERSON",

        # Organization entities
        "HOSP": "ORGANIZATION",
        "PATORG": "ORGANIZATION",
        "L-HOSP": "ORGANIZATION",
        "U-HOSP": "ORGANIZATION",
        "L-PATORG": "ORGANIZATION",
        "U-PATORG": "ORGANIZATION",

        # Location entities
        "LOC": "LOCATION",
        "L-LOC": "LOCATION",
        "U-LOC": "LOCATION",

        # Other entities
        "AGE": "AGE",
        "L-AGE": "AGE",
        "U-AGE": "AGE",
        "DATE": "DATE_TIME",
        "L-DATE": "DATE_TIME",
        "U-DATE": "DATE_TIME",
        "EMAIL": "EMAIL_ADDRESS",
        "L-EMAIL": "EMAIL_ADDRESS",
        "U-EMAIL": "EMAIL_ADDRESS",
        "PHONE": "PHONE_NUMBER",
        "L-PHONE": "PHONE_NUMBER",
        "U-PHONE": "PHONE_NUMBER",
        "ID": "US_DRIVER_LICENSE",
        "L-ID": "US_DRIVER_LICENSE",
        "U-ID": "US_DRIVER_LICENSE",

        # Ignore
        "OTHERPHI": "O",
        "L-OTHERPHI": "O",
        "U-OTHERPHI": "O",
    }

    model_config = [{"lang_code": "en", "model_name": {
        "spacy": "en_core_web_sm",
        "transformers": "obi/deid_roberta_i2b2"
    }}]

    ner_model_configuration = NerModelConfiguration(
        labels_to_ignore=["O", "OTHERPHI"],
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
        "PATIENT": "PERSON",
        "STAFF": "PERSON",
        "ORGANIZATION": "ORGANIZATION",
        "HOSP": "ORGANIZATION",
        "PATORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "STREET_ADDRESS": "LOCATION",
        "LOCATION": "LOCATION",
        "LOC": "LOCATION",
        "AGE": "AGE",
        "DATE_TIME": "DATE_TIME",
        "DATE": "DATE_TIME",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "EMAIL": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "PHONE": "PHONE_NUMBER",
        "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
        "US_SSN": "US_SSN",
        "ID": "US_DRIVER_LICENSE",
        "CREDIT_CARD": "CREDIT_CARD",
        "IBAN_CODE": "IBAN_CODE",
        "IP_ADDRESS": "IP_ADDRESS",
        "DOMAIN_NAME": "URL",
        "URL": "URL",
        # Unsupported - filtered out
        "NRP": "O",
        "TITLE": "O",
        "ZIP_CODE": "O",
        "O": "O"
    }
    
    dataset = SpanEvaluator.align_entity_types(
        dataset, entities_mapping=entities_mapping, allow_missing_mappings=False
    )

    experiment = get_experiment_tracker()
    evaluator = SpanEvaluator(model=analyzer_engine, iou_threshold=0.7)

    params = {
        "dataset_name": dataset_name,
        "model_name": "obi/deid_roberta_i2b2",
        "model_size": "480MB",
        "entity_mapping": "CORRECTED_WITH_BIOLU",
        "model_entities": "AGE,DATE,EMAIL,HOSP,ID,LOC,PATIENT,PATORG,PHONE,STAFF (with BIOLU prefixes)"
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
    print("RESULTS - RoBERTa-i2b2 (CORRECTED)")
    print(f"{'='*80}")
    print(f"F-Score:   {results.pii_f:.4f}")
    print(f"Precision: {results.pii_precision:.4f}")
    print(f"Recall:    {results.pii_recall:.4f}")
    print(f"{'='*80}")

    results_dict = {
        "model": "obi/deid_roberta_i2b2",
        "pii_f": results.pii_f,
        "pii_precision": results.pii_precision,
        "pii_recall": results.pii_recall,
        "per_entity": {}
    }

    metrics_dict = results.to_log()
    for key, value in metrics_dict.items():
        if '_' in key and key.split('_')[0] not in ['pii', 'n']:
            results_dict["per_entity"][key] = value

    output_file = data_dir / "roberta_i2b2_corrected_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    main()


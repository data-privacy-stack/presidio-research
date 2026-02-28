from pathlib import Path
from pprint import pprint
from collections import Counter
from typing import Dict, List
import json

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import Evaluator, ModelError, Plotter
from presidio_evaluator.experiment_tracking import get_experiment_tracker
from presidio_evaluator.models import PresidioAnalyzerWrapper


def test_notebook():
    test_dir = Path(__file__).parent  # Get the directory of this test file
    data_path = test_dir.parent / "data" / "generated_small.json"
    dataset = InputSample.read_dataset_json(data_path)
    print(len(dataset))

    def get_entity_counts(dataset: List[InputSample]) -> Dict:
        """Return a dictionary with counter per entity type."""
        entity_counter = Counter()
        for sample in dataset:
            for tag in sample.tags:
                entity_counter[tag] += 1
        return entity_counter

    entity_counts = get_entity_counts(dataset)
    print("Count per entity:")
    pprint(entity_counts.most_common(), compact=True)

    print(
        "\nMin and max number of tokens in dataset: "
        f"Min: {min([len(sample.tokens) for sample in dataset])}, "
        f"Max: {max([len(sample.tokens) for sample in dataset])}"
    )

    print(
        f"Min and max sentence length in dataset: "
        f"Min: {min([len(sample.full_text) for sample in dataset])}, "
        f"Max: {max([len(sample.full_text) for sample in dataset])}"
    )

    print("\nExample InputSample:")
    print(dataset[0])

    print("A few examples sentences containing each entity:\n")
    for entity in entity_counts.keys():
        samples = [sample for sample in dataset if entity in set(sample.tags)]
        if len(samples) > 1 and entity != "O":
            print(
                f"Entity: <{entity}> two example sentences:\n"
                f"\n1) {samples[0].full_text}"
                f"\n2) {samples[1].full_text}"
                f"\n------------------------------------\n"
            )

    from presidio_analyzer import AnalyzerEngine

    # Loading the vanilla Analyzer Engine, with the default NER model.
    analyzer_engine = AnalyzerEngine(default_score_threshold=0.4)

    pprint("Supported entities for English:")
    pprint(analyzer_engine.get_supported_entities("en"), compact=True)

    print("\nLoaded recognizers for English:")
    pprint(
        [
            rec.name
            for rec in analyzer_engine.registry.get_recognizers("en", all_fields=True)
        ],
        compact=True,
    )

    print("\nLoaded NER models:")
    pprint(analyzer_engine.nlp_engine.models)

    entities_mapping = PresidioAnalyzerWrapper.presidio_entities_map  # default mapping

    print("Using this mapping between the dataset and Presidio's entities:")
    pprint(entities_mapping, compact=True)

    # ## 5. Set up the Evaluator object

    # In[8]:

    # Set up the experiment tracker to log the experiment for reproducibility
    experiment = get_experiment_tracker()

    # Create the evaluator object with entity mapping
    evaluator = Evaluator(model=analyzer_engine, entity_mapping=entities_mapping)

    evaluation_results = evaluator.evaluate_all(dataset)
    results = evaluator.calculate_score(evaluation_results)

    results_df = evaluator.get_results_dataframe(evaluation_results)
    print(results_df)

    experiment.log_metrics(results.to_log())
    entities, confmatrix = results.to_confusion_matrix()
    experiment.log_confusion_matrix(matrix=confmatrix, labels=entities)

    # Track model and dataset params
    params = {"dataset_name": data_path, "model_name": evaluator.model.name}
    params.update(evaluator.model.to_log())
    experiment.log_parameters(params)
    experiment.log_dataset_hash(dataset)
    experiment.log_parameter("entity_mappings", json.dumps(entities_mapping))

    # Plot output
    # plotter = Plotter(
    #     results=results, model_name=evaluator.model.name, beta=2
    # )

    # plotter.plot_scores()

    # In[12]:

    pprint(
        {
            "PII F": results.pii_f,
            "PII recall": results.pii_recall,
            "PII precision": results.pii_precision,
        }
    )

    # plotter.plot_confusion_matrix(entities=entities, confmatrix=confmatrix )

    # In[14]:

    # plotter.plot_most_common_tokens()

    # ### 7a. False positives
    # #### Most common false positive tokens:

    # In[15]:

    ModelError.most_common_fp_tokens(results.model_errors)

    # #### More FP analysis

    # In[16]:

    fps_df = ModelError.get_fps_dataframe(results.model_errors, entity=["PERSON"])
    fps_df[["full_text", "token", "annotation", "prediction"]].head(20)

    # ### 7b. False negatives (FN)
    #
    # #### Most common false negative examples + a few samples with FN

    # In[17]:

    ModelError.most_common_fn_tokens(results.model_errors, n=15)

    # #### More FN analysis

    # In[18]:

    fns_df = ModelError.get_fns_dataframe(results.model_errors, entity=["PHONE_NUMBER"])

    # In[19]:

    fns_df[["full_text", "token", "annotation", "prediction"]].head(20)

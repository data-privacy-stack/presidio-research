import json
from collections import Counter
from pathlib import Path
from pprint import pprint

import pytest
from presidio_analyzer import AnalyzerEngine

from presidio_evaluator import InputSample
from presidio_evaluator.entity_mapping import CanonicalMapper, IssueType
from presidio_evaluator.evaluation import ModelError, Plotter, SpanEvaluator
from presidio_evaluator.evaluation.token_evaluator import TokenEvaluator
from presidio_evaluator.experiment_tracking import get_experiment_tracker
from presidio_evaluator.models import PresidioAnalyzerWrapper


@pytest.fixture(scope="module")
def dataset() -> list[InputSample]:
    data_path = Path(__file__).parent.parent / "data" / "generated_large.json"
    return InputSample.read_dataset_json(data_path)


@pytest.fixture(scope="module")
def analyzer_engine() -> AnalyzerEngine:
    return AnalyzerEngine(default_score_threshold=0.4)


@pytest.mark.integration
def test_notebook(dataset: list[InputSample], analyzer_engine: AnalyzerEngine):
    """Mirrors the exact flow of notebooks/4_Evaluate_Presidio_Analyzer.ipynb."""
    # --- 1. Load dataset ---
    dataset_name = "generated_large.json"
    dataset = dataset[:150]
    assert len(dataset) > 0

    # --- 2. Dataset statistics ---
    def get_entity_counts(dataset: list[InputSample]) -> Counter:
        entity_counter = Counter()
        for sample in dataset:
            for tag in sample.tags:
                entity_counter[tag] += 1
        return entity_counter

    entity_counts = get_entity_counts(dataset)
    pprint(entity_counts.most_common(), compact=True)
    for entity in entity_counts.keys():
        samples = [sample for sample in dataset if entity in set(sample.tags)]
        if len(samples) > 1 and entity != "O":
            print(f"Entity: <{entity}> example: {samples[0].full_text}")

    # --- 3. Define the Presidio Analyzer ---
    pprint(analyzer_engine.get_supported_entities("en"), compact=True)

    # --- 4. Wrap analyzer + experiment tracking ---
    wrapped_analyzer = PresidioAnalyzerWrapper(analyzer_engine=analyzer_engine)
    experiment = get_experiment_tracker()
    params = {"dataset_name": dataset_name, "model_name": wrapped_analyzer.name}
    params.update(wrapped_analyzer.to_log())
    experiment.log_parameters(params)
    experiment.log_dataset_hash(dataset)

    # --- 5. Predict ---
    results_df = wrapped_analyzer.predict_dataset(dataset)
    assert list(results_df.columns) == [
        "sentence_id",
        "token",
        "annotation",
        "prediction",
        "start_indices",
    ]

    # --- 6. Map entities ---
    # Presidio outputs depth-2 labels; resolve to depth-3 before extraction.
    # Unknown/prediction-only labels are suppressed for testing purposes.
    _PRESIDIO_LABEL_MAP = {
        "PERSON": "NAME",
        "LOCATION": "LOC",
        "ORGANIZATION": "ORG",
        "DATE_TIME": "DATE",
    }

    def _resolve_blocking(m: CanonicalMapper) -> None:
        """Resolve all blocking issues: remap known depth-2, suppress unknowns."""
        resolutions = {}
        for issue in m.get_issues():
            if issue.type in (IssueType.COLLISION_CROSS_BRANCH,):
                for lbl in issue.labels:
                    resolutions[lbl] = _PRESIDIO_LABEL_MAP.get(lbl)
            elif issue.type in (IssueType.UNRESOLVED, IssueType.PREDICTION_ONLY):
                for lbl in issue.labels:
                    resolutions[lbl] = None
        if resolutions:
            m.map(resolutions)

    mapper = CanonicalMapper()
    mapper.analyze(results_df)
    _resolve_blocking(mapper)
    mapped_results = mapper.get_mapped_results_dataframe()

    # Apply overrides (mirrors notebook cell 20)
    mapper.map({"ORGANIZATION": None})
    mapper.analyze(results_df)
    _resolve_blocking(mapper)
    mapped_results = mapper.get_mapped_results_dataframe()
    assert mapped_results.detailed.shape[0] == results_df.shape[0]  # same row count

    experiment.log_parameter("entity_mappings", json.dumps(mapper.get_mapping()))

    # --- 7. Evaluate ---
    evaluator = SpanEvaluator(iou_threshold=0.75)
    results = evaluator.calculate_score_on_df(results_df=mapped_results.detailed)

    assert results.pii_precision is not None
    assert results.pii_recall is not None

    experiment.log_metrics(results.to_log())
    entities, confmatrix = results.to_confusion_matrix()
    experiment.log_confusion_matrix(matrix=confmatrix, labels=entities)
    experiment.end()

    pprint(
        {
            "PII F": results.pii_f,
            "PII recall": results.pii_recall,
            "PII precision": results.pii_precision,
        }
    )

    # --- 8. Plot (display_mode="none" avoids kaleido dependency in tests) ---
    plotter = Plotter(
        results=results, model_name=wrapped_analyzer.name, display_mode="none", beta=2
    )
    plotter.plot_scores()
    plotter.plot_confusion_matrix(entities=entities, confmatrix=confmatrix)
    plotter.plot_most_common_tokens()

    # --- 9. Error analysis ---
    ModelError.most_common_fp_tokens(results.model_errors)
    fps_df = ModelError.get_fps_dataframe(results.model_errors, entity="PERSON")
    if fps_df is not None:
        fps_df[["full_text", "token", "annotation", "prediction"]].head(20)

    ModelError.most_common_fn_tokens(results.model_errors, n=15)
    fns_df = ModelError.get_fns_dataframe(results.model_errors, entity="PHONE_NUMBER")
    if fns_df is not None:
        fns_df[["full_text", "token", "annotation", "prediction"]].head(20)


@pytest.mark.integration
def test_full_pipeline_integration(
    dataset: list[InputSample], analyzer_engine: AnalyzerEngine
):
    """
    Integration test for the full 3-step pipeline:
        1. model.predict_dataset(dataset)          -> results_df
        2. mapper.get_mapped_results_dataframe()   -> mapped_df
        3. evaluator.calculate_score_on_df()       -> EvaluationResult
        4. Plotter.plot_scores()                   -> no exception

    Covers: SpanEvaluator path and TokenEvaluator path.
    """
    dataset = dataset[:200]  # small slice
    model = PresidioAnalyzerWrapper(
        analyzer_engine=analyzer_engine, entities_to_keep=["PERSON"]
    )

    # Step 1: predict
    results_df = model.predict_dataset(dataset)
    assert list(results_df.columns) == [
        "sentence_id",
        "token",
        "annotation",
        "prediction",
        "start_indices",
    ]

    # Step 2: map
    # Presidio outputs depth-2 labels (PERSON etc.); resolve to depth-3 first.
    # Unknown/prediction-only labels are suppressed for testing purposes.
    _PRESIDIO_LABEL_MAP = {
        "PERSON": "NAME",
        "LOCATION": "LOC",
        "ORGANIZATION": "ORG",
        "DATE_TIME": "DATE",
    }
    mapper = CanonicalMapper()
    mapper.analyze(results_df)
    resolutions = {}
    for issue in mapper.get_issues():
        if issue.type == IssueType.COLLISION_CROSS_BRANCH:
            for lbl in issue.labels:
                resolutions[lbl] = _PRESIDIO_LABEL_MAP.get(lbl)
        elif issue.type in (IssueType.UNRESOLVED, IssueType.PREDICTION_ONLY):
            for lbl in issue.labels:
                resolutions[lbl] = None
    if resolutions:
        mapper.map(resolutions)
    mapped_results = mapper.get_mapped_results_dataframe()
    assert mapped_results.detailed.shape[0] == results_df.shape[0]  # same row count
    span_evaluator = SpanEvaluator(skip_words=[])
    token_evaluator = TokenEvaluator(skip_words=[])
    evaluators = [span_evaluator, token_evaluator]

    # PERSON maps to NAME after projection
    for evaluator in evaluators:
        result_global = evaluator.calculate_score_on_df(
            results_df=mapped_results.detailed, level="both"
        )
        assert result_global.pii_recall >= 0.1
        assert result_global.pii_precision >= 0.1
        assert result_global.entity_precision_dict is not None
        assert result_global.entity_recall_dict is not None
        per_type_key = next(
            (k for k in result_global.per_type if k in ("NAME", "PERSON")), None
        )
        if per_type_key:
            assert result_global.per_type[per_type_key].precision >= 0.1
            assert result_global.per_type[per_type_key].recall >= 0.1
        assert result_global.model_errors is not None
        assert len(result_global.model_errors) > 0

        # Step 4: Plotter — must not raise
        plotter = Plotter(results=result_global, display_mode="none")
        plotter.plot_scores(output_folder=None)

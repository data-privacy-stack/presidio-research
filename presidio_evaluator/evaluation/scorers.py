"""E2E scoring pipelines for the different models"""

import warnings

from presidio_analyzer import EntityRecognizer

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import EvaluationResult
from presidio_evaluator.models import (
    BaseModel,
)


def score_model(
    model: BaseModel,
    entities_to_keep: list[str],
    input_samples: list[InputSample],
    verbose: bool = False,
    beta: float = 2.5,
) -> EvaluationResult:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.

    Please use the new 5-step pipeline instead:
        results_df = model.predict_dataset(dataset)
        mapper = CanonicalMapper()
        results_df_mapped = mapper.get_mapped_results_dataframe(results_df)
        evaluator = SpanEvaluator()
        result = evaluator.calculate_score_on_df(results_df_mapped)

    See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.
    """
    warnings.warn(
        "score_model() is deprecated and will be removed in a future version. "
        "Please use model.predict_dataset() + CanonicalMapper + evaluator.calculate_score_on_df(). "
        "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.",
        DeprecationWarning,
        stacklevel=2,
    )

    raise NotImplementedError(
        "score_model() is no longer functional. "
        "Please use the new pipeline: model.predict_dataset() -> CanonicalMapper.get_mapped_results_dataframe() "
        "-> evaluator.calculate_score_on_df(). "
        "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.",
    )


def score_presidio_recognizer(
    recognizer: EntityRecognizer,
    entities_to_keep: list[str],
    input_samples: list[InputSample] | None = None,
    labeling_scheme: str = "BILUO",
    with_nlp_artifacts: bool = False,
    verbose: bool = False,
) -> EvaluationResult:
    """
    DEPRECATED: This function is deprecated.

    Please use the new 5-step pipeline instead:
        model = PresidioRecognizerWrapper(recognizer, ...)
        results_df = model.predict_dataset(dataset)
        mapper = CanonicalMapper()
        results_df_mapped = mapper.get_mapped_results_dataframe(results_df)
        evaluator = SpanEvaluator()
        result = evaluator.calculate_score_on_df(results_df_mapped)

    See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.
    """
    warnings.warn(
        "score_presidio_recognizer() is deprecated. "
        "Please use model.predict_dataset() + CanonicalMapper + evaluator.calculate_score_on_df(). "
        "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.",
        DeprecationWarning,
        stacklevel=2,
    )

    if not input_samples:
        print("Reading dataset")
        input_samples = InputSample.read_dataset_json(
            "../../data/synth_dataset_v2.json",
        )
    else:
        input_samples = list(input_samples)

    raise NotImplementedError(
        "score_presidio_recognizer() is no longer functional. "
        "Please use the new pipeline: model.predict_dataset() -> CanonicalMapper.get_mapped_results_dataframe() "
        "-> evaluator.calculate_score_on_df(). "
        "See notebooks/4_Evaluate_Presidio_Analyzer.ipynb for a full example.",
    )

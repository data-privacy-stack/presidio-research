"""E2E scoring pipelines for the different models"""

import warnings
from typing import List, Optional

from presidio_analyzer import EntityRecognizer
from presidio_analyzer.nlp_engine import SpacyNlpEngine

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import EvaluationResult, Evaluator
from presidio_evaluator.models import (
    PresidioRecognizerWrapper,
    PresidioAnalyzerWrapper,
    BaseModel,
)


def score_model(
    model: BaseModel,
    entities_to_keep: List[str],
    input_samples: List[InputSample],
    verbose: bool = False,
    beta: float = 2.5,
) -> EvaluationResult:
    """
    DEPRECATED: This function is deprecated and will be removed in a future version.
    
    Please use the evaluation patterns shown in the notebooks instead:
    - See notebooks/4_Evaluate_Presidio_Analyzer.ipynb
    - See notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb
    
    The new approach requires passing entity_mapping to the evaluator:
        evaluator = Evaluator(
            model=model,
            entity_mapping={'DATASET_ENTITY': 'MODEL_ENTITY'},
            entities_to_keep=entities_to_keep
        )
    
    Run data through a model and gather results and stats
    """
    warnings.warn(
        "score_model() is deprecated and will be removed in a future version. " 
        "Please use the evaluation patterns shown in notebooks/4_Evaluate_Presidio_Analyzer.ipynb. "
        "The new approach requires passing entity_mapping to the evaluator.",
        DeprecationWarning,
        stacklevel=2
    )
    
    raise NotImplementedError(
        "score_model() is no longer functional because it does not accept entity_mapping parameter. "
        "The evaluator now requires entity_mapping to be provided. "
        "Please use the evaluation pattern shown in notebooks/4_Evaluate_Presidio_Analyzer.ipynb: \\n"
        "  evaluator = Evaluator(model=model, entity_mapping={...}, entities_to_keep=entities_to_keep)\\n"
        "  results = evaluator.evaluate_all(dataset)"
    )


def score_presidio_recognizer(
    recognizer: EntityRecognizer,
    entities_to_keep: List[str],
    input_samples: Optional[List[InputSample]] = None,
    labeling_scheme: str = "BILUO",
    with_nlp_artifacts: bool = False,
    verbose: bool = False,
) -> EvaluationResult:
    """
    DEPRECATED: This function is deprecated.
    
    Please use the evaluation patterns shown in the notebooks instead:
    - See notebooks/4_Evaluate_Presidio_Analyzer.ipynb
    - See notebooks/5_Evaluate_Custom_Presidio_Analyzer.ipynb
    
    The new approach requires passing entity_mapping to the evaluator:
        model = PresidioRecognizerWrapper(recognizer, ...)
        evaluator = Evaluator(
            model=model,
            entity_mapping={'DATASET_ENTITY': 'MODEL_ENTITY'},
            entities_to_keep=entities_to_keep
        )
        results = evaluator.evaluate_all(dataset)
    
    Run data through one EntityRecognizer and gather results and stats
    """
    warnings.warn(
        "score_presidio_recognizer() is deprecated. "
        "Please use the evaluation patterns shown in notebooks/4_Evaluate_Presidio_Analyzer.ipynb. "
        "The new approach requires passing entity_mapping to the evaluator.",
        DeprecationWarning,
        stacklevel=2
    )

    if not input_samples:
        print("Reading dataset")
        input_samples = InputSample.read_dataset_json(
            "../../data/synth_dataset_v2.json"
        )
    else:
        input_samples = list(input_samples)

    # NOTE: Entity alignment removed - entity_mapping should be passed to evaluator
    # For proper usage, see notebooks/4_Evaluate_Presidio_Analyzer.ipynb
    raise NotImplementedError(
        "score_presidio_recognizer() requires entity_mapping to be passed to the evaluator. "
        "This function can no longer automatically align entity types. "
        "Please use the evaluation pattern shown in notebooks/4_Evaluate_Presidio_Analyzer.ipynb instead."
    )

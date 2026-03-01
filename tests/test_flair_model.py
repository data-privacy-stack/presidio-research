import sys

import pytest

from presidio_evaluator.evaluation import Evaluator
from tests.conftest import assert_model_results_gt
from presidio_evaluator.models.flair_model import FlairModel


@pytest.mark.slow
@pytest.mark.skipif("flair" not in sys.modules, reason="requires the Flair library")
def test_flair_simple(small_dataset):

    flair_model = FlairModel(model_path="ner", entities_to_keep=["PERSON"])
    evaluator = Evaluator(model=flair_model, entity_mapping = {})
    evaluation_results = evaluator.evaluate_all(small_dataset)
    scores = evaluator.calculate_score(evaluation_results)

    assert_model_results_gt(scores, "PERSON", 0)

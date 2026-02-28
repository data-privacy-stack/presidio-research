import os
from typing import List

import numpy as np
import pytest

# pytest configuration file
# the configuration allow 3 kind of tests:
#   * unmarked tests run on all pytest execution
#   * tests with large datasets\long testing time are marked as "slow" and have to be run with pytest run --runslow
#   * tests with inconclusive result are marked as "inconclusive" have to be run with pytest run --runinconclusive
#   * tests can be both slow and inconclusive and have to be run with pytest run --runslow --runinconclusive
from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import EvaluationResult


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runinconclusive",
        action="store_true",
        default=False,
        help="run inconclusive tests",
    )


def pytest_collection_modifyitems(items, config):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--runinconclusive"):
        skip_inconclusive = pytest.mark.skip(
            reason="need --runinconclusive option to run"
        )
        for item in items:
            if "inconclusive" in item.keywords:
                item.add_marker(skip_inconclusive)


@pytest.fixture(scope="session")
def small_dataset() -> List[InputSample]:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(
        os.path.join(dir_path, "data/generated_small.json")
    )
    return input_samples


@pytest.fixture
def default_entity_mapping():
    """Default identity entity mapping for tests (entities map to themselves)."""
    return {
        "PERSON": "PERSON",
        "GPE": "GPE",
        "ORG": "ORG",
        "LOCATION": "LOCATION",
        "EMAIL": "EMAIL",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "CREDIT_CARD": "CREDIT_CARD",
        "IP_ADDRESS": "IP_ADDRESS",
        "US_SSN": "US_SSN",
        "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
        "DATE_TIME": "DATE_TIME",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
    }


def assert_model_results_gt(
    scores: EvaluationResult, entity: str = "PERSON", threshold: float = 0.1
):
    np.testing.assert_almost_equal(
        scores.pii_precision, scores.entity_precision_dict[entity]
    )
    np.testing.assert_almost_equal(scores.pii_recall, scores.entity_recall_dict[entity])
    assert scores.pii_recall > threshold
    assert scores.pii_precision > threshold

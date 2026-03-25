import os

import pytest
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import CreditCardRecognizer

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import TokenEvaluator
from presidio_evaluator.models import PresidioRecognizerWrapper


# test case parameters for tests with dataset which was previously generated.
class GeneratedTextTestCase:
    def __init__(self, test_name, test_input, acceptance_threshold, marks):
        self.test_name = test_name
        self.test_input = test_input
        self.acceptance_threshold = acceptance_threshold
        self.marks = marks

    def to_pytest_param(self):
        return pytest.param(
            self.test_input,
            self.acceptance_threshold,
            id=self.test_name,
            marks=self.marks,
        )


# generated-text test cases
cc_test_generate_text_testdata = [
    # small set fixture which expects all type results.
    GeneratedTextTestCase(
        test_name="small-set",
        test_input="{}/data/generated_small.json",
        acceptance_threshold=1,
        marks=pytest.mark.none,
    ),
    # large set fixture which expects all type results. marked as "slow"
    GeneratedTextTestCase(
        test_name="large_set",
        test_input="{}/data/generated_large.json",
        acceptance_threshold=1,
        marks=pytest.mark.slow,
    ),
]


# credit card recognizer tests on generated data
@pytest.mark.parametrize(
    "test_input,acceptance_threshold",
    [testcase.to_pytest_param() for testcase in cc_test_generate_text_testdata],
)
def test_credit_card_recognizer_with_generated_text(test_input, acceptance_threshold):
    """
    Test credit card recognizer with a generated dataset text file
    :param test_input: input text file location
    :param acceptance_threshold: minimum precision/recall
     allowed for tests to pass
    """

    # read test input from generated file
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    input_samples = InputSample.read_dataset_json(test_input.format(dir_path))

    nlp_engine = SpacyNlpEngine()
    model = PresidioRecognizerWrapper(
        recognizer=CreditCardRecognizer(),
        nlp_engine=nlp_engine,
        entities_to_keep=["CREDIT_CARD"],
    )
    results_df = model.predict_dataset(input_samples)
    evaluator = TokenEvaluator(entities_to_keep=model.entities)
    scores = evaluator.calculate_score_on_df(results_df)

    assert acceptance_threshold <= scores.pii_f

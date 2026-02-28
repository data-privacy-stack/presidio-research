import pytest
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import Evaluator
from presidio_evaluator.models import PresidioRecognizerWrapper


class GeneratedTextTestCase:
    """
    Test case parameters for tests with dataset which was previously generated.
    """

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
    # small dataset, inconclusive results
    GeneratedTextTestCase(
        test_name="small-set",
        test_input="{}/data/generated_small.json",
        acceptance_threshold=0.5,
        marks=pytest.mark.none,
    ),
    # large dataset - test is slow and inconclusive
    GeneratedTextTestCase(
        test_name="large-set",
        test_input="{}/data/generated_large.json",
        acceptance_threshold=0.5,
        marks=pytest.mark.slow,
    ),
]


# credit card recognizer tests on generated data
@pytest.mark.parametrize(
    "test_input,acceptance_threshold",
    [testcase.to_pytest_param() for testcase in cc_test_generate_text_testdata],
)
def test_spacy_recognizer_with_generated_text(test_input, acceptance_threshold):
    """
    Test spacy recognizer with a generated dataset text file
    :param test_input: input text file location
    :param acceptance_threshold: minimim precision/recall
     allowed for tests to pass
    """

    # read test input from generated file
    import os
    from presidio_analyzer.nlp_engine import SpacyNlpEngine

    dir_path = os.path.dirname(os.path.realpath(__file__))
    input_samples = InputSample.read_dataset_json(test_input.format(dir_path))
    
    nlp_engine = SpacyNlpEngine()
    model = PresidioRecognizerWrapper(
        recognizer=SpacyRecognizer(),
        nlp_engine=nlp_engine,
        entities_to_keep=["PERSON"],
        with_nlp_artifacts=True
    )
    evaluator = Evaluator(model=model, entity_mapping={})
    evaluation_results = evaluator.evaluate_all(input_samples)
    scores = evaluator.calculate_score(evaluation_results)
    
    assert acceptance_threshold <= scores.pii_f

import pytest

from presidio_evaluator import InputSample, Span
from presidio_evaluator.evaluation import TokenEvaluator
from presidio_evaluator.models.presidio_analyzer_wrapper import PresidioAnalyzerWrapper


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
analyzer_test_generate_text_testdata = [
    # small set fixture which expects all results.
    GeneratedTextTestCase(
        test_name="small-set",
        test_input="{}/data/generated_small.json",
        acceptance_threshold=0.3,
        marks=pytest.mark.none,
    ),
    # small set fixture which expects all results.
    GeneratedTextTestCase(
        test_name="large-set",
        test_input="{}/data/generated_large.json",
        acceptance_threshold=0.3,
        marks=pytest.mark.slow,
    ),
]


def test_analyzer_simple_input():
    model = PresidioAnalyzerWrapper(entities_to_keep=["PERSON"])
    sample = InputSample(
        full_text="My name is Mike",
        masked="My name is [PERSON]",
        spans=[Span("PERSON", "Mike", 10, 14)],
        create_tags_from_span=True,
    )

    results_df = model.predict_dataset([sample])
    evaluator = TokenEvaluator()
    metrics = evaluator.calculate_score_on_df(results_df)

    assert metrics.pii_precision == 1
    assert metrics.pii_recall == 1


# analyzer tests on generated data
@pytest.mark.parametrize(
    "test_input,acceptance_threshold",
    [testcase.to_pytest_param() for testcase in analyzer_test_generate_text_testdata],
)
def test_analyzer_with_generated_text(test_input, acceptance_threshold):
    """
    Test analyzer with a generated dataset text file
    :param test_input: input text file location
    :param acceptance_threshold: minimum precision/recall
     allowed for tests to pass
    """
    # read test input from generated file

    import os

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    input_samples = InputSample.read_dataset_json(test_input.format(dir_path))

    analyzer = PresidioAnalyzerWrapper()
    results_df = analyzer.predict_dataset(input_samples)
    evaluator = TokenEvaluator()
    scores = evaluator.calculate_score_on_df(results_df)

    assert acceptance_threshold <= scores.pii_precision
    assert acceptance_threshold <= scores.pii_recall

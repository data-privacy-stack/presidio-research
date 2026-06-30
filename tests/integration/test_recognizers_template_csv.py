import os

import numpy as np
import pandas as pd
import pytest
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_analyzer.predefined_recognizers import CreditCardRecognizer

from presidio_evaluator.data_generator import PresidioSentenceFaker
from presidio_evaluator.evaluation import TokenEvaluator
from presidio_evaluator.models import PresidioRecognizerWrapper


class TemplateTextTestCase:
    """
    Test case parameters for tests with dataset generated from a template and csv values
    """

    def __init__(
        self,
        test_name,
        pii_csv,
        utterances,
        num_of_examples,
        acceptance_threshold,
        marks,
    ):
        self.test_name = test_name
        self.pii_csv = pii_csv
        self.utterances = utterances
        self.num_of_examples = num_of_examples
        self.acceptance_threshold = acceptance_threshold
        self.marks = marks

    def to_pytest_param(self):
        return pytest.param(
            self.pii_csv,
            self.utterances,
            self.num_of_examples,
            self.acceptance_threshold,
            id=self.test_name,
            marks=self.marks,
        )


# template-dataset test cases
cc_test_template_testdata = [
    # large dataset fixture. marked as slow
    TemplateTextTestCase(
        test_name="fake-names-100",
        pii_csv="{}/data/FakeNameGenerator.com_100.csv",
        utterances="{}/data/templates.txt",
        num_of_examples=100,
        acceptance_threshold=0.9,
        marks=pytest.mark.none,
    )
]


# credit card recognizer tests on template-generates data
@pytest.mark.parametrize(
    "pii_csv, utterances, num_of_examples, acceptance_threshold",
    [testcase.to_pytest_param() for testcase in cc_test_template_testdata],
)
def test_credit_card_recognizer_with_template(
    pii_csv, utterances, num_of_examples, acceptance_threshold
):
    """
    Test credit card recognizer with a dataset generated from
    template and a CSV values file
    :param pii_csv: input csv file location
    :param utterances: template file location
    :param num_of_examples: number of samples to be used from dataset
    to test
    :param acceptance_threshold: minimum precision/recall
     allowed for tests to pass
    """

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    with open(utterances.format(dir_path)) as utterances_file:
        templates = [line.rstrip() for line in utterances_file]

    pii_csv = pii_csv.format(dir_path)
    base_records = pd.read_csv(pii_csv)
    sentence_faker = PresidioSentenceFaker(
        "en_US",
        lower_case_ratio=0.05,
        sentence_templates=templates,
        base_records=base_records,
    )
    input_samples = sentence_faker.generate_new_fake_sentences(num_of_examples)

    nlp_engine = SpacyNlpEngine()
    model = PresidioRecognizerWrapper(
        recognizer=CreditCardRecognizer(),
        nlp_engine=nlp_engine,
        entities_to_keep=["CREDIT_CARD"],
    )
    results_df = model.predict_dataset(input_samples)
    evaluator = TokenEvaluator(entities_to_keep=model.entities)
    scores = evaluator.calculate_score_on_df(results_df)

    if not np.isnan(scores.pii_f):
        assert acceptance_threshold <= scores.pii_f

from collections import Counter

import pytest

from presidio_evaluator.evaluation import EvaluationResult, Evaluator, ModelError
from tests.mocks import (
    MockTokensModel,
)

@pytest.fixture(scope="session")
def evaluation_result() -> EvaluationResult:
    results = Counter(
        {
            ("O", "O"): 30,
            ("ANIMAL", "ANIMAL"): 4,
            ("ANIMAL", "O"): 2,
            ("O", "ANIMAL"): 1,
            ("PERSON", "PERSON"): 2,
        }
    )

    model_errors =[ModelError(error_type="FP",
                              annotation="PER",
                              prediction="LOC",
                              token="Bob",
                              full_text="Bob likes Jane")]
    evaluation_result = EvaluationResult(results=results,
                                         model_errors=model_errors,
                                         text="Hi Bob",
                                         pii_recall=-1.0,
                                         pii_precision=-1.0,
                                         pii_f=-1.0,
                                         entity_recall_dict={"PER": -1.0},
                                         entity_precision_dict = {"PER": -1.0},
                                         n=-1)
    return evaluation_result

@pytest.fixture(scope="session")
def evaluator():
    model = MockTokensModel(prediction=None)
    evaluator = Evaluator(model=model)
    return evaluator


@pytest.fixture(scope="session")
def scores(evaluation_result, evaluator):
    return evaluator.calculate_score([evaluation_result])


def test_to_confusion_matrix(scores):
    entities, confmatrix = scores.to_confusion_matrix()
    assert "ANIMAL" in entities
    assert "PERSON" in entities
    assert "O" in entities

    s = scores.results
    assert confmatrix[0] == [s[("ANIMAL", "ANIMAL")],
                             s[("ANIMAL", "PERSON")],
                             s[("ANIMAL", "O")]]
    assert confmatrix[1] == [s[("PERSON", "ANIMAL")],
                             s[("PERSON", "PERSON")],
                             s[("PERSON", "O")]]
    assert confmatrix[2] == [s[("O", "ANIMAL")],
                             s[("O", "PERSON")],
                             s[("O", "O")]]

def test_str(scores):
    return_str = str(scores)
    assert (
        "PERSON             100.00%             100.00%                   2"
        in return_str
    )
    assert (
        "ANIMAL              80.00%              66.67%                   6"
        in return_str
    )
    assert (
        "PII              85.71%              75.00%                   8" in return_str
    )


def test_to_confusion_df(scores):
   conf_df = scores.to_confusion_df()
   assert "recall" in conf_df
   rownames = list(set(scores.n_dict.keys()).union("O"))
   rownames.append("precision")
   colnames = list(set(scores.n_dict.keys()).union("O"))
   colnames.append("recall")

   for row in conf_df.iterrows():
       assert row[0] in rownames

   for col in conf_df.columns.to_list():
       assert col in colnames


def test_to_log(evaluation_result):
    log_dict = evaluation_result.to_log()
    assert log_dict["pii_f"] == -1
    assert log_dict["pii_recall"] == -1
    assert log_dict["pii_precision"] == -1
    assert log_dict["PER_precision"] == -1
    assert log_dict["PER_recall"] == -1
    assert log_dict["PER_recall"] == -1
    assert log_dict["n"] == -1

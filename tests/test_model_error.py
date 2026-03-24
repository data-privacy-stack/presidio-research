from collections import Counter

import pytest

from presidio_evaluator.evaluation import ErrorType, ModelError


@pytest.fixture(scope="module")
def model_errors():
    model_errors = [
        ModelError(
            error_type=ErrorType.FP,
            annotation="O",
            prediction="LOC",
            token="Bob",
            full_text="Bob likes Jane",
        ),
        ModelError(
            error_type=ErrorType.FN,
            annotation="LOC",
            prediction="O",
            token="Bob",
            full_text="Bob likes Jane",
        ),
        ModelError(
            error_type=ErrorType.WrongEntity,
            annotation="PER",
            prediction="LOC",
            token="Bob",
            full_text="Bob likes Jane",
        ),
    ]

    return model_errors


@pytest.fixture(scope="module")
def model_errors_extended():
    results = Counter(
        {
            ("X", "X"): 50,
            ("Y", "Y"): 60,
            ("Z", "Z"): 70,
            ("X", "O"): 5,
            ("Y", "O"): 6,
            ("Z", "O"): 7,
            ("O", "X"): 5,
            ("O", "Y"): 6,
            ("O", "Z"): 7,
            ("X", "Y"): 5,
            ("X", "Z"): 5,
            ("Y", "X"): 6,
            ("Y", "Z"): 6,
            ("Z", "X"): 7,
            ("Z", "Y"): 7,
        }
    )
    model_errors = []
    model_errors.extend([ModelError(ErrorType.FN, "X", "O", "a", "")] * 5)
    model_errors.extend([ModelError(ErrorType.FN, "Y", "O", "b", "")] * 6)
    model_errors.extend([ModelError(ErrorType.FN, "Z", "O", "c", "")] * 7)
    model_errors.extend([ModelError(ErrorType.FP, "O", "X", "d", "")] * 5)
    model_errors.extend([ModelError(ErrorType.FP, "O", "Y", "e", "")] * 6)
    model_errors.extend([ModelError(ErrorType.FP, "O", "Z", "f", "")] * 7)
    model_errors.extend([ModelError(ErrorType.WrongEntity, "X", "Y", "g", "")] * 5)
    model_errors.extend([ModelError(ErrorType.WrongEntity, "X", "Z", "h", "")] * 5)
    model_errors.extend([ModelError(ErrorType.WrongEntity, "Y", "X", "i", "")] * 6)
    model_errors.extend([ModelError(ErrorType.WrongEntity, "Y", "Z", "j", "")] * 6)
    model_errors.extend([ModelError(ErrorType.WrongEntity, "Z", "X", "k", "")] * 7)

    return results, model_errors


def test_model_error_most_common_fp_tokens(model_errors):
    most_common = ModelError.most_common_fp_tokens(model_errors)
    assert most_common[0] == ("Bob", 2)  # 1 fp and 1 wrong entity


def test_model_error_most_common_fn_tokens(model_errors):
    most_common = ModelError.most_common_fn_tokens(model_errors)
    assert most_common[0] == ("Bob", 2)  # 1 fn and 1 wrong entity


def test_get_false_positives_existing_entity_returns_errors(model_errors):
    fps = ModelError.get_false_positives(model_errors, "LOC")
    assert fps == [model_errors[0], model_errors[2]]


def test_get_false_positives_no_entity_returns_none(model_errors):
    fps = ModelError.get_false_positives(model_errors, "XYZ")
    assert len(fps) == 0


def test_get_false_negatives_existing_entity_returns_errors(model_errors):
    fns = ModelError.get_false_negatives(model_errors, "LOC")
    assert fns == [model_errors[1]]


def test_get_false_negatives_no_entity_returns_none(model_errors):
    fns = ModelError.get_false_negatives(model_errors, "XYZ")
    assert len(fns) == 0


def test_get_wrong_entity_existing_entity_returns_errors(model_errors):
    wrong1 = ModelError.get_wrong_entities(model_errors, "PER")
    assert wrong1 == [model_errors[2]]


def test_get_wrong_negatives_no_entity_returns_none(model_errors):
    wrong = ModelError.get_wrong_entities(model_errors, "XYZ")
    assert len(wrong) == 0


def test_get_errors_df(model_errors):
    fp_df = ModelError.get_errors_df(
        model_errors, entity="LOC", error_type=ErrorType.FP
    )
    assert fp_df["error_type"][0] == ErrorType.FP
    assert fp_df["annotation"][0] == "O"
    assert fp_df["prediction"][0] == "LOC"
    assert fp_df["token"][0] == "Bob"
    assert fp_df["full_text"][0] == "Bob likes Jane"

    fn_df = ModelError.get_errors_df(
        model_errors, entity="LOC", error_type=ErrorType.FN
    )
    assert fn_df["error_type"][0] == ErrorType.FN
    assert fn_df["annotation"][0] == "LOC"
    assert fn_df["prediction"][0] == "O"
    assert fn_df["token"][0] == "Bob"
    assert fn_df["full_text"][0] == "Bob likes Jane"

    wrong_df = ModelError.get_errors_df(
        model_errors, entity="LOC", error_type=ErrorType.WrongEntity
    )
    assert wrong_df["error_type"][0] == ErrorType.WrongEntity
    assert wrong_df["annotation"][0] == "PER"
    assert wrong_df["prediction"][0] == "LOC"
    assert wrong_df["token"][0] == "Bob"
    assert wrong_df["full_text"][0] == "Bob likes Jane"


def test_get_most_common_fps(model_errors_extended):
    results, model_errors = model_errors_extended

    most_common_fp_X = ModelError.most_common_fp_tokens(model_errors, entity="X")
    assert len(most_common_fp_X) == 3  # one false positive and two wrong entities
    for fp in most_common_fp_X:
        assert fp[0] in ["k", "i", "d"]


def test_get_most_common_fns(model_errors_extended):
    results, model_errors = model_errors_extended

    most_common_fn_X = ModelError.most_common_fn_tokens(model_errors, entity="X")
    assert len(most_common_fn_X) == 3  # one false positive and two wrong entities
    for fn in most_common_fn_X:
        assert fn[0] in ["a", "g", "h"]


def test_get_wrong_annotated_entities(model_errors_extended):
    resuls, model_errors = model_errors_extended
    wrong_entities = ModelError.get_wrong_entities(model_errors, annotated_entity="X")
    assert len(wrong_entities) == 10
    for wrong in wrong_entities:
        assert wrong.annotation == "X"
        assert wrong.token in ["g", "h"]


def test_get_wrong_predicted_entities(model_errors_extended):
    resuls, model_errors = model_errors_extended
    wrong_entities = ModelError.get_wrong_entities(model_errors, predicted_entity="X")
    assert len(wrong_entities) == 13
    for wrong in wrong_entities:
        assert wrong.prediction == "X"
        assert wrong.token in ["i", "k"]

import pandas as pd
import pytest

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import BaseEvaluator, ErrorType, EvaluationResult
from tests.mocks import MockTokensModel


class MockEvaluator(BaseEvaluator):
    def calculate_score(
        self,
        evaluation_results: list[EvaluationResult],
        entities: list[str] | None = None,
        beta: float = 2.0,
    ) -> EvaluationResult:
        pass


def test_evaluate_sample_wrong_entities_to_keep_correct_statistics():
    prediction = ["O", "O", "O", "ANIMAL"]
    model = MockTokensModel(prediction=prediction)

    evaluator = MockEvaluator(model=model, entities_to_keep=["SPACESHIP"])

    sample = InputSample(
        full_text="I am the walrus", masked="I am the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "ANIMAL"]

    evaluated = evaluator.evaluate_sample(sample, prediction)
    assert evaluated.results[("O", "O")] == 4


def test_evaluate_same_entity_correct_statistics():
    prediction = ["O", "ANIMAL", "O", "ANIMAL"]
    model = MockTokensModel(prediction=prediction)
    evaluator = MockEvaluator(
        model=model, entities_to_keep=["ANIMAL"], skip_words=["-"]
    )
    sample = InputSample(
        full_text="I dog the walrus", masked="I [ANIMAL] the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "ANIMAL"]

    evaluation_result = evaluator.evaluate_sample(sample, prediction)
    assert evaluation_result.results[("O", "O")] == 2
    assert evaluation_result.results[("ANIMAL", "ANIMAL")] == 1
    assert evaluation_result.results[("O", "ANIMAL")] == 1


def test_evaluate_multiple_entities_to_keep_correct_statistics():
    prediction = ["O", "ANIMAL", "O", "ANIMAL"]
    entities_to_keep = ["ANIMAL", "PLANT", "SPACESHIP"]
    model = MockTokensModel(prediction=prediction)
    evaluator = MockEvaluator(
        model=model, entities_to_keep=entities_to_keep, skip_words=["-"]
    )

    sample = InputSample(
        full_text="I dog the walrus", masked="I [ANIMAL] the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "ANIMAL"]

    evaluation_result = evaluator.evaluate_sample(sample, prediction)
    assert evaluation_result.results[("O", "O")] == 2
    assert evaluation_result.results[("ANIMAL", "ANIMAL")] == 1
    assert evaluation_result.results[("O", "ANIMAL")] == 1


@pytest.mark.parametrize(
    "tags, predicted_tags, expected_dict",
    [
        (
            ["O", "ID", "SSN"],
            ["O", "SSN", "SSN"],
            {("O", "O"): 1, ("SSN", "SSN"): 2},
        ),
        (
            ["O", "SSN", "SSN"],
            ["O", "ID", "SSN"],
            {("O", "O"): 1, ("SSN", "SSN"): 2},
        ),
        (
            ["O", "MID", "SSN"],
            ["O", "SSN", "SSN"],
            {("O", "O"): 1, ("MID", "SSN"): 1, ("SSN", "SSN"): 1},
        ),
    ],
)
def test_generic_entities_are_treated_like_specific_entities(
    tags, predicted_tags, expected_dict
):
    model = MockTokensModel(prediction=predicted_tags)
    evaluator = MockEvaluator(model=model)

    tokens = ["A", "123", "456"]

    sample = InputSample(full_text=" ".join(tokens), spans=None)
    sample.tokens = tokens
    sample.tags = tags

    evaluated = evaluator.evaluate_sample(sample, predicted_tags)

    assert evaluated.results == expected_dict


def test_error_type_classification():
    """
    Test that error types are correctly classified:
    - FP: Only when predicting an entity where there should be none (O)
    - FN: When missing an entity (predicting O instead of entity)
    - WrongEntity: When predicting wrong entity type (entity mismatch)
    """
    prediction = ["O", "EMAIL", "PHONE", "LOCATION", "PERSON"]

    evaluator = MockEvaluator(model=MockTokensModel(prediction))

    # Ground truth: [PERSON, O, EMAIL, PHONE, O]
    # Prediction:   [PERSON, EMAIL, PHONE, LOCATION, PERSON]
    sample = InputSample(
        full_text="John details john@mail.com 123-456-7890 today",
        tokens=["John", "details", "john@mail.com", "123-456-7890", "today"],
        tags=["PERSON", "O", "EMAIL", "PHONE", "O"],
    )

    result = evaluator.evaluate_sample(sample, prediction)

    # Verify error types
    errors = result.model_errors

    # Classify each error
    fps = [e for e in errors if e.error_type == ErrorType.FP]
    fns = [e for e in errors if e.error_type == ErrorType.FN]
    wrong_entities = [e for e in errors if e.error_type == ErrorType.WrongEntity]

    # Should be 2 FPs: "is"->EMAIL and "there"->PERSON
    assert len(fps) == 2
    assert any(e.token == "details" and e.prediction == "EMAIL" for e in fps)
    assert any(e.token == "today" and e.prediction == "PERSON" for e in fps)

    # Should be 1 FNs: Missing PERSON (pun not intended :))
    assert len(fns) == 1
    assert any(e.token == "John" and e.annotation == "PERSON" for e in fns)

    # Should be 2 WrongEntity: PHONE->LOCATION, EMAIL->PHONE
    assert len(wrong_entities) == 2
    assert any(
        e.token == "john@mail.com"
        and e.annotation == "EMAIL"
        and e.prediction == "PHONE"
        for e in wrong_entities
    )
    assert any(
        e.token == "123-456-7890"
        and e.annotation == "PHONE"
        and e.prediction == "LOCATION"
        for e in wrong_entities
    )


def test_get_results_dataframe_basic():
    """Test the basic functionality of get_results_dataframe without entity filtering."""
    evaluation_results = [
        EvaluationResult(
            tokens=["I", "am", "John", "Smith", "from", "New", "York"],
            actual_tags=["O", "O", "PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            predicted_tags=["O", "O", "PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            start_indices=[0, 1, 2, 3, 4, 5, 6],
            results={},  # Not needed for these tests
        )
    ]

    evaluator = MockEvaluator(model=None)
    df = evaluator.get_results_dataframe(evaluation_results)

    # Verify the dataframe has the correct shape and columns
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (7, 5)
    assert list(df.columns) == [
        "sentence_id",
        "token",
        "annotation",
        "prediction",
        "start_indices",
    ]

    # Verify the data is correct
    assert list(df["token"]) == ["I", "am", "John", "Smith", "from", "New", "York"]
    assert list(df["annotation"]) == [
        "O",
        "O",
        "PERSON",
        "PERSON",
        "O",
        "LOCATION",
        "LOCATION",
    ]
    assert list(df["prediction"]) == [
        "O",
        "O",
        "PERSON",
        "PERSON",
        "O",
        "LOCATION",
        "LOCATION",
    ]
    assert list(df["start_indices"]) == [0, 1, 2, 3, 4, 5, 6]


def test_get_results_dataframe_with_entity_filtering():
    """Test that get_results_dataframe filters entities correctly when provided with a list."""
    evaluation_results = [
        EvaluationResult(
            tokens=["I", "am", "John", "Smith", "from", "New", "York"],
            actual_tags=["O", "O", "PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            predicted_tags=["O", "O", "PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            start_indices=[0, 1, 2, 3, 4, 5, 6],
            results={},
        )
    ]

    evaluator = MockEvaluator(model=None)
    # Filter to only include PERSON entities
    df = evaluator.get_results_dataframe(evaluation_results, entities=["PERSON"])

    # Verify that LOCATION tags are filtered out (replaced with "O")
    assert list(df["annotation"]) == ["O", "O", "PERSON", "PERSON", "O", "O", "O"]
    assert list(df["prediction"]) == ["O", "O", "PERSON", "PERSON", "O", "O", "O"]


def test_get_results_dataframe_with_multiple_entities():
    """Test filtering with multiple entities."""
    evaluation_results = [
        EvaluationResult(
            tokens=[
                "My",
                "name",
                "is",
                "John",
                "Smith",
                "and",
                "my",
                "email",
                "is",
                "john@example.com",
            ],
            actual_tags=[
                "O",
                "O",
                "O",
                "PERSON",
                "PERSON",
                "O",
                "O",
                "O",
                "O",
                "EMAIL",
            ],
            predicted_tags=[
                "O",
                "O",
                "O",
                "PERSON",
                "PERSON",
                "O",
                "O",
                "O",
                "O",
                "EMAIL",
            ],
            start_indices=list(range(10)),
            results={},
        )
    ]

    evaluator = MockEvaluator(model=None)
    # Filter to only include PERSON entities
    df_person = evaluator.get_results_dataframe(evaluation_results, entities=["PERSON"])
    assert list(df_person["annotation"]) == [
        "O",
        "O",
        "O",
        "PERSON",
        "PERSON",
        "O",
        "O",
        "O",
        "O",
        "O",
    ]

    # Filter to only include EMAIL entities
    df_email = evaluator.get_results_dataframe(evaluation_results, entities=["EMAIL"])
    assert list(df_email["annotation"]) == [
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
        "EMAIL",
    ]

    # Include both PERSON and EMAIL entities
    df_both = evaluator.get_results_dataframe(
        evaluation_results, entities=["PERSON", "EMAIL"]
    )
    assert list(df_both["annotation"]) == [
        "O",
        "O",
        "O",
        "PERSON",
        "PERSON",
        "O",
        "O",
        "O",
        "O",
        "EMAIL",
    ]


def test_get_results_dataframe_with_mismatched_predictions():
    """Test that filtering works correctly when annotations and predictions have different entity types."""
    evaluation_results = [
        EvaluationResult(
            tokens=["John", "Smith", "lives", "in", "New", "York"],
            actual_tags=["PERSON", "PERSON", "O", "O", "LOCATION", "LOCATION"],
            predicted_tags=[
                "PERSON",
                "PERSON",
                "O",
                "O",
                "CITY",
                "CITY",
            ],  # Predicted CITY instead of LOCATION
            start_indices=list(range(6)),
            results={},
        )
    ]

    evaluator = MockEvaluator(model=None)
    # Filter to only include PERSON entities
    df_person = evaluator.get_results_dataframe(evaluation_results, entities=["PERSON"])
    assert list(df_person["annotation"]) == ["PERSON", "PERSON", "O", "O", "O", "O"]
    assert list(df_person["prediction"]) == ["PERSON", "PERSON", "O", "O", "O", "O"]

    # Filter to only include LOCATION entities
    df_location = evaluator.get_results_dataframe(
        evaluation_results, entities=["LOCATION"]
    )
    assert list(df_location["annotation"]) == [
        "O",
        "O",
        "O",
        "O",
        "LOCATION",
        "LOCATION",
    ]
    assert list(df_location["prediction"]) == [
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
    ]  # CITY is filtered out

    # Filter to only include CITY entities
    df_city = evaluator.get_results_dataframe(evaluation_results, entities=["CITY"])
    assert list(df_city["annotation"]) == [
        "O",
        "O",
        "O",
        "O",
        "O",
        "O",
    ]  # LOCATION is filtered out
    assert list(df_city["prediction"]) == ["O", "O", "O", "O", "CITY", "CITY"]


def test_get_results_dataframe_with_multiple_sentences():
    """Test filtering with multiple evaluation results (sentences)."""
    evaluation_results = [
        EvaluationResult(
            tokens=["John", "Smith", "lives", "in", "New", "York"],
            actual_tags=["PERSON", "PERSON", "O", "O", "LOCATION", "LOCATION"],
            predicted_tags=["PERSON", "PERSON", "O", "O", "LOCATION", "LOCATION"],
            start_indices=list(range(6)),
            results={},
        ),
        EvaluationResult(
            tokens=["Jane", "Doe", "works", "at", "Microsoft"],
            actual_tags=["PERSON", "PERSON", "O", "O", "ORG"],
            predicted_tags=["PERSON", "PERSON", "O", "O", "ORG"],
            start_indices=list(range(5)),
            results={},
        ),
    ]

    evaluator = MockEvaluator(model=None)
    # Filter to only include PERSON entities
    df = evaluator.get_results_dataframe(evaluation_results, entities=["PERSON"])

    # Verify that the dataframe has the correct shape and columns
    assert df.shape == (
        11,
        5,
    )  # 6 tokens in first sentence + 5 tokens in second sentence

    # Check that only PERSON entities are included and others are filtered out
    assert list(df[df["sentence_id"] == 0]["annotation"]) == [
        "PERSON",
        "PERSON",
        "O",
        "O",
        "O",
        "O",
    ]
    assert list(df[df["sentence_id"] == 1]["annotation"]) == [
        "PERSON",
        "PERSON",
        "O",
        "O",
        "O",
    ]


def test_empty_evaluation_results():
    """Test that an error is raised when evaluation results are empty."""
    evaluator = MockEvaluator(model=None)
    with pytest.raises(ValueError):
        evaluator.get_results_dataframe([])


def test_evaluation_results_without_tokens():
    """Test that an error is raised when evaluation results don't have tokens."""
    # Create an EvaluationResult with empty tokens
    empty_result = EvaluationResult(
        tokens=[], actual_tags=[], predicted_tags=[], start_indices=[], results={}
    )

    evaluator = MockEvaluator(model=None)
    with pytest.raises(ValueError):
        evaluator.get_results_dataframe([empty_result])


def test_results_to_dataframe_with_entity_filtering():
    """
    Test that get_results_dataframe correctly filters entities when predictions
    and actual tags are different
    """
    prediction = ["PERSON", "EMAIL", "PHONE", "CITY", "PERSON"]
    tokens = ["John", "details", "john@example.com", "New York", "Smith"]
    tags = ["PERSON", "O", "EMAIL", "LOCATION", "PERSON"]
    start_indices = [0, 5, 13, 27, 40]
    evaluator = MockEvaluator(model=MockTokensModel(prediction))

    sample = InputSample(
        full_text="John details john@example.com New York Smith",
        tokens=tokens,
        start_indices=start_indices,
        tags=tags,
    )

    results = [evaluator.evaluate_sample(sample, prediction)]

    # Test filtering for just PERSON entities
    df_person = evaluator.get_results_dataframe(results, entities=["PERSON"])

    # Verify that other entities are filtered out (replaced with "O")
    assert list(df_person["annotation"]) == ["PERSON", "O", "O", "O", "PERSON"]
    assert list(df_person["prediction"]) == ["PERSON", "O", "O", "O", "PERSON"]

    # Test filtering for just EMAIL entities
    df_email = evaluator.get_results_dataframe(results, entities=["EMAIL"])
    assert list(df_email["annotation"]) == ["O", "O", "EMAIL", "O", "O"]
    assert list(df_email["prediction"]) == ["O", "EMAIL", "O", "O", "O"]

    # Test filtering for LOCATION entity (which is predicted as CITY)
    df_location = evaluator.get_results_dataframe(results, entities=["LOCATION"])
    assert list(df_location["annotation"]) == ["O", "O", "O", "LOCATION", "O"]
    assert list(df_location["prediction"]) == [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]  # CITY is filtered out

    # Test filtering for CITY entity (which is annotated as LOCATION)
    df_city = evaluator.get_results_dataframe(results, entities=["CITY"])
    assert list(df_city["annotation"]) == [
        "O",
        "O",
        "O",
        "O",
        "O",
    ]  # LOCATION is filtered out
    assert list(df_city["prediction"]) == ["O", "O", "O", "CITY", "O"]


def test_get_results_dataframe_emits_deprecation_warning():
    """get_results_dataframe() must emit a DeprecationWarning (soft deprecation)."""
    import warnings

    prediction = ["PERSON", "O"]
    tokens = ["Alice", "here"]
    tags = ["PERSON", "O"]
    start_indices = [0, 6]
    evaluator = MockEvaluator(model=MockTokensModel(prediction), skip_words=[])

    sample = InputSample(
        full_text="Alice here",
        tokens=tokens,
        start_indices=start_indices,
        tags=tags,
    )
    result = evaluator.evaluate_sample(sample, prediction)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = evaluator.get_results_dataframe([result])

    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "get_results_dataframe()" in str(caught[0].message)
    assert "predict_dataset" in str(caught[0].message)
    assert "get_mapped_results_dataframe" in str(caught[0].message)
    # method still executes (soft deprecation)
    assert list(df["annotation"]) == ["PERSON", "O"]

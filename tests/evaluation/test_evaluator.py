import pytest

from presidio_evaluator import InputSample
from presidio_evaluator.evaluation import DeprecationError, ErrorType, EvaluationResult
from presidio_evaluator.evaluation.token_evaluator import TokenEvaluator
from tests.mocks import MockTokensModel


class MockEvaluator(TokenEvaluator):
    def calculate_score(
        self,
        evaluation_results: list[EvaluationResult],
        entities: list[str] | None = None,
        beta: float = 2.0,
    ) -> EvaluationResult:
        pass


def test_evaluate_sample_wrong_entities_to_keep_correct_statistics():
    prediction = ["O", "O", "O", "ANIMAL"]

    evaluator = MockEvaluator(model=None, entities_to_keep=["SPACESHIP"])

    sample = InputSample(
        full_text="I am the walrus", masked="I am the [ANIMAL]", spans=None
    )
    sample.tokens = ["I", "am", "the", "walrus"]
    sample.tags = ["O", "O", "O", "ANIMAL"]

    evaluated = evaluator.evaluate_sample(sample, prediction)
    assert evaluated.results[("O", "O")] == 4


def test_evaluate_same_entity_correct_statistics():
    prediction = ["O", "ANIMAL", "O", "ANIMAL"]
    evaluator = MockEvaluator(model=None, entities_to_keep=["ANIMAL"], skip_words=["-"])
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
    evaluator = MockEvaluator(
        model=None, entities_to_keep=entities_to_keep, skip_words=["-"]
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
    evaluator = MockEvaluator()

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

    evaluator = MockEvaluator()

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
    """get_results_dataframe() has been removed and raises DeprecationError."""
    evaluator = MockEvaluator()
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_get_results_dataframe_with_entity_filtering():
    """get_results_dataframe() has been removed and raises DeprecationError."""
    evaluator = MockEvaluator()
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_get_results_dataframe_with_multiple_entities():
    """get_results_dataframe() has been removed and raises DeprecationError."""
    evaluator = MockEvaluator()
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_get_results_dataframe_with_mismatched_predictions():
    """get_results_dataframe() has been removed and raises DeprecationError."""
    evaluator = MockEvaluator()
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_get_results_dataframe_with_multiple_sentences():
    """get_results_dataframe() has been removed and raises DeprecationError."""
    evaluator = MockEvaluator()
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_empty_evaluation_results():
    """get_results_dataframe() raises DeprecationError regardless of input."""
    evaluator = MockEvaluator()
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_evaluation_results_without_tokens():
    """get_results_dataframe() raises DeprecationError regardless of input."""
    evaluator = MockEvaluator()
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_results_to_dataframe_with_entity_filtering():
    """get_results_dataframe() has been removed and raises DeprecationError."""
    evaluator = MockEvaluator()
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_get_results_dataframe_emits_deprecation_warning():
    """get_results_dataframe() now raises DeprecationError (hard deprecation)."""
    evaluator = MockEvaluator(model=None, skip_words=[])
    with pytest.raises(DeprecationError):
        evaluator.get_results_dataframe([])


def test_model_constructor_raises_deprecation_error():
    """Passing a non-None model to the evaluator constructor raises DeprecationError."""
    from presidio_evaluator.evaluation import DeprecationError as DE

    model = MockTokensModel(prediction=["O"])
    with pytest.raises(DE):
        MockEvaluator(model=model)


def test_model_none_constructor_does_not_raise():
    """SpanEvaluator() must not raise."""
    from presidio_evaluator.evaluation import SpanEvaluator

    evaluator = SpanEvaluator(model=None, skip_words=[])
    assert evaluator is not None


def test_evaluate_sample_emits_deprecation_warning():
    """evaluate_sample() must emit DeprecationWarning (soft deprecation — still executes)."""
    import warnings

    evaluator = MockEvaluator(model=None, skip_words=[])
    sample = InputSample(
        full_text="Alice here",
        tokens=["Alice", "here"],
        start_indices=[0, 6],
        tags=["PERSON", "O"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = evaluator.evaluate_sample(sample, ["PERSON", "O"])

    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) >= 1
    assert any("evaluate_sample" in str(w.message) for w in deprecation_warnings)
    # still executes
    assert result is not None

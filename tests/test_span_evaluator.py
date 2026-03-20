import numpy as np
import pandas as pd
import pytest
from presidio_evaluator.data_objects import Span

from presidio_evaluator.evaluation import EvaluationResult, ErrorType, SpanEvaluator
from tests.mocks import MockModel, MockTokensModel


@pytest.fixture
def span_evaluator():
    """Create a SpanEvaluator instance for testing."""
    return SpanEvaluator(
        model=MockModel(), 
        entity_mapping={},  # No mapping needed for these tests
        iou_threshold=0.75, 
        char_based=True, 
        skip_words=None
    )


# Test Scenario group 1: single span overlaps


def assert_error_types(expected_error_types, result, scenario):
    # Check error types
    if expected_error_types:
        error_types = [error.error_type for error in result.model_errors]
        for expected_type in expected_error_types:
            assert (
                expected_type in error_types
            ), f"In {scenario}, expected error type {expected_type} not found in {error_types}"


def assert_confusion_matrix(expected_results, result, scenario):
    # Check confusion matrix results
    for (ann, pred), expected_count in expected_results.items():
        actual_count = result.results.get((ann, pred), 0)
        assert actual_count == expected_count, (
            f"In {scenario}, confusion matrix entry ({ann}, {pred}) "
            f"expected to be {expected_count}, got {actual_count}"
        )


def assert_metric(expected_pii_metric, metric_name, result, scenario):
    metric = ""
    match metric_name:
        case "precision":
            metric = result.precision
        case "recall":
            metric = result.recall
        case "f_beta":
            metric = result.f_beta
        case "pii_precision":
            metric = result.pii_precision
        case "pii_recall":
            metric = result.pii_recall
        case "pii_f":
            metric = result.pii_f
        case _:
            raise ValueError(f"Unknown metric name: {metric_name}")

    if np.isnan(expected_pii_metric):
        assert np.isnan(
            metric
        ), f"In {scenario}, expected {metric_name} score to be None, got {metric}"
    else:
        assert (
            metric == pytest.approx(expected_pii_metric, 3)
        ), f"In {scenario}, expected {metric_name} score {expected_pii_metric}, got {metric}"


@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_tp, expected_fp, expected_fn, expected_results, expected_error_types",
    [
        # Scenario 1: Same type overlap with IoU > threshold
        (
            "Scenario 1: Same type overlap with IoU > threshold",
            ["O", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "PERSON", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            1,  # true positives
            0,  # false positives
            0,  # false negatives
            {("PERSON", "PERSON"): 1},  # confusion matrix
            [],  # no errors
        ),
        # Scenario 2: No overlap with annotated
        (
            "Scenario 2: No overlap with annotated",
            ["O", "PERSON", "PERSON", "O"],
            ["O", "O", "O", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            0,  # true positives
            0,  # false positives
            1,  # false negatives
            {("PERSON", "O"): 1},  # confusion matrix
            [ErrorType.FN],  # error types
        ),
        # Scenario 3: No overlap with predicted
        (
            "Scenario 3: No overlap with predicted",
            ["O", "O", "O", "O"],
            ["O", "PERSON", "PERSON", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            0,  # true positives
            1,  # false positives
            0,  # false negatives
            {("O", "PERSON"): 1},  # confusion matrix
            [ErrorType.FP],  # error types
        ),
        # Scenario 4: Different type overlap with IoU > threshold
        (
            "Scenario 4: Different type overlap with IoU > threshold",
            ["O", "LOCATION", "LOCATION", "O"],
            ["O", "PERSON", "PERSON", "O"],
            ["The", "New", "York", "visited"],
            [0, 4, 8, 13],
            0,  # true positives
            1,  # false positives
            1,  # false negatives
            {("LOCATION", "PERSON"): 1},  # confusion matrix
            [ErrorType.FP, ErrorType.FN, ErrorType.WrongEntity],  # error types
        ),
        # Scenario 5a: Same type overlap with IoU < threshold
        (
            "Scenario 5a: Same type overlap with IoU < threshold",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "O", "O", "O"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 15, 23],
            0,  # true positives
            1,  # false positives
            1,  # false negatives
            {("PERSON", "O"): 1, ("O", "PERSON"): 1},  # confusion matrix
            [ErrorType.FN, ErrorType.FP],  # error types
        ),
        # Scenario 5b: Different type overlap with IoU < threshold
        (
            "Scenario 5b: Different type overlap with IoU < threshold",
            ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["O", "PERSON", "O", "O", "O"],
            ["The", "New", "York", "Mets", "visited"],
            [0, 4, 8, 13, 18],
            0,  # true positives
            1,  # false positives
            1,  # false negatives
            {("ORGANIZATION", "O"): 1, ("O", "PERSON"): 1},  # confusion matrix
            [ErrorType.FN, ErrorType.FP],  # error types
        ),
    ],
)
def test_scenario_group1(
    span_evaluator,
    scenario,
    annotation,
    prediction,
    tokens,
    start_indices,
    expected_tp,
    expected_fp,
    expected_fn,
    expected_results,
    expected_error_types,
):
    """Test each scenario in Group 1: single span overlaps."""
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run evaluation
    result = EvaluationResult()
    result = span_evaluator.calculate_score_on_df(
        per_type=True, results_df=df, evaluation_result=result
    )

    # Check true positives, false positives, and false negatives
    total_tp = sum(pii_type.true_positives for pii_type in result.per_type.values())
    total_fp = sum(pii_type.false_positives for pii_type in result.per_type.values())
    total_fn = sum(pii_type.false_negatives for pii_type in result.per_type.values())

    assert (
        total_tp == expected_tp
    ), f"In {scenario}, expected {expected_tp} TPs, got {total_tp}"
    assert (
        total_fp == expected_fp
    ), f"In {scenario}, expected {expected_fp} FPs, got {total_fp}"
    assert (
        total_fn == expected_fn
    ), f"In {scenario}, expected {expected_fn} FNs, got {total_fn}"

    # Check confusion matrix results
    for (ann, pred), expected_count in expected_results.items():
        actual_count = result.results.get((ann, pred), 0)
        assert actual_count == expected_count, (
            f"In {scenario}, confusion matrix entry ({ann}, {pred}) "
            f"expected to be {expected_count}, got {actual_count}"
        )

    # Check error types
    if expected_error_types:
        error_types = [error.error_type for error in result.model_errors]
        for expected_type in expected_error_types:
            assert (
                expected_type in error_types
            ), f"In {scenario}, expected error type {expected_type} not found in {error_types}"

        assert len(result.model_errors) == len(expected_error_types)


# Test Scenario group 2: annotated spans overlapping with multiple prediction spans


@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_tp, expected_fp, expected_fn, expected_results, expected_error_types",
    [
        # Scenario 6A: Cumulative IoU with spans of the same type > threshold
        (
            "Scenario 6A: Cumulative IoU with spans of the same type > threshold",
            ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["The", "New", "York", "Mets", "visited"],
            [0, 4, 8, 13, 18],
            1,  # true positives (combined spans make a good match)
            0,  # false positives
            0,  # false negatives
            {("ORGANIZATION", "ORGANIZATION"): 1},  # confusion matrix
            [],  # no errors
        ),
        # Scenario 6B: Cumulative IoU with spans of the same type < threshold
        (
            "Scenario 6B: Cumulative IoU with spans of the same type < threshold",
            ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["O", "ORGANIZATION", "O", "ORGANIZATION", "O"],
            ["The", "New", "York", "Mets", "visited"],
            [0, 4, 8, 13, 18],
            0,  # true positives
            1,  # false positives
            1,  # false negatives
            {
                ("ORGANIZATION", "O"): 1,
                ("O", "ORGANIZATION"): 1,
            },  # confusion matrix
            [ErrorType.FN, ErrorType.FP],  # errors
        ),
        # Scenario 7A: Cumulative IoU with spans of different types > threshold
        (
            "Scenario 7A: Cumulative IoU with spans of different types > threshold",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "LOCATION", "LOCATION", "O"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 17, 25],
            0,  # true positives
            2,  # false positives
            1,  # false negatives
            {("PERSON", "LOCATION"): 1, ("O", "PERSON"): 1},
            [ErrorType.WrongEntity, ErrorType.FP],  # errors
        ),
        # Scenario 7B: Cumulative IoU with spans of different types < threshold
        (
            "Scenario 7B: Cumulative IoU with spans of different types < threshold",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "O", "LOCATION", "O", "PERSON"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 17, 25],
            0,  # true positives
            2,  # false positives
            1,  # false negatives
            {
                ("PERSON", "O"): 1,
                ("O", "LOCATION"): 1,
                ("O", "PERSON"): 1,
            },  # confusion matrix
            [ErrorType.FN, ErrorType.FP, ErrorType.FP],  # errors
        ),
        # Mixed case with both same and different types
        (
            "Mixed case: overlapping with both same and different entity types",
            ["O", "PERSON", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "O", "LOCATION", "PERSON", "O"],
            ["The", "John", "C", "B", "Johnson", "visited"],
            [0, 4, 9, 11, 13, 21],
            1,  # true positives (the parts that match)
            1,  # false positives (the wrong type)
            0,  # false negatives (since cumulative IoU is good)
            {
                ("PERSON", "PERSON"): 1,
                ("O", "LOCATION"): 1,
            },  # confusion matrix. (O, LOCATION) because of low IoU
            [
                ErrorType.FP
            ],  # Not WrongEntity since cumulative IoU between Location and Person is low
        ),
    ],
)
def test_scenario_group2(
    span_evaluator,
    scenario,
    annotation,
    prediction,
    tokens,
    start_indices,
    expected_tp,
    expected_fp,
    expected_fn,
    expected_results,
    expected_error_types,
):
    """Test each scenario in Group 2: annotated spans overlapping with multiple prediction spans."""
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run evaluation
    result = EvaluationResult()
    result = span_evaluator.calculate_score_on_df(
        per_type=True, results_df=df, evaluation_result=result
    )

    # Check true positives, false positives, and false negatives
    total_tp = sum(pii_type.true_positives for pii_type in result.per_type.values())
    total_fp = sum(pii_type.false_positives for pii_type in result.per_type.values())
    total_fn = sum(pii_type.false_negatives for pii_type in result.per_type.values())

    assert (
        total_tp == expected_tp
    ), f"In {scenario}, expected {expected_tp} TPs, got {total_tp}"
    assert (
        total_fp == expected_fp
    ), f"In {scenario}, expected {expected_fp} FPs, got {total_fp}"
    assert (
        total_fn == expected_fn
    ), f"In {scenario}, expected {expected_fn} FNs, got {total_fn}"

    assert_confusion_matrix(expected_results, result, scenario)

    assert_error_types(expected_error_types, result, scenario)


# Test global PII evaluatiom metrics


@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_precision, expected_recall, expected_f, expected_tp, expected_fp, expected_fn, expected_annotated, expected_predicted",
    [
        # Perfect match
        (
            "Scenario 1: Perfect match",
            ["O", "PERSON", "PERSON", "O"],
            ["O", "LOCATION", "PERSON", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            1.0,  # precision
            1.0,  # recall
            1.0,  # F1 score
            1,  # true positives
            0,  # false positives
            0,  # false negatives
            1,  # annotated PII spans
            1,  # predicted PII spans
        ),
        # No matches
        (
            "Scenario 2: No matches",
            ["O", "PERSON", "ORGANIZATION", "O"],
            ["O", "O", "O", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            np.nan,  # precision
            0.0,  # recall
            np.nan,  # F1 score
            0,  # true positives
            0,  # false positives
            1,  # false negatives
            1,  # annotated PII spans
            0,  # predicted PII spans
        ),
        # One match out of two
        (
            "Scenario 3: Partial match",
            ["O", "PERSON", "O", "CAT"],
            ["O", "PERSON", "O", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            1.0,  # precision
            0.5,  # recall
            0.5555,  # F1 score
            1,  # true positives
            0,  # false positives
            1,  # false negatives
            2,  # annotated PII spans
            1,  # predicted PII spans
        ),
        # Global entities with overlap but IoU below threshold - results in FN count update
        (
            "Scenario 4: Global entities with overlap below IoU threshold",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "LOCATION", "O", "O", "O"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 15, 23],
            0.0,  # precision (0 TP out of 1 predicted)
            0.0,  # recall (0 TP out of 1 annotated)
            np.nan,  # F1 score
            0,  # true positives
            1,  # false positives
            1,  # false negatives
            1,  # annotated PII spans
            1,  # predicted PII spans
        ),
        # Global entities with multiple overlaps and IoU above threshold - results in TP
        (
            "Scenario 5: Global entities with multiple overlaps above IoU threshold",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "LOCATION", "LOCATION", "ORGANIZATION", "O"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 15, 23],
            1.0,  # precision (1 TP out of 1 predicted - all predictions are treated as single PII)
            1.0,  # recall (1 TP out of 1 annotated)
            1.0,  # F1 score
            1,  # true positives
            0,  # false positives
            0,  # false negatives
            1,  # annotated PII spans
            1,  # predicted PII spans (multiple entity types become single PII span)
        ),
        # Global entities with multiple prediction spans overlapping one annotation - updates TP and pred count
        (
            "Scenario 6: Global entities with multiple prediction spans overlapping annotation",
            ["O", "PERSON", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "LOCATION", "ORGANIZATION", "O", "PERSON", "O"],
            ["The", "John", "Smith", "Jr", "Doe", "visited"],
            [0, 4, 9, 15, 18, 22],
            1.0,  # precision (1 TP out of 1 predicted PII span)
            1.0,  # recall (1 TP out of 1 annotated PII span)
            1.0,  # F1 score
            1,  # true positives
            0,  # false positives
            0,  # false negatives
            1,  # annotated PII spans (one PERSON span)
            1,  # predicted PII spans (multiple entity types become single PII span)
        ),
        # Global entities with standalone predictions (no annotation overlap) - results in FP count update
        (
            "Scenario 7: Global entities with standalone predictions (no annotation overlap)",
            ["O", "O", "O", "O", "O"],
            ["O", "PERSON", "LOCATION", "O", "O"],
            ["The", "quick", "brown", "fox", "jumped"],
            [0, 4, 10, 16, 20],
            0.0,  # precision (0 TP out of 1 predicted PII)
            np.nan,  # recall (0 TP out of 0 annotated)
            np.nan,  # F1 score
            0,  # true positives
            1,  # false positives (standalone predictions become one PII span)
            0,  # false negatives
            0,  # annotated PII spans
            1,  # predicted PII spans (multiple entity types become single PII span)
        ),
    ],
)
def test_global_metrics(
    span_evaluator,
    scenario,
    annotation,
    prediction,
    tokens,
    start_indices,
    expected_precision,
    expected_recall,
    expected_f,
    expected_tp,
    expected_fp,
    expected_fn,
    expected_annotated,
    expected_predicted,
):
    """Test global PII evaluation metrics."""
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run evaluation
    result = EvaluationResult()
    pii_df = span_evaluator.create_global_entities_df(df)
    result = span_evaluator.calculate_score_on_df(
        per_type=False, results_df=pii_df, evaluation_result=result
    )

    # Check global counts
    assert (
        result.pii_annotated == expected_annotated
    ), f"In {scenario}, expected {expected_annotated} annotated PII spans, got {result.pii_annotated}"

    assert (
        result.pii_predicted == expected_predicted
    ), f"In {scenario}, expected {expected_predicted} predicted PII spans, got {result.pii_predicted}"

    assert (
        result.pii_true_positives == expected_tp
    ), f"In {scenario}, expected {expected_tp} true positives, got {result.pii_true_positives}"

    assert (
        result.pii_false_positives == expected_fp
    ), f"In {scenario}, expected {expected_fp} false positives, got {result.pii_false_positives}"

    assert (
        result.pii_false_negatives == expected_fn
    ), f"In {scenario}, expected {expected_fn} false negatives, got {result.pii_false_negatives}"

    # Check global metrics
    assert_metric(expected_precision, "pii_precision", result, scenario)
    assert_metric(expected_recall, "pii_recall", result, scenario)
    assert_metric(expected_f, "pii_f", result, scenario)


# Test Both per_type and global metrics together
@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_per_type_metrics, expected_global_metrics",
    [
        # Perfect detection scenario
        (
            "Perfect detection: all entities found correctly",
            ["PERSON", "PERSON", "O", "O", "LOCATION", "O"],
            ["PERSON", "PERSON", "O", "O", "LOCATION", "O"],
            ["John", "Smith", "lives", "in", "Boston", "today"],
            [0, 5, 11, 17, 20, 27],
            {
                "PERSON": {"precision": 1.0, "recall": 1.0, "f_beta": 1.0},
                "LOCATION": {"precision": 1.0, "recall": 1.0, "f_beta": 1.0},
            },
            {"precision": 1.0, "recall": 1.0, "f_beta": 1.0},
        ),
        # Address partially detected, with multiple pred spans per annotated span
        (
            "Address partially detected: street number found but street name missed",
            ["PERSON", "O", "O", "ADDRESS", "ADDRESS", "ADDRESS", "O"],
            ["PERSON", "O", "O", "ADDRESS", "O", "ADDRESS", "O"],
            ["Alice", "lives", "at", "123", "Main", "Circle", "downtown"],
            [0, 6, 12, 15, 19, 24, 31],
            {
                "PERSON": {"precision": 1.0, "recall": 1.0, "f_beta": 1.0},
                "ADDRESS": {"precision": 0.0, "recall": 0.0, "f_beta": np.nan},
            },
            {"precision": 1.0, "recall": 0.6666666666666666, "f_beta": 0.8},
        ),
        # Mixed entity type confusion
        (
            "Entity type confusion: location detected as person",
            ["PERSON", "PERSON", "O", "O", "LOCATION", "O"],
            ["PERSON", "PERSON", "O", "O", "PERSON", "O"],
            ["Bob", "Davis", "went", "to", "Chicago", "yesterday"],
            [0, 4, 10, 15, 18, 26],
            {
                "PERSON": {"precision": 0.5, "recall": 1.0, "f_beta": 0.5},
                "LOCATION": {"precision": np.nan, "recall": 0.0, "f_beta": np.nan},
            },
            {"precision": 0.6666666666666666, "recall": 1.0, "f_beta": 0.8},
        ),
        # Complete miss for one entity type
        (
            "Complete miss: phone number not detected at all",
            ["PERSON", "O", "PHONE_NUMBER", "PHONE_NUMBER", "O"],
            ["PERSON", "O", "O", "O", "O"],
            ["Sarah", "called", "555", "1234", "today"],
            [0, 6, 13, 17, 22],
            {
                "PERSON": {"precision": 1.0, "recall": 1.0, "f_beta": 1.0},
                "PHONE_NUMBER": {"precision": np.nan, "recall": 0.0, "f_beta": np.nan},
            },
            {"precision": 1.0, "recall": 0.5, "f_beta": 0.6666666666666666},
        ),
        # False positive scenario
        (
            "False positive: common word detected as entity",
            ["O", "PERSON", "O", "O", "O", "O"],
            ["O", "PERSON", "O", "LOCATION", "O", "O"],
            ["The", "president", "spoke", "about", "security", "issues"],
            [0, 4, 14, 20, 26, 35],
            {
                "PERSON": {"precision": 1.0, "recall": 1.0, "f_beta": 1.0},
                "LOCATION": {"precision": 0.0, "recall": 0.0, "f_beta": 0.0},
            },
            {"precision": 0.5, "recall": 1.0, "f_beta": 0.6666666666666666},
        ),
        # No entities detected
        (
            "No detection: all entities missed",
            ["PERSON", "PERSON", "O", "LOCATION", "O"],
            ["O", "O", "O", "O", "O"],
            ["Emma", "Thompson", "visited", "Paris", "today"],
            [0, 5, 14, 22, 28],
            {
                "PERSON": {"precision": np.nan, "recall": 0.0, "f_beta": np.nan},
                "LOCATION": {"precision": np.nan, "recall": 0.0, "f_beta": np.nan},
            },
            {"precision": np.nan, "recall": 0.0, "f_beta": np.nan},
        ),
    ],
)
def test_combined_per_type_and_global_metrics(
    span_evaluator,
    scenario,
    annotation,
    prediction,
    tokens,
    start_indices,
    expected_per_type_metrics,
    expected_global_metrics,
):
    """Test that per-type and global metrics are calculated correctly together."""
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run per-type evaluation
    result = EvaluationResult()
    result = span_evaluator.calculate_score_on_df(
        per_type=True, results_df=df, evaluation_result=result
    )

    # Check per-type metrics
    for entity_type, expected_metrics in expected_per_type_metrics.items():
        if entity_type in result.per_type:
            per_type_result = result.per_type[entity_type]
            assert_metric(
                expected_metrics["precision"], "precision", per_type_result, scenario
            )
            assert_metric(
                expected_metrics["recall"], "recall", per_type_result, scenario
            )
            assert_metric(
                expected_metrics["f_beta"], "f_beta", per_type_result, scenario
            )

    # Run global evaluation
    pii_df = span_evaluator.create_global_entities_df(df)
    result = span_evaluator.calculate_score_on_df(
        per_type=False, results_df=pii_df, evaluation_result=result
    )

    # Check global metrics
    assert_metric(
        expected_global_metrics["precision"], "pii_precision", result, scenario
    )
    assert_metric(expected_global_metrics["recall"], "pii_recall", result, scenario)
    assert_metric(expected_global_metrics["f_beta"], "pii_f", result, scenario)


# Test per-token IoU
def test_calculate_iou_token_based():
    """Test IoU calculation with token-based evaluation."""
    span1 = Span(
        entity_type="PERSON",
        entity_value="John Smith",
        start_position=0,
        end_position=10,
        normalized_tokens=["john", "smith"],
        normalized_start_index=0,
        normalized_end_index=10,
    )

    # Same tokens
    span2 = Span(
        entity_type="PERSON",
        entity_value="John Smith",
        start_position=0,
        end_position=10,
        normalized_tokens=["john", "smith"],
        normalized_start_index=0,
        normalized_end_index=10,
    )

    # Subset of tokens
    span3 = Span(
        entity_type="PERSON",
        entity_value="John",
        start_position=0,
        end_position=4,
        normalized_tokens=["john"],
        normalized_start_index=0,
        normalized_end_index=4,
    )

    # Different tokens
    span4 = Span(
        entity_type="PERSON",
        entity_value="Mary",
        start_position=15,
        end_position=19,
        normalized_tokens=["mary"],
        normalized_start_index=15,
        normalized_end_index=19,
    )
    span_evaluator = SpanEvaluator(
        model=MockModel(), 
        entity_mapping={},
        iou_threshold=0.75, 
        char_based=False, 
        skip_words=[]
    )
    # Test token-based IoU calculations for individual spans
    iou_exact = span_evaluator.calculate_iou(span1, span2, char_based=False)
    iou_partial = span_evaluator.calculate_iou(span1, span3, char_based=False)
    iou_none = span_evaluator.calculate_iou(span1, span4, char_based=False)

    assert iou_exact > 0.5  # Should be high for exact match
    assert 0 < iou_partial < iou_exact  # Should be lower for partial match
    assert iou_none == 0.0  # Should be zero for no overlap

    # Test cases for overlapping prediction spans per annotation span

    # Case 1: Multiple prediction spans covering the same annotation (perfect overlap)
    annotation_full = Span(
        entity_type="ORGANIZATION",
        entity_value="New York Mets",
        start_position=4,
        end_position=17,
        normalized_tokens=["new", "york", "mets"],
        normalized_start_index=4,
        normalized_end_index=17,
    )

    pred_span_1 = Span(
        entity_type="ORGANIZATION",
        entity_value="New",
        start_position=4,
        end_position=7,
        normalized_tokens=["new"],
        normalized_start_index=4,
        normalized_end_index=7,
    )

    pred_span_2 = Span(
        entity_type="ORGANIZATION",
        entity_value="York Mets",
        start_position=8,
        end_position=17,
        normalized_tokens=["york", "mets"],
        normalized_start_index=8,
        normalized_end_index=17,
    )

    # Combined IoU should be 1.0 (perfect coverage)
    combined_iou_perfect = span_evaluator._calculate_combined_iou(
        annotation_full, [pred_span_1, pred_span_2]
    )
    assert (
        combined_iou_perfect == 1.0
    ), f"Expected perfect IoU 1.0, got {combined_iou_perfect}"

    # Case 2: Multiple prediction spans with partial overlap
    pred_span_partial = Span(
        entity_type="ORGANIZATION",
        entity_value="New",
        start_position=4,
        end_position=7,
        normalized_tokens=["new"],
        normalized_start_index=4,
        normalized_end_index=7,
    )

    # Only covers "new" out of "new york mets"
    combined_iou_partial = span_evaluator._calculate_combined_iou(
        annotation_full, [pred_span_partial]
    )
    expected_partial_iou = 1 / 3  # 1 token intersection / 3 tokens union
    assert (
        combined_iou_partial == expected_partial_iou
    ), f"Expected IoU {expected_partial_iou}, got {combined_iou_partial}"

    # Case 3: Multiple prediction spans with extra tokens (lower IoU)
    annotation_simple = Span(
        entity_type="PERSON",
        entity_value="John Smith",
        start_position=0,
        end_position=10,
        normalized_tokens=["john", "smith"],
        normalized_start_index=0,
        normalized_end_index=10,
    )

    pred_extra_1 = Span(
        entity_type="PERSON",
        entity_value="John",
        start_position=0,
        end_position=4,
        normalized_tokens=["john"],
        normalized_start_index=0,
        normalized_end_index=4,
    )

    pred_extra_2 = Span(
        entity_type="PERSON",
        entity_value="Smith Jr",
        start_position=5,
        end_position=13,
        normalized_tokens=["smith", "jr"],
        normalized_start_index=5,
        normalized_end_index=13,
    )

    # IoU = 2 (john, smith) / 3 (john, smith, jr) = 2/3
    combined_iou_extra = span_evaluator._calculate_combined_iou(
        annotation_simple, [pred_extra_1, pred_extra_2]
    )
    expected_extra_iou = 2 / 3
    assert (
        abs(combined_iou_extra - expected_extra_iou) < 0.001
    ), f"Expected IoU {expected_extra_iou}, got {combined_iou_extra}"

    # Case 4: Empty prediction spans list
    combined_iou_empty = span_evaluator._calculate_combined_iou(annotation_simple, [])
    assert (
        combined_iou_empty == 0.0
    ), f"Expected IoU 0.0 for empty predictions, got {combined_iou_empty}"


# Test error analysis


@pytest.mark.parametrize(
    "scenario, annotation, prediction, tokens, start_indices, expected_error_types, expected_error_count, expected_explanations, expected_confusion_matrix",
    [
        # No overlapping predictions (FN)
        (
            "No overlap: Single annotation with no predictions",
            ["O", "PERSON", "PERSON", "O"],
            ["O", "O", "O", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            [ErrorType.FN],
            1,
            ["Entity PERSON not detected."],
            {("PERSON", "O"): 1},
        ),
        (
            "No overlap: Multiple annotations with no predictions",
            ["PERSON", "PERSON", "O", "LOCATION", "O"],
            ["O", "O", "O", "O", "O"],
            ["John", "Smith", "visited", "Boston", "today"],
            [0, 5, 11, 19, 26],
            [ErrorType.FN, ErrorType.FN],
            2,
            ["Entity PERSON not detected.", "Entity LOCATION not detected."],
            {("PERSON", "O"): 1, ("LOCATION", "O"): 1},
        ),
        # Single overlapping prediction: Same type, high IoU → TP (no error)
        (
            "Single overlap TP: Same type with high IoU",
            ["O", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "PERSON", "O"],
            ["The", "John", "Smith", "visited"],
            [0, 4, 9, 15],
            [],  # No errors expected for TP
            0,
            [],
            {("PERSON", "PERSON"): 1},
        ),
        # Single overlapping prediction: Different type, high IoU → WrongEntity
        (
            "Single overlap WrongEntity: Different types with high IoU",
            ["O", "LOCATION", "LOCATION", "O"],
            ["O", "PERSON", "PERSON", "O"],
            ["The", "New", "York", "visited"],
            [0, 4, 8, 13],
            [ErrorType.FN, ErrorType.FP, ErrorType.WrongEntity],
            3,
            ["Wrong entity type: LOCATION detected as PERSON"],
            {("LOCATION", "PERSON"): 1},
        ),
        # Single overlapping prediction: Same type, low IoU → FN
        (
            "Single overlap FN+FP: Same type with low IoU",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "O", "O", "O"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 15, 23],
            [ErrorType.FN, ErrorType.FP],
            2,
            ["Entity PERSON not detected due to low iou",
             "Entity PERSON falsely detected"],
            {("PERSON", "O"): 1, ("O", "PERSON"): 1},
        ),
        # Single overlapping prediction: Different type, low IoU → FN + FP
        (
            "Single overlap FN+FP: Different types with low IoU",
            ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["O", "PERSON", "O", "O", "O"],
            ["The", "New", "York", "Mets", "visited"],
            [0, 4, 8, 13, 18],
            [ErrorType.FN, ErrorType.FP],
            2,
            [
                "Entity ORGANIZATION not detected. iou with PERSON=",
                "Entity PERSON falsely detected",
            ],
            {("ORGANIZATION", "O"): 1, ("O", "PERSON"): 1},
        ),
        # Multiple overlapping predictions: Same type, low cumulative IoU due to skip words → FN
        (
            "Multiple overlap TP: Same type with high cumulative IoU",
            ["ADDRESS", "ADDRESS", "ADDRESS", "ADDRESS", "ADDRESS", "O"],
            ["ADDRESS", "O", "ADDRESS", "ADDRESS", "ADDRESS", "O"],
            ["123", "Main", "Street", "Suite", "100", "is"],
            [0, 4, 9, 16, 22, 26],
            [ErrorType.FN, ErrorType.FP],
            2,
            ["Entity ADDRESS not detected due to low iou=",
             "Entity ADDRESS falsely detected"],
            {("ADDRESS", "O"): 1, ("O", "ADDRESS"): 1},
        ),
        # Multiple overlapping predictions: Same type, low cumulative IoU → FN
        (
            "Multiple overlap FN: Same type with low cumulative IoU",
            ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"],
            ["O", "ORGANIZATION", "O", "ORGANIZATION", "O"],
            ["The", "New", "York", "Mets", "visited"],
            [0, 4, 8, 13, 18],
            [ErrorType.FN, ErrorType.FP],
            2,
            ["Entity ORGANIZATION not detected due to low iou",
             "Entity ORGANIZATION falsely detected"],
            {("ORGANIZATION", "O"): 1, ("O", "ORGANIZATION"): 1},
        ),
        # Multiple overlapping predictions: Different type, high cumulative IoU → WrongEntity
        (
            "Multiple overlap WrongEntity: Different types with high cumulative IoU",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "LOCATION", "LOCATION", "O"],
            ["The", "John", "Smith", "Johnson", "visited"],
            [0, 4, 9, 17, 25],
            [ErrorType.FN, ErrorType.WrongEntity, ErrorType.FP, ErrorType.FP],
            4,
            [
                "Entity PERSON not detected due to low iou",
                "Entity PERSON falsely detected",
                "Wrong entity type: PERSON detected as LOCATION",
                "Entity LOCATION falsely detected",
            ],
            {("PERSON", "LOCATION"): 1, ("O", "PERSON"): 1, ("PERSON", "O"): 1},
        ),
        # Multiple overlapping predictions: Different type, low cumulative IoU → FN + FP
        (
            "Multiple overlap FN+FP: Different types with low cumulative IoU",
            ["O", "PERSON", "PERSON", "PERSON", "O", "O"],
            ["O", "O", "LOCATION", "O", "ORGANIZATION", "O"],
            ["The", "John", "Smith", "Johnson", "works", "hard"],
            [0, 4, 9, 17, 25, 31],
            [ErrorType.FN, ErrorType.FP, ErrorType.FP],
            3,
            [
                "Entity PERSON not detected. iou with LOCATION=",
                "Entity LOCATION falsely detected",
                "False prediction with no overlap: ORGANIZATION",
            ],
            {("PERSON", "O"): 1, ("O", "LOCATION"): 1, ("O", "ORGANIZATION"): 1},
        ),
        # Multiple overlapping predictions: Mixed types scenario - both same and different types overlapping
        (
            "Multiple overlap mixed: Both same and different entity types",
            ["O", "PERSON", "PERSON", "PERSON", "PERSON", "O"],
            ["O", "PERSON", "O", "LOCATION", "PERSON", "O"],
            ["The", "John", "C", "B", "Johnson", "visited"],
            [0, 4, 9, 11, 13, 21],
            [
                ErrorType.FP,
            ],  # Only FP for LOCATION since cumulative PERSON IoU is good enough
            1,
            ["Entity LOCATION falsely detected"],
            {("PERSON", "PERSON"): 1, ("O", "LOCATION"): 1},
        ),
        # Standalone false positives: Single standalone FP
        (
            "Standalone FP: Single prediction with no annotation overlap",
            ["O", "O", "O", "O"],
            ["O", "PERSON", "PERSON", "O"],
            ["The", "quick", "brown", "fox"],
            [0, 4, 10, 16],
            [ErrorType.FP],
            1,
            ["False prediction with no overlap: PERSON"],
            {("O", "PERSON"): 1},
        ),
        # Standalone false positives: Multiple standalone FPs
        (
            "Multiple standalone FPs: Multiple predictions with no annotation overlap",
            ["O", "O", "O", "O", "O"],
            ["PERSON", "O", "LOCATION", "LOCATION", "O"],
            ["John", "went", "to", "New", "York"],
            [0, 5, 10, 13, 17],
            [ErrorType.FP, ErrorType.FP],
            2,
            [
                "False prediction with no overlap: PERSON",
                "False prediction with no overlap: LOCATION",
            ],
            {("O", "PERSON"): 1, ("O", "LOCATION"): 1},
        ),
        # Complex mixed scenarios: Combination of all error types
        (
            "Complex scenario: Mixed overlaps and standalone predictions",
            ["PERSON", "O", "LOCATION", "LOCATION", "LOCATION", "O"],
            ["O", "O", "LOCATION", "PERSON", "PERSON", "PHONE_NUMBER"],
            ["Alice", "went", "to", "New", "York", "today"],
            [0, 6, 11, 14, 18, 23],
            [
                ErrorType.FN,
                ErrorType.FN,
                ErrorType.FP,
                ErrorType.WrongEntity,
                ErrorType.FP,
            ],
            5,
            [
                "Entity PERSON not detected.",
                "Wrong entity type: LOCATION detected as PERSON",
                "Entity LOCATION not detected. iou with PERSON=1.00",
                "Entity PERSON falsely detected",
                "False prediction with no overlap: PHONE_NUMBER",
            ],
            {("PERSON", "O"): 1, ("LOCATION", "PERSON"): 1, ("O", "PHONE_NUMBER"): 1},
        ),
    ],
)
def test_match_predictions_with_annotations_error_generation(
    span_evaluator,
    scenario,
    annotation,
    prediction,
    tokens,
    start_indices,
    expected_error_types,
    expected_error_count,
    expected_explanations,
    expected_confusion_matrix,
):
    """
    Test error generation in _match_predictions_with_annotations method covering all scenarios:

    1. No overlapping predictions → FN errors
    2. Single overlapping predictions → TP, WrongEntity, FN, or FN+FP errors
    3. Multiple overlapping predictions → TP, WrongEntity, FN, or FN+FP errors
    4. Standalone predictions → FP errors
    5. Complex mixed scenarios
    """
    # Build the DataFrame expected by SpanEvaluator
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "prediction": prediction,
            "start_indices": start_indices,
        }
    )

    # Run evaluation
    result = EvaluationResult()
    result = span_evaluator.calculate_score_on_df(
        per_type=True, results_df=df, evaluation_result=result
    )

    # Check that expected error types are present
    error_types = [error.error_type for error in result.model_errors]

    # Verify each expected error type is present
    for expected_type in expected_error_types:
        assert (
            expected_type in error_types
        ), f"In {scenario}, expected error type {expected_type} not found in {error_types}"

    # Check that we have the expected number of errors
    assert (
        len(result.model_errors) == expected_error_count
    ), f"In {scenario}, expected {expected_error_count} errors, got {len(result.model_errors)}"

    # Count occurrences of each error type
    error_type_counts = {}
    for error_type in error_types:
        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

    expected_type_counts = {}
    for expected_type in expected_error_types:
        expected_type_counts[expected_type] = (
            expected_type_counts.get(expected_type, 0) + 1
        )

    for expected_type, expected_count in expected_type_counts.items():
        actual_count = error_type_counts.get(expected_type, 0)
        assert (
            actual_count == expected_count
        ), f"In {scenario}, expected {expected_count} errors of type {expected_type}, got {actual_count}"

    # Check error explanations contain expected text
    for i, expected_explanation in enumerate(expected_explanations):
        if i < len(result.model_errors):
            actual_explanation = result.model_errors[i].explanation
            assert (
                expected_explanation in actual_explanation
            ), f"In {scenario}, expected explanation to contain '{expected_explanation}', got '{actual_explanation}'"

    # Validate confusion matrix entries
    for (
        expected_ann,
        expected_pred,
    ), expected_count in expected_confusion_matrix.items():
        actual_count = result.results.get((expected_ann, expected_pred), 0)
        assert (
            actual_count == expected_count
        ), f"In {scenario}, confusion matrix entry ({expected_ann}, {expected_pred}) expected {expected_count}, got {actual_count}"


# Test span creation and skip words handling


@pytest.mark.parametrize(
    "scenario, annotation, tokens, start_indices, skip_words, expected_spans_after_processing",
    [
        # Adjacent spans that should be merged (same type)
        (
            "Adjacent merging: Same type spans separated by skip words",
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["John", "Smith", "and", "Jane", "Doe"],
            [0, 5, 11, 15, 20],
            ["and"],
            [
                {
                    "entity_type": "PERSON",
                    "entity_value": "John Smith Jane Doe",
                    "start_position": 0,
                    "end_position": 23,
                    "normalized_tokens": ["john", "smith", "jane", "doe"],
                    "token_start": 0,
                    "token_end": 5,
                }
            ],
        ),
        # Adjacent spans that should NOT be merged (different types)
        (
            "No merging: Different entity types with skip words between",
            ["PERSON", "PERSON", "O", "LOCATION", "LOCATION"],
            ["John", "Smith", "visited", "New", "York"],
            [0, 5, 11, 19, 23],
            ["visited"],
            [
                {
                    "entity_type": "PERSON",
                    "entity_value": "John Smith",
                    "start_position": 0,
                    "end_position": 10,
                    "normalized_tokens": ["john", "smith"],
                    "token_start": 0,
                    "token_end": 2,
                },
                {
                    "entity_type": "LOCATION",
                    "entity_value": "New York",
                    "start_position": 19,
                    "end_position": 27,
                    "normalized_tokens": ["new", "york"],
                    "token_start": 3,
                    "token_end": 5,
                },
            ],
        ),
        # Adjacent spans separated by non-skip words (no merging)
        (
            "No merging: Same type spans separated by non-skip words",
            ["PERSON", "PERSON", "O", "PERSON", "PERSON"],
            ["John", "Smith", "visited", "Jane", "Doe"],
            [0, 5, 11, 19, 24],
            [],  # "visited" is not a skip word
            [
                {
                    "entity_type": "PERSON",
                    "entity_value": "John Smith",
                    "start_position": 0,
                    "end_position": 10,
                    "normalized_tokens": ["john", "smith"],
                    "token_start": 0,
                    "token_end": 2,
                },
                {
                    "entity_type": "PERSON",
                    "entity_value": "Jane Doe",
                    "start_position": 19,
                    "end_position": 27,
                    "normalized_tokens": ["jane", "doe"],
                    "token_start": 3,
                    "token_end": 5,
                },
            ],
        ),
        # Skip words within entities
        (
            "Skip words in entity: Entity tokens with skip words removed",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["The", "Mr.", "John", "Smith", "visited"],
            [0, 4, 8, 13, 19],
            ["mr."],
            [
                {
                    "entity_type": "PERSON",
                    "entity_value": "Mr. John Smith",
                    "start_position": 4,
                    "end_position": 18,
                    "normalized_tokens": ["john", "smith"],  # "mr." should be skipped
                    "token_start": 1,
                    "token_end": 4,
                }
            ],
        ),
        # Complex merging with multiple skip words
        (
            "Complex merging: Multiple spans with various skip word separators",
            [
                "ORGANIZATION",
                "ORGANIZATION",
                "O",
                "ORGANIZATION",
                "ORGANIZATION",
                "O",
                "ORGANIZATION",
            ],
            ["Apple", "Inc.", "Ltd.", "Google", "LLC", "and", "Microsoft"],
            [0, 6, 11, 16, 24, 29, 33],
            ["ltd.", "and"],
            [
                {
                    "entity_type": "ORGANIZATION",
                    "entity_value": "Apple Inc. Google LLC Microsoft",
                    "start_position": 0,
                    "end_position": 42,
                    "normalized_tokens": [
                        "apple",
                        "inc.",
                        "google",
                        "llc",
                        "microsoft",
                    ],
                    "token_start": 0,
                    "token_end": 7,
                }
            ],
        ),
        # Entity entirely composed of skip words (should be filtered out)
        (
            "Skip words only: Entity entirely of skip words gets filtered",
            ["O", "PERSON", "PERSON", "O"],
            ["The", "the", "and", "visited"],
            [0, 4, 8, 12],
            ["the", "and"],
            [],  # Should create no spans since all entity tokens are skip words
        ),
        # Multiple entities with various skip word scenarios
        (
            "Mixed scenarios: Multiple entities with different skip word patterns",
            [
                "PERSON",
                "PERSON",
                "O",
                "O",
                "LOCATION",
                "LOCATION",
                "LOCATION",
                "O",
                "ORGANIZATION",
            ],
            [
                "Dr.",
                "John",
                "visited",
                "the",
                "New",
                "York",
                "City",
                "and",
                "Microsoft",
            ],
            [0, 4, 9, 17, 21, 25, 30, 35, 39],
            ["dr.", "the", "and"],
            [
                {
                    "entity_type": "PERSON",
                    "entity_value": "Dr. John",
                    "start_position": 0,
                    "end_position": 8,
                    "normalized_tokens": ["john"],  # "dr." skipped
                    "token_start": 0,
                    "token_end": 2,
                },
                {
                    "entity_type": "LOCATION",
                    "entity_value": "New York City",
                    "start_position": 21,
                    "end_position": 34,
                    "normalized_tokens": ["new", "york", "city"],
                    "token_start": 4,
                    "token_end": 7,
                },
                {
                    "entity_type": "ORGANIZATION",
                    "entity_value": "Microsoft",
                    "start_position": 39,
                    "end_position": 48,
                    "normalized_tokens": ["microsoft"],
                    "token_start": 8,
                    "token_end": 9,
                },
            ],
        ),
        # Merging with consecutive skip words
        (
            "Consecutive skip words merging: Multiple skip words between spans",
            ["LOCATION", "LOCATION", "O", "O", "O", "LOCATION", "LOCATION"],
            ["New", "York", "and", "NY", "and", "United", "States"],
            [0, 4, 9, 11, 14, 16, 23],
            ["and", "ny"],
            [
                {
                    "entity_type": "LOCATION",
                    "entity_value": "New York United States",
                    "start_position": 0,
                    "end_position": 29,
                    "normalized_tokens": ["new", "york", "united", "states"],
                    "token_start": 0,
                    "token_end": 7,
                }
            ],
        ),
        # Case sensitivity in skip words
        (
            "Case sensitivity: Skip words work with different cases",
            ["O", "PERSON", "PERSON", "PERSON", "PERSON", "PERSON"],
            ["THE", "John", "Smith", "AND", "Jane", "Doe"],
            [0, 4, 9, 15, 19, 24],
            ["the", "and"],  # lowercase skip words should match uppercase tokens
            [
                {
                    "entity_type": "PERSON",
                    "entity_value": "John Smith AND Jane Doe",
                    "start_position": 4,
                    "end_position": 27,
                    "normalized_tokens": [
                        "john",
                        "smith",
                        "jane",
                        "doe",
                    ],  # Skip words removed
                    "token_start": 1,
                    "token_end": 6,
                }
            ],
        ),
        # Empty skip words list (use default list)
        (
            "No skip words: All tokens preserved in normalization",
            ["O", "PERSON", "PERSON", "PERSON", "O"],
            ["The", "Dr.", "John", "Smith", "visited"],
            [0, 4, 8, 13, 19],
            [],
            [
                {
                    "entity_type": "PERSON",
                    "entity_value": "Dr. John Smith",
                    "start_position": 4,
                    "end_position": 18,
                    "normalized_tokens": [
                        "dr.",
                        "john",
                        "smith",
                    ],  # All tokens preserved
                    "token_start": 1,
                    "token_end": 4,
                }
            ],
        ),
    ],
)
def test_span_creation_with_skip_words(
    scenario,
    annotation,
    tokens,
    start_indices,
    skip_words,
    expected_spans_after_processing,
):
    """
    Test span creation and adjacent span merging functionality with skip words.

    This test covers:
    1. Basic span creation from token sequences
    2. Skip word normalization within entities
    3. Adjacent span merging when separated by skip words
    4. Prevention of merging when spans are different types or separated by non-skip words
    5. Filtering out entities that are entirely composed of skip words
    6. Complex scenarios with multiple entities and various skip word patterns
    """
    # Create evaluator with specific skip words
    span_evaluator = SpanEvaluator(
        model=MockModel(), 
        entity_mapping={},
        iou_threshold=0.75, 
        char_based=True, 
        skip_words=skip_words
    )

    # Build the DataFrame
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(tokens),
            "token": tokens,
            "annotation": annotation,
            "start_indices": start_indices,
        }
    )

    # Process spans (create spans then merge adjacent ones - this is what happens in evaluation)
    annotation_spans = span_evaluator._create_spans(df=df, column="annotation")
    annotation_spans = span_evaluator._merge_adjacent_spans(
        spans=annotation_spans, df=df
    )
    # Check number of spans after processing
    assert (
        len(annotation_spans) == len(expected_spans_after_processing)
    ), f"In {scenario}, expected {len(expected_spans_after_processing)} spans after processing, got {len(annotation_spans)}"

    # Check each span's properties after processing
    for i, expected_span in enumerate(expected_spans_after_processing):
        actual_span = annotation_spans[i]

        assert (
            actual_span.entity_type == expected_span["entity_type"]
        ), f"In {scenario}, span {i} entity_type expected {expected_span['entity_type']}, got {actual_span.entity_type}"

        assert (
            actual_span.entity_value == expected_span["entity_value"]
        ), f"In {scenario}, span {i} entity_value expected {expected_span['entity_value']}, got {actual_span.entity_value}"

        assert (
            actual_span.start_position == expected_span["start_position"]
        ), f"In {scenario}, span {i} start_position expected {expected_span['start_position']}, got {actual_span.start_position}"

        assert (
            actual_span.end_position == expected_span["end_position"]
        ), f"In {scenario}, span {i} end_position expected {expected_span['end_position']}, got {actual_span.end_position}"

        assert (
            actual_span.normalized_tokens == expected_span["normalized_tokens"]
        ), f"In {scenario}, span {i} normalized_tokens expected {expected_span['normalized_tokens']}, got {actual_span.normalized_tokens}"

        assert (
            actual_span.token_start == expected_span["token_start"]
        ), f"In {scenario}, span {i} token_start expected {expected_span['token_start']}, got {actual_span.token_start}"

        assert (
            actual_span.token_end == expected_span["token_end"]
        ), f"In {scenario}, span {i} token_end expected {expected_span['token_end']}, got {actual_span.token_end}"


# ── Integration tests: entity_mapping + calculate_score pipeline ─────────────


def test_entity_mapping_end_to_end():
    """
    Integration test that mirrors the full evaluation flow in notebook 5:

      Dataset labels (GPE, STREET_ADDRESS)  ->  entity_mapping  ->  model namespace (LOCATION)
      Model predictions already in model namespace (LOCATION, PERSON)

    The mock model always returns Presidio-style predictions (LOCATION, PERSON),
    simulating what PresidioAnalyzerWrapper returns after its internal
    model-to-Presidio remapping.

    Full pipeline exercised:
      InputSample (dataset tags)
        -> evaluate_all()           [runs model, stores raw tags in EvaluationResult]
        -> calculate_score()
            -> get_results_dataframe()   [normalises annotations via entity_mapping]
            -> _create_spans()           [must NOT remap again]
            -> _find_best_match()        [span IoU matching in model namespace]
        -> final EvaluationResult with per_type metrics

    Assertions:
      1. Dataset labels (GPE, STREET_ADDRESS) do not appear in get_results_dataframe()
         annotation column -- they must be normalised to model entity names.
      2. per_type results are keyed by model entity names (LOCATION, PERSON).
      3. Perfectly-predicted spans produce TPs, not FNs (entity_mapping applied once).
      4. Overall pii_recall > 0.
    """
    from presidio_evaluator import InputSample

    # Dataset uses its own vocabulary (GPE); model returns Presidio vocabulary (LOCATION, PERSON)
    tokens = ["Alice", "visited", "New", "York", "yesterday"]
    dataset_tags = ["PERSON", "O", "GPE", "GPE", "O"]
    predicted_tags = ["PERSON", "O", "LOCATION", "LOCATION", "O"]
    start_indices = [0, 6, 14, 18, 23]

    # Mirrors EntityMappingHelper output from notebook 5:
    # dataset entity names -> model (Presidio) entity names
    entity_mapping = {
        "GPE": "LOCATION",
        "STREET_ADDRESS": "LOCATION",
        "PERSON": "PERSON",
    }
    entities_to_keep = ["LOCATION", "PERSON"]

    sample = InputSample(
        full_text="Alice visited New York yesterday",
        tokens=tokens,
        tags=dataset_tags,
        start_indices=start_indices,
    )

    model = MockTokensModel(
        prediction=predicted_tags,
        entities_to_keep=entities_to_keep,
    )
    evaluator = SpanEvaluator(
        model=model,
        entity_mapping=entity_mapping,
        entities_to_keep=entities_to_keep,
        skip_words=[],
        iou_threshold=0.5,
    )

    # ── Full pipeline ──────────────────────────────────────────────────────────
    evaluation_results = evaluator.evaluate_all([sample])
    result = evaluator.calculate_score(evaluation_results)

    # 1. Annotation column must contain model entity names, not dataset names
    df = evaluator.get_results_dataframe(evaluation_results)
    annotation_values = set(df["annotation"].unique())
    assert "GPE" not in annotation_values, (
        f"Dataset label 'GPE' leaked into annotation column: {annotation_values}"
    )
    assert "STREET_ADDRESS" not in annotation_values, (
        f"Dataset label 'STREET_ADDRESS' leaked into annotation column: {annotation_values}"
    )
    assert "LOCATION" in annotation_values and "PERSON" in annotation_values, (
        f"Expected model entity names in annotation column, got: {annotation_values}"
    )

    # 2. Per-type results keyed by model entity names
    assert "LOCATION" in result.per_type, (
        f"Expected 'LOCATION' in per_type results, got: {list(result.per_type.keys())}"
    )
    assert "GPE" not in result.per_type, (
        "Dataset label 'GPE' must not appear as a per_type key"
    )

    # 3. Perfectly-predicted spans are TPs, not FNs (entity_mapping applied exactly once)
    location_metrics = result.per_type["LOCATION"]
    assert location_metrics.true_positives == 1, (
        f"Expected 1 TP for LOCATION (GPE->LOCATION), got {location_metrics.true_positives}"
    )
    assert location_metrics.false_negatives == 0, (
        f"Expected 0 FNs for LOCATION, got {location_metrics.false_negatives}"
    )

    person_metrics = result.per_type["PERSON"]
    assert person_metrics.true_positives == 1, (
        f"Expected 1 TP for PERSON, got {person_metrics.true_positives}"
    )

    # 4. Overall recall > 0
    assert result.pii_recall > 0, (
        "pii_recall is 0 -- entity_mapping may have been applied more than once, "
        "putting annotation spans in a different namespace than predictions."
    )

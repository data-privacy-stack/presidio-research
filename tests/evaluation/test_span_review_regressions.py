"""Regressions for span identity and prediction accounting."""

import pandas as pd
import pytest

from presidio_evaluator.entity_mapping import CanonicalMapper
from presidio_evaluator.entity_mapping.data_objects import (
    ANNOTATION_SPAN_ID,
    PREDICTION_SPAN_ID,
)
from presidio_evaluator.evaluation import ErrorType, EvaluationResult, SpanEvaluator
from tests.helpers import make_results_df as make_df


@pytest.mark.parametrize("column", ["pred_a", "custom_gold"])
def test_arbitrary_label_columns_are_rejected(column):
    df = make_df(["John", "Smith", "32"], ["NAME"] * 3, ["NAME", "NAME", "AGE"])
    df[column] = ["PII"] * 3
    with pytest.raises(ValueError, match="Unsupported label column"):
        SpanEvaluator(skip_words=[])._create_spans(df, column)


@pytest.mark.parametrize(
    ("column", "id_column"),
    [("annotation", ANNOTATION_SPAN_ID), ("prediction", PREDICTION_SPAN_ID)],
)
def test_missing_span_id_column_falls_back_to_label_runs(column, id_column):
    df = make_df(["John", "32", "Ana"], ["NAME", "AGE", "NAME"], ["NAME"] * 3).drop(
        columns=[id_column]
    )
    spans = SpanEvaluator(skip_words=[])._create_spans(df, column)
    expected = 3 if column == "annotation" else 1
    assert len(spans) == expected


def test_mapper_derives_ids_from_finest_labels_when_missing():
    raw = make_df(
        ["John", "Smith", "32", ",", "Ana"],
        ["NAME", "NAME", "AGE", "O", "NAME"],
        ["NAME", "NAME", "AGE", "O", "NAME"],
    ).drop(columns=[ANNOTATION_SPAN_ID, PREDICTION_SPAN_ID])
    mapper = CanonicalMapper()
    mapper.analyze(raw)
    mapped = mapper.get_mapped_results_dataframe()
    for level in ("binary", "branch", "detailed"):
        df = mapped.get_level(level)
        assert df[ANNOTATION_SPAN_ID].tolist() == [0, 0, 1, None, 2]
        gold, predictions = SpanEvaluator()._process_sentence_spans(df)
        assert len(gold) == len(predictions) == 3


def test_span_ids_use_sentence_positions_not_dataframe_index():
    df = make_df(["32", "John"], ["AGE", "NAME"], ["O", "O"])
    df.index = [100, 400]
    spans = SpanEvaluator(skip_words=[])._create_spans(df, "annotation")
    assert [(s.entity_value, s.token_start, s.token_end) for s in spans] == [
        ("32", 0, 1),
        ("John", 1, 2),
    ]


@pytest.mark.parametrize("char_based", [False, True])
@pytest.mark.parametrize("threshold", [0.5, 1.0])
@pytest.mark.parametrize("level", ["binary", "branch", "detailed"])
def test_one_prediction_over_two_gold_spans_counted_once(char_based, threshold, level):
    raw = make_df(["John", "Smith", "32"], ["NAME", "NAME", "AGE"], ["NAME"] * 3)
    mapper = CanonicalMapper()
    mapper.analyze(raw)
    df = mapper.get_mapped_results_dataframe().get_level(level)
    evaluator = SpanEvaluator(iou_threshold=threshold, char_based=char_based)
    gold, predictions = evaluator._process_sentence_spans(df)
    assert len(gold) == 2
    assert len(predictions) == 1

    result = evaluator.calculate_score_on_df(df, level="both")
    expected_tp = int(threshold == 0.5)
    assert result.pii_annotated == 2
    assert result.pii_predicted == 1
    assert result.pii_true_positives == expected_tp
    assert result.pii_false_positives == 1 - expected_tp
    assert result.pii_false_negatives == 2 - expected_tp
    assert result.pii_precision == expected_tp
    assert result.pii_recall == expected_tp / 2
    assert sum(m.num_predicted for m in result.per_type.values()) == 1
    assert sum(m.true_positives for m in result.per_type.values()) == expected_tp
    assert sum(m.false_positives for m in result.per_type.values()) == 1 - expected_tp
    assert sum(m.false_negatives for m in result.per_type.values()) == 2 - expected_tp

    entity_result = evaluator.calculate_score_on_df(df, level="entity")
    assert sum(e.error_type == ErrorType.FP for e in entity_result.model_errors) == (
        1 - expected_tp
    )
    assert sum(e.error_type == ErrorType.FN for e in entity_result.model_errors) == (
        2 - expected_tp
    )
    assert (
        sum(
            count for (ann, pred), count in entity_result.results.items() if pred != "O"
        )
        == 1
    )


@pytest.mark.parametrize("reverse_gold", [False, True])
@pytest.mark.parametrize("per_type", [False, True])
def test_small_early_overlap_cannot_steal_later_true_positive(reverse_gold, per_type):
    evaluator = SpanEvaluator(iou_threshold=0.75, skip_words=[])
    df = make_df(["32", "John", "Smith"], ["PII"] * 3, ["PII"] * 3)
    df[ANNOTATION_SPAN_ID] = [0, 1, 1]
    df[PREDICTION_SPAN_ID] = [0, 0, 0]
    gold, predictions = evaluator._process_sentence_spans(df)
    if reverse_gold:
        gold.reverse()
    result = evaluator._match_predictions_with_annotations(
        gold, predictions, EvaluationResult(), per_type=per_type
    )
    if per_type:
        metrics = result.per_type["PII"]
        assert (
            metrics.true_positives,
            metrics.false_positives,
            metrics.false_negatives,
        ) == (1, 0, 1)
        assert metrics.num_predicted == 1
    else:
        assert (
            result.pii_true_positives,
            result.pii_false_positives,
            result.pii_false_negatives,
        ) == (1, 0, 1)
        assert result.pii_predicted == 1


@pytest.mark.parametrize("reverse_gold", [False, True])
def test_prediction_cannot_be_true_positive_for_two_gold_spans(reverse_gold):
    evaluator = SpanEvaluator(iou_threshold=0.4, skip_words=[])
    df = make_df(["John", "Mary"], ["PII"] * 2, ["PII"] * 2)
    df[ANNOTATION_SPAN_ID] = [0, 1]
    df[PREDICTION_SPAN_ID] = [0, 0]
    gold, predictions = evaluator._process_sentence_spans(df)
    if reverse_gold:
        gold.reverse()
    result = evaluator._match_predictions_with_annotations(
        gold, predictions, EvaluationResult()
    )
    metrics = result.per_type["PII"]
    assert metrics.num_predicted == 1
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    missed = [e for e in result.model_errors if e.error_type == ErrorType.FN]
    assert [e.full_text for e in missed] == ["Mary"]


def test_qualifying_same_type_match_preferred_over_wrong_type_overlap():
    evaluator = SpanEvaluator(iou_threshold=0.25, skip_words=[])
    df = make_df(["32", "John"], ["AGE", "NAME"], ["AGE"] * 2)
    result = evaluator.calculate_score_on_df(df, level="entity")
    assert result.per_type["AGE"].true_positives == 1
    assert result.per_type["AGE"].num_predicted == 1
    assert result.per_type["AGE"].false_positives == 0
    assert result.per_type["NAME"].false_negatives == 1


@pytest.mark.parametrize("char_based", [False, True])
def test_shared_prediction_in_multiple_overlap_group_not_counted_again(char_based):
    raw = make_df(
        ["John", "Smith", "32", "44"],
        ["NAME", "NAME", "AGE", "AGE"],
        ["NAME", "NAME", "NAME", "AGE"],
    )
    mapper = CanonicalMapper()
    mapper.analyze(raw)
    df = mapper.get_mapped_results_dataframe().binary
    result = SpanEvaluator(
        iou_threshold=1.0, char_based=char_based, skip_words=[]
    ).calculate_score_on_df(df)
    assert result.pii_predicted == 2
    assert result.pii_false_positives == 2
    assert result.pii_false_negatives == 2
    assert result.per_type["PII"].num_predicted == 2
    assert result.per_type["PII"].false_positives == 2


def test_fragment_aggregation_still_counts_as_one_prediction():
    df = make_df(
        ["New", "York", "Mets"],
        ["ORGANIZATION"] * 3,
        ["ORGANIZATION", "O", "ORGANIZATION"],
    )
    result = SpanEvaluator(
        iou_threshold=0.5, char_based=False, skip_words=[]
    ).calculate_score_on_df(df, level="entity")
    metrics = result.per_type["ORGANIZATION"]
    assert metrics.true_positives == metrics.num_predicted == 1
    assert metrics.false_positives == metrics.false_negatives == 0


def test_mapper_passes_span_ids_through_every_level():
    raw = make_df(
        ["John", "Smith", ",", "32"],
        ["NAME", "NAME", "O", "AGE"],
        ["FIRSTNAME", "LASTNAME", "O", "AGE"],
    )
    before = raw.copy(deep=True)
    mapper = CanonicalMapper()
    mapper.analyze(raw)
    mapped = mapper.get_mapped_results_dataframe()
    for level in ("original", "binary", "branch", "detailed"):
        df = mapped.get_level(level)
        assert list(df.columns) == list(raw.columns)
        assert df[ANNOTATION_SPAN_ID].tolist() == [0, 0, None, 1]
        assert df[PREDICTION_SPAN_ID].tolist() == [0, 1, None, 2]
        if level != "original":
            gold, predictions = SpanEvaluator()._process_sentence_spans(df)
            assert len(gold) == 2
            assert len(predictions) == 3
    pd.testing.assert_frame_equal(raw, before)
    pd.testing.assert_frame_equal(mapped.original, raw)
    assert mapped.detailed["prediction"].tolist() == ["NAME", "NAME", "O", "AGE"]

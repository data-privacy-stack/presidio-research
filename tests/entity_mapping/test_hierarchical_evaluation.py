"""Tests for BaseEvaluator.calculate_hierarchical_scores()."""

import pandas as pd
import pytest

from presidio_evaluator.entity_mapping import CanonicalMapper, MappedResults
from presidio_evaluator.evaluation import EvaluationResult
from presidio_evaluator.evaluation.span_evaluator import SpanEvaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_evaluator = SpanEvaluator()


def _make_results(annotations: list[str], predictions: list[str]) -> MappedResults:
    """Build MappedResults via the mapper from raw annotation/prediction lists."""
    n = max(len(annotations), len(predictions))
    annotations = list(annotations) + ["O"] * (n - len(annotations))
    predictions = list(predictions) + ["O"] * (n - len(predictions))
    df = pd.DataFrame(
        {
            "sentence_id": list(range(n)),
            "token": [f"tok{i}" for i in range(n)],
            "annotation": annotations,
            "prediction": predictions,
            "start_indices": list(range(n)),
        }
    )
    mapper = CanonicalMapper()
    mapper.analyze(df)
    return mapper.get_mapped_results_dataframe()


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dict_with_three_keys(self):
        results = _make_results(["NAME", "O"], ["NAME", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert set(scores.keys()) == {"binary", "branch", "detailed"}

    def test_each_value_is_evaluation_result(self):
        results = _make_results(["NAME", "O"], ["NAME", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        for level in ("binary", "branch", "detailed"):
            assert isinstance(scores[level], EvaluationResult)

    def test_all_O_df(self):
        """No PII at all — function must not raise."""
        results = _make_results(["O", "O"], ["O", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert set(scores.keys()) == {"binary", "branch", "detailed"}

    def test_raises_for_plain_dataframe(self):
        """Passing a plain DataFrame should raise TypeError."""
        df = pd.DataFrame({"annotation": ["NAME"], "prediction": ["NAME"]})
        with pytest.raises(TypeError, match="MappedResults"):
            _evaluator.calculate_hierarchical_scores(df)


# ---------------------------------------------------------------------------
# binary — PII vs. O
# ---------------------------------------------------------------------------


class TestBinaryLevel:
    def test_perfect_pii_detection(self):
        """Model correctly identifies all PII tokens."""
        results = _make_results(
            ["NAME", "EMAIL_ADDRESS", "O"],
            ["NAME", "EMAIL_ADDRESS", "O"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["binary"].pii_f == pytest.approx(1.0, abs=1e-6)

    def test_collapses_different_entity_types_to_pii(self):
        """NAME annotation matched by EMAIL_ADDRESS prediction = TP at binary level."""
        results = _make_results(["NAME", "O"], ["EMAIL_ADDRESS", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["binary"].pii_recall > 0


# ---------------------------------------------------------------------------
# branch — branch-level (depth-2)
# ---------------------------------------------------------------------------


class TestBranchLevel:
    def test_depth3_maps_to_depth2_ancestor(self):
        """NAME (depth-3, branch PERSON) should map to PERSON at branch level."""
        results = _make_results(["NAME", "O"], ["NAME", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["branch"].pii_precision == pytest.approx(1.0, abs=1e-6)

    def test_different_branches_are_different(self):
        """NAME (PERSON branch) vs EMAIL_ADDRESS (CONTACT branch) differ at branch level."""
        results = _make_results(["NAME", "O"], ["EMAIL_ADDRESS", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        person_metrics = scores["branch"].per_type.get("PERSON")
        assert person_metrics is not None
        assert person_metrics.recall == pytest.approx(0.0, abs=1e-6)

    def test_depth2_entity_unchanged(self):
        """PERSON (depth-2) should map to PERSON at branch level unchanged."""
        results = _make_results(["PERSON", "O"], ["PERSON", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["branch"].pii_f == pytest.approx(1.0, abs=1e-6)

    def test_mixed_depth_same_branch(self):
        """NAME annotation and PERSON prediction — same PERSON branch at branch level."""
        results = _make_results(["NAME"] * 5, ["PERSON"] * 5)
        scores = _evaluator.calculate_hierarchical_scores(results)
        person_branch = scores["branch"].per_type.get("PERSON")
        assert person_branch is not None
        assert person_branch.recall == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# detailed — canonical surface (used as-is from MappedResults)
# ---------------------------------------------------------------------------


class TestDetailedLevel:
    def test_uses_resolved_labels(self):
        """detailed uses the resolved hierarchy node."""
        results = _make_results(["NAME", "O"], ["NAME", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["detailed"].pii_f == pytest.approx(1.0, abs=1e-6)

    def test_mismatch_penalised(self):
        """PERSON predicting on NAME annotation creates FP+FN at detailed level."""
        results = _make_results(["NAME", "O"], ["PERSON", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        name_metrics = scores["detailed"].per_type.get("NAME")
        assert name_metrics is not None
        assert name_metrics.recall == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Granularity bonus property
# ---------------------------------------------------------------------------


class TestGranularityBonus:
    def test_specific_prediction_scores_tp_at_all_levels(self):
        """Model predicting NAME (depth-3) on NAME annotation gets TP at all levels."""
        results = _make_results(["NAME", "O"], ["NAME", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        for level in ("binary", "branch", "detailed"):
            assert scores[level].pii_f == pytest.approx(1.0, abs=1e-6), (
                f"Expected perfect score at {level}"
            )

    def test_coarse_prediction_scores_tp_at_binary_branch_but_not_detailed(self):
        """Model predicting PERSON (depth-2) on NAME: TP at binary/branch but not detailed."""
        results = _make_results(["NAME", "O"], ["PERSON", "O"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["binary"].pii_recall > 0
        person_branch = scores["branch"].per_type.get("PERSON")
        assert person_branch is not None
        assert person_branch.recall == pytest.approx(1.0, abs=1e-6)
        name_detailed = scores["detailed"].per_type.get("NAME")
        assert name_detailed is not None
        assert name_detailed.recall == pytest.approx(0.0, abs=1e-6)

    def test_calculate_score_on_df_matches_branch_level(self):
        """calculate_score_on_df(results.branch) == scores['branch']."""
        results = _make_results(
            ["NAME", "EMAIL_ADDRESS", "O"], ["NAME", "EMAIL_ADDRESS", "O"]
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        direct = _evaluator.calculate_score_on_df(results.branch)
        assert scores["branch"].pii_f == pytest.approx(direct.pii_f, abs=1e-6)

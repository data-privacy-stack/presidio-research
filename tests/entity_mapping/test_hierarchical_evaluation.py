"""Tests for calculate_hierarchical_scores()."""

import pandas as pd
import pytest

from presidio_evaluator.entity_mapping import (
    EntityHierarchy,
    calculate_hierarchical_scores,
)
from presidio_evaluator.evaluation import EvaluationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mapped_df(annotations: list[str], predictions: list[str]) -> pd.DataFrame:
    """Build a minimal DataFrame matching get_mapped_results_dataframe() output."""
    n = max(len(annotations), len(predictions))
    annotations = list(annotations) + ["O"] * (n - len(annotations))
    predictions = list(predictions) + ["O"] * (n - len(predictions))
    return pd.DataFrame(
        {
            "sentence_id": list(range(n)),
            "token": [f"tok{i}" for i in range(n)],
            "annotation": annotations,
            "prediction": predictions,
            "start_indices": list(range(n)),
            "original_annotation": annotations,
            "original_prediction": predictions,
        }
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dict_with_three_keys(self):
        df = _make_mapped_df(["NAME", "O"], ["NAME", "O"])
        results = calculate_hierarchical_scores(df)
        assert set(results.keys()) == {"L0", "L1", "L2"}

    def test_each_value_is_evaluation_result(self):
        df = _make_mapped_df(["NAME", "O"], ["NAME", "O"])
        results = calculate_hierarchical_scores(df)
        for level in ("L0", "L1", "L2"):
            assert isinstance(results[level], EvaluationResult)

    def test_accepts_custom_hierarchy(self):
        h = EntityHierarchy()
        df = _make_mapped_df(["NAME", "O"], ["NAME", "O"])
        results = calculate_hierarchical_scores(df, hierarchy=h)
        assert set(results.keys()) == {"L0", "L1", "L2"}

    def test_all_O_df(self):
        """No PII at all — function must not raise."""
        df = _make_mapped_df(["O", "O"], ["O", "O"])
        results = calculate_hierarchical_scores(df)
        assert set(results.keys()) == {"L0", "L1", "L2"}


# ---------------------------------------------------------------------------
# L0 — PII vs. O
# ---------------------------------------------------------------------------


class TestL0Projection:
    def test_l0_perfect_pii_detection(self):
        """Model correctly identifies all PII tokens."""
        # NAME and EMAIL_ADDRESS are both PII; model predicts them
        df = _make_mapped_df(
            ["NAME", "EMAIL_ADDRESS", "O"],
            ["NAME", "EMAIL_ADDRESS", "O"],
        )
        results = calculate_hierarchical_scores(df)
        # At L0 both map to PII — should be perfect
        assert results["L0"].pii_f == pytest.approx(1.0, abs=1e-6)

    def test_l0_collapses_different_entity_types_to_pii(self):
        """NAME annotation matched by EMAIL_ADDRESS prediction = TP at L0."""
        # Different entity types but both PII — at L0 both = "PII" = TP
        df = _make_mapped_df(
            ["NAME", "O"],
            ["EMAIL_ADDRESS", "O"],
        )
        results = calculate_hierarchical_scores(df)
        # At L0: annotation=PII, prediction=PII → should be a match
        assert results["L0"].pii_recall > 0


# ---------------------------------------------------------------------------
# L1 — branch-level (depth-2)
# ---------------------------------------------------------------------------


class TestL1Projection:
    def test_l1_depth3_maps_to_depth2_ancestor(self):
        """NAME (depth-3, branch PERSON) should map to PERSON at L1."""
        df = _make_mapped_df(["NAME", "O"], ["NAME", "O"])
        results = calculate_hierarchical_scores(df)
        # L1 has PERSON TP — precision should be 1.0
        assert results["L1"].pii_precision == pytest.approx(1.0, abs=1e-6)

    def test_l1_different_branches_are_different(self):
        """NAME (PERSON branch) vs EMAIL_ADDRESS (CONTACT branch) are different at L1."""
        # annotation=NAME (→PERSON@L1), prediction=EMAIL_ADDRESS (→CONTACT@L1): mismatch at L1
        df = _make_mapped_df(
            ["NAME", "O"],
            ["EMAIL_ADDRESS", "O"],
        )
        results = calculate_hierarchical_scores(df)
        # At L1: annotation maps to PERSON, prediction maps to CONTACT — entity-type mismatch.
        # PERSON was annotated but not predicted as PERSON → recall = 0.
        person_metrics = results["L1"].per_type.get("PERSON")
        assert person_metrics is not None
        assert person_metrics.recall == pytest.approx(0.0, abs=1e-6)

    def test_l1_depth2_entity_unchanged(self):
        """PERSON (depth-2) should map to PERSON at L1 unchanged."""
        df = _make_mapped_df(["PERSON", "O"], ["PERSON", "O"])
        results = calculate_hierarchical_scores(df)
        assert results["L1"].pii_f == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# L2 — canonical surface (depth-3, used as-is)
# ---------------------------------------------------------------------------


class TestL2Projection:
    def test_l2_uses_labels_as_is(self):
        """L2 should use mapped_df annotation/prediction unchanged."""
        df = _make_mapped_df(["NAME", "O"], ["NAME", "O"])
        results = calculate_hierarchical_scores(df)
        assert results["L2"].pii_f == pytest.approx(1.0, abs=1e-6)

    def test_l2_mismatch_penalised(self):
        """PERSON predicting on NAME annotation creates FP+FN at L2."""
        # At L2: annotation=NAME, prediction=PERSON → entity-type mismatch.
        # NAME was annotated but PERSON was predicted → NAME has recall = 0 at L2.
        df = _make_mapped_df(["NAME", "O"], ["PERSON", "O"])
        results = calculate_hierarchical_scores(df)
        name_metrics = results["L2"].per_type.get("NAME")
        assert name_metrics is not None
        assert name_metrics.recall == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Granularity bonus property
# ---------------------------------------------------------------------------


class TestGranularityBonus:
    def test_specific_prediction_scores_tp_at_all_levels(self):
        """Model predicting NAME (depth-3) on NAME annotation gets TP at L0, L1, L2."""
        df = _make_mapped_df(["NAME", "O"], ["NAME", "O"])
        results = calculate_hierarchical_scores(df)
        for level in ("L0", "L1", "L2"):
            assert results[level].pii_f == pytest.approx(1.0, abs=1e-6), (
                f"Expected perfect score at {level}"
            )

    def test_coarse_prediction_scores_tp_at_l0_l1_but_not_l2(self):
        """Model predicting PERSON (depth-2) on NAME annotation gets TP at L0/L1 but not L2."""
        df = _make_mapped_df(["NAME", "O"], ["PERSON", "O"])
        results = calculate_hierarchical_scores(df)
        # At L0: NAME→PII, PERSON→PII → PII TP (span detected)
        assert results["L0"].pii_recall > 0
        # At L1: NAME→PERSON, PERSON→PERSON → PERSON TP (entity recall = 1)
        person_l1 = results["L1"].per_type.get("PERSON")
        assert person_l1 is not None
        assert person_l1.recall == pytest.approx(1.0, abs=1e-6)
        # At L2: annotation=NAME, prediction=PERSON → NAME has recall=0 (never predicted as NAME)
        name_l2 = results["L2"].per_type.get("NAME")
        assert name_l2 is not None
        assert name_l2.recall == pytest.approx(0.0, abs=1e-6)

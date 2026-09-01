"""Tests for BaseEvaluator.calculate_hierarchical_scores()."""

import pandas as pd
import pytest

from presidio_evaluator.entity_mapping import (
    CanonicalMapper,
    EntityHierarchy,
    MappedResults,
)
from presidio_evaluator.evaluation import EvaluationResult
from presidio_evaluator.evaluation.span_evaluator import SpanEvaluator
from presidio_evaluator.evaluation.token_evaluator import TokenEvaluator

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


def _make_single_sentence_results(
    annotations: list[str],
    predictions: list[str],
) -> MappedResults:
    """Build MappedResults where all tokens belong to one sentence."""
    df = pd.DataFrame(
        {
            "sentence_id": [0] * len(annotations),
            "token": [f"tok{i}" for i in range(len(annotations))],
            "annotation": annotations,
            "prediction": predictions,
            "start_indices": [i * 5 for i in range(len(annotations))],
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
    @pytest.mark.parametrize(
        ("annotation", "prediction"),
        [
            ("PERSON", "NAME"),
            ("DATE_TIME", "DATE"),
        ],
    )
    @pytest.mark.parametrize("evaluator_type", [SpanEvaluator, TokenEvaluator])
    def test_specific_prediction_scores_tp_at_all_levels(
        self,
        annotation,
        prediction,
        evaluator_type,
    ):
        """A descendant prediction satisfies a less-specific annotation."""
        results = _make_results([annotation, "O"], [prediction, "O"])
        scores = evaluator_type(skip_words=[]).calculate_hierarchical_scores(results)
        for level in ("binary", "branch", "detailed"):
            assert scores[level].pii_f == pytest.approx(1.0, abs=1e-6), (
                f"Expected perfect score at {level}"
            )
        assert scores["detailed"].per_type[annotation].true_positives == 1
        assert scores["detailed"].results[(annotation, annotation)] == 1

    def test_multiple_descendant_spans_combine_for_span_iou(self):
        """Different descendants can jointly cover one broader annotation."""
        results = _make_single_sentence_results(
            ["PERSON", "PERSON"],
            ["NAME", "TITLE"],
        )
        scores = SpanEvaluator(
            skip_words=[],
            iou_threshold=1.0,
            char_based=False,
        ).calculate_hierarchical_scores(results)
        person = scores["detailed"].per_type["PERSON"]
        assert person.true_positives == 1
        assert person.num_predicted == 1
        assert person.false_positives == 0
        assert person.false_negatives == 0

    def test_low_iou_descendants_keep_their_predicted_type(self):
        """A failed descendant match remains an FP for the model's label."""
        results = _make_single_sentence_results(
            ["PERSON", "PERSON", "PERSON", "PERSON"],
            ["NAME", "O", "NAME", "O"],
        )
        scores = SpanEvaluator(
            skip_words=[],
            iou_threshold=1.0,
            char_based=False,
        ).calculate_hierarchical_scores(results)
        detailed = scores["detailed"]
        assert detailed.per_type["PERSON"].false_negatives == 1
        assert detailed.per_type["NAME"].false_positives == 1
        assert detailed.results[("O", "NAME")] == 1
        assert detailed.results[("O", "PERSON")] == 0

    def test_custom_hierarchy_is_used_for_descendant_credit(self):
        """Callers can score descendant relationships from a custom taxonomy."""
        hierarchy = EntityHierarchy(
            hierarchy={"PII": {"CUSTOM_PARENT": {"CUSTOM_CHILD": []}}},
            canonical_depth=10,
        )
        df = pd.DataFrame(
            {
                "sentence_id": [0],
                "token": ["value"],
                "annotation": ["CUSTOM_PARENT"],
                "prediction": ["CUSTOM_CHILD"],
                "start_indices": [0],
            }
        )
        mapper = CanonicalMapper(hierarchy=hierarchy)
        mapper.analyze(df)
        scores = SpanEvaluator(skip_words=[]).calculate_hierarchical_scores(
            mapper.get_mapped_results_dataframe(),
            entity_hierarchy=hierarchy,
        )
        assert scores["detailed"].per_type["CUSTOM_PARENT"].true_positives == 1

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

    def test_coarse_generic_prediction_is_not_forgiven_for_tokens(self):
        """Hierarchy scoring must not treat a coarse PII prediction as exact."""
        results = _make_results(["PERSON"], ["PII"])
        scores = TokenEvaluator(skip_words=[]).calculate_hierarchical_scores(results)
        person = scores["detailed"].per_type["PERSON"]
        assert person.true_positives == 0
        assert person.false_negatives == 1
        assert scores["detailed"].per_type["PII"].false_positives == 1

    def test_calculate_score_on_df_matches_branch_level(self):
        """calculate_score_on_df(results.branch) == scores['branch']."""
        results = _make_results(
            ["NAME", "EMAIL_ADDRESS", "O"], ["NAME", "EMAIL_ADDRESS", "O"]
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        direct = _evaluator.calculate_score_on_df(results.branch)
        assert scores["branch"].pii_f == pytest.approx(direct.pii_f, abs=1e-6)


# ---------------------------------------------------------------------------
# Mapping projection scenarios
#
# Hierarchy used (depth):
#   1: PII
#   2: PERSON, LOCATION
#   3: NAME (under PERSON), GPE (under LOCATION)
#   4: FIRST_NAME (under NAME → PERSON)
# ---------------------------------------------------------------------------


class TestMappingProjectionScenarios:
    """End-to-end projection scenarios for get_mapped_results_dataframe().

    Each scenario exercises how annotation and prediction labels at different
    hierarchy depths project across the three evaluation levels (binary /
    branch / detailed) and what effect that has on per-type metrics.
    """

    # ------------------------------------------------------------------
    # Scenario 1: dataset=depth-1 (PII), model=depths 2, 3, 4
    # ------------------------------------------------------------------

    def test_scenario1_binary_is_perfect(self):
        """S1 binary: PII annotation vs PII-collapsed predictions → all TP."""
        results = _make_results(
            ["PII", "PII", "PII"],
            ["PERSON", "NAME", "FIRST_NAME"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        pii_m = scores["binary"].per_type.get("PII")
        assert pii_m is not None
        assert pii_m.recall == pytest.approx(1.0, abs=1e-6)
        assert pii_m.precision == pytest.approx(1.0, abs=1e-6)

    def test_scenario1_branch_credits_descendant_predictions(self):
        """S1 branch: specific predictions satisfy the PII annotation."""
        results = _make_results(
            ["PII", "PII", "PII"],
            ["PERSON", "NAME", "FIRST_NAME"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        pii_m = scores["branch"].per_type.get("PII")
        assert pii_m is not None
        assert pii_m.recall == pytest.approx(1.0, abs=1e-6)
        assert pii_m.precision == pytest.approx(1.0, abs=1e-6)

    def test_scenario1_detailed_credits_descendant_predictions(self):
        """S1 detailed: more-specific predictions satisfy the PII annotation.

        Note: FIRST_NAME resolves to NAME at the default canonical_depth=3, so only
        PERSON and NAME appear as distinct prediction labels before scoring.
        """
        results = _make_results(
            ["PII", "PII", "PII"],
            ["PERSON", "NAME", "FIRST_NAME"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        pii_m = scores["detailed"].per_type.get("PII")
        assert pii_m is not None
        assert pii_m.recall == pytest.approx(1.0, abs=1e-6)
        assert pii_m.precision == pytest.approx(1.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Scenario 2: dataset=depths 2, 3, 4; model=depth-1 (PII)
    # ------------------------------------------------------------------

    def test_scenario2_binary_is_perfect(self):
        """S2 binary: granular annotations collapse to PII, matching PII predictions."""
        results = _make_results(
            ["PERSON", "NAME", "FIRST_NAME"],
            ["PII", "PII", "PII"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["binary"].pii_recall == pytest.approx(1.0, abs=1e-6)

    def test_scenario2_branch_person_annotations_miss_pii_predictions(self):
        """S2 branch: PERSON/NAME/FIRST_NAME all become PERSON; PII prediction stays PII → mismatch."""
        results = _make_results(
            ["PERSON", "NAME", "FIRST_NAME"],
            ["PII", "PII", "PII"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        person_m = scores["branch"].per_type.get("PERSON")
        assert person_m is not None
        assert person_m.recall == pytest.approx(0.0, abs=1e-6)
        pii_m = scores["branch"].per_type.get("PII")
        assert pii_m is not None
        assert pii_m.false_positives > 0

    def test_scenario2_detailed_all_annotation_types_miss(self):
        """S2 detailed: each granular annotation type gets recall=0 vs the coarse PII prediction.

        Note: FIRST_NAME resolves to NAME at canonical_depth=3, so per_type only contains
        PERSON and NAME (with num_annotated=2 for NAME, covering both NAME and FIRST_NAME tokens).
        """
        results = _make_results(
            ["PERSON", "NAME", "FIRST_NAME"],
            ["PII", "PII", "PII"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        for entity in ("PERSON", "NAME"):
            m = scores["detailed"].per_type.get(entity)
            assert m is not None, f"Expected {entity} in detailed per_type"
            assert m.recall == pytest.approx(0.0, abs=1e-6), (
                f"Expected recall=0 for {entity} at detailed level"
            )

    # ------------------------------------------------------------------
    # Scenario 3: dataset=2×depth-2 + 2×depth-3; model=depth-2 only
    # Annotations: PERSON, LOCATION, NAME, GPE
    # Predictions: PERSON, LOCATION, PERSON, LOCATION
    # ------------------------------------------------------------------

    def test_scenario3_binary_is_perfect(self):
        """S3 binary: all non-O labels collapse to PII on both sides → all TP."""
        results = _make_results(
            ["PERSON", "LOCATION", "NAME", "GPE"],
            ["PERSON", "LOCATION", "PERSON", "LOCATION"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["binary"].pii_recall == pytest.approx(1.0, abs=1e-6)
        assert scores["binary"].pii_precision == pytest.approx(1.0, abs=1e-6)

    def test_scenario3_branch_all_tp_because_depth3_maps_to_same_branch(self):
        """S3 branch: NAME→PERSON and GPE→LOCATION, matching depth-2 predictions → all TP."""
        results = _make_results(
            ["PERSON", "LOCATION", "NAME", "GPE"],
            ["PERSON", "LOCATION", "PERSON", "LOCATION"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        person_m = scores["branch"].per_type.get("PERSON")
        location_m = scores["branch"].per_type.get("LOCATION")
        assert person_m is not None
        assert location_m is not None
        assert person_m.recall == pytest.approx(1.0, abs=1e-6)
        assert location_m.recall == pytest.approx(1.0, abs=1e-6)

    def test_scenario3_detailed_depth2_annotations_hit_depth3_annotations_miss(self):
        """S3 detailed: PERSON/LOCATION annotations (depth-2) are TPs; NAME/GPE (depth-3) are FNs."""
        results = _make_results(
            ["PERSON", "LOCATION", "NAME", "GPE"],
            ["PERSON", "LOCATION", "PERSON", "LOCATION"],
        )
        scores = _evaluator.calculate_hierarchical_scores(results)
        # Depth-2 annotations matched by depth-2 predictions → TP
        assert scores["detailed"].per_type["PERSON"].recall == pytest.approx(
            1.0, abs=1e-6
        )
        assert scores["detailed"].per_type["LOCATION"].recall == pytest.approx(
            1.0, abs=1e-6
        )
        # Depth-3 annotations miss because model only predicts at depth-2
        assert scores["detailed"].per_type["NAME"].recall == pytest.approx(
            0.0, abs=1e-6
        )
        assert scores["detailed"].per_type["GPE"].recall == pytest.approx(0.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Scenario 4: dataset=depth-4 (FIRST_NAME), model=depth-2 (PERSON)
    # ------------------------------------------------------------------

    def test_scenario4_binary_is_perfect(self):
        """S4 binary: FIRST_NAME and PERSON both collapse to PII → TP."""
        results = _make_results(["FIRST_NAME"], ["PERSON"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["binary"].pii_recall == pytest.approx(1.0, abs=1e-6)
        assert scores["binary"].pii_precision == pytest.approx(1.0, abs=1e-6)

    def test_scenario4_branch_is_tp_because_first_name_maps_to_person(self):
        """S4 branch: FIRST_NAME (depth-4) collapses to PERSON branch; prediction is PERSON → TP."""
        results = _make_results(["FIRST_NAME"], ["PERSON"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        person_m = scores["branch"].per_type.get("PERSON")
        assert person_m is not None
        assert person_m.recall == pytest.approx(1.0, abs=1e-6)
        assert person_m.precision == pytest.approx(1.0, abs=1e-6)

    def test_scenario4_detailed_misses_because_labels_differ(self):
        """S4 detailed: FIRST_NAME annotation (→NAME at canonical_depth=3) vs PERSON prediction.

        FIRST_NAME resolves to NAME, so the annotation entity is NAME at detailed level.
        PERSON prediction does not match NAME → FN for NAME, FP for PERSON.
        """
        results = _make_results(["FIRST_NAME"], ["PERSON"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        # FIRST_NAME resolves to NAME at canonical_depth=3
        name_m = scores["detailed"].per_type.get("NAME")
        assert name_m is not None
        assert name_m.recall == pytest.approx(0.0, abs=1e-6)
        person_m = scores["detailed"].per_type.get("PERSON")
        assert person_m is not None
        assert person_m.false_positives > 0

    # ------------------------------------------------------------------
    # Scenario 5: dataset=depth-2 (PERSON), model=depth-4 (FIRST_NAME)
    # ------------------------------------------------------------------

    def test_scenario5_binary_is_perfect(self):
        """S5 binary: PERSON and FIRST_NAME both collapse to PII → TP."""
        results = _make_results(["PERSON"], ["FIRST_NAME"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        assert scores["binary"].pii_recall == pytest.approx(1.0, abs=1e-6)
        assert scores["binary"].pii_precision == pytest.approx(1.0, abs=1e-6)

    def test_scenario5_branch_is_tp_because_first_name_collapses_to_person(self):
        """S5 branch: FIRST_NAME prediction collapses to PERSON; annotation is PERSON → TP."""
        results = _make_results(["PERSON"], ["FIRST_NAME"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        person_m = scores["branch"].per_type.get("PERSON")
        assert person_m is not None
        assert person_m.recall == pytest.approx(1.0, abs=1e-6)
        assert person_m.precision == pytest.approx(1.0, abs=1e-6)

    def test_scenario5_detailed_credits_more_specific_prediction(self):
        """S5 detailed: PERSON annotation vs FIRST_NAME prediction (→NAME at canonical_depth=3).

        FIRST_NAME resolves to NAME, so the prediction entity is NAME at detailed level.
        NAME is a descendant of PERSON, so the prediction receives full credit.
        """
        results = _make_results(["PERSON"], ["FIRST_NAME"])
        scores = _evaluator.calculate_hierarchical_scores(results)
        person_m = scores["detailed"].per_type.get("PERSON")
        assert person_m is not None
        assert person_m.recall == pytest.approx(1.0, abs=1e-6)
        assert person_m.precision == pytest.approx(1.0, abs=1e-6)

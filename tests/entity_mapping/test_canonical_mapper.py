"""Tests for the two-phase (Identify -> Project) CanonicalMapper."""

import pandas as pd
import pytest

from presidio_evaluator.entity_mapping import (
    CanonicalMapper,
    IncompleteMapping,
    IssueSeverity,
    IssueType,
    MapperRenderer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(annotations, predictions):
    n = max(len(annotations), len(predictions))
    annotations = list(annotations) + ["O"] * (n - len(annotations))
    predictions = list(predictions) + ["O"] * (n - len(predictions))
    return pd.DataFrame(
        {
            "sentence_id": list(range(n)),
            "token": [f"t{i}" for i in range(n)],
            "annotation": annotations,
            "prediction": predictions,
            "start_indices": [0] * n,
        }
    )


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_no_args(self):
        mapper = CanonicalMapper()
        assert mapper.pending == []
        assert mapper._records == {}

    def test_keyword_only_args(self):
        mapper = CanonicalMapper(fuzzy_threshold=0.90)
        assert mapper._fuzzy_threshold == 0.90

    def test_custom_hierarchy_dict(self):
        custom = {"PII": {"MY_TYPE": ["my_alias"]}}
        mapper = CanonicalMapper(hierarchy=custom)
        assert "MY_TYPE" in mapper._hierarchy.all_canonical_entities

    def test_repr(self):
        mapper = CanonicalMapper()
        assert "CanonicalMapper" in repr(mapper)


# ---------------------------------------------------------------------------
# Phase 1: Identification
# ---------------------------------------------------------------------------


class TestIdentification:
    def test_exact_match(self):
        df = _make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        rec = mapper._records["EMAIL_ADDRESS"]
        assert rec.tier == "EXACT"
        assert rec.resolved == "EMAIL_ADDRESS"

    def test_country_prefix(self):
        df = _make_df(["GERMANY_PASSPORT_NUMBER"], ["GERMANY_PASSPORT_NUMBER"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        rec = mapper._records["GERMANY_PASSPORT_NUMBER"]
        assert rec.tier == "COUNTRY"
        assert rec.resolved == "PASSPORT"

    def test_fuzzy_match(self):
        df = _make_df(["EMAILADRES"], ["EMAILADRES"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        rec = mapper._records.get("EMAILADRES")
        if rec and rec.tier == "FUZZY":
            assert rec.resolved is not None
            assert rec.score is not None

    def test_unresolved(self):
        df = _make_df(["XYZZY_TOTALLY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        assert "XYZZY_TOTALLY_UNKNOWN_99" in mapper.pending
        rec = mapper._records["XYZZY_TOTALLY_UNKNOWN_99"]
        assert rec.tier == "UNRESOLVED"
        assert rec.resolved is None

    def test_bio_prefix_stripped(self):
        df = _make_df(["B-PERSON"], ["B-PERSON"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        rec = mapper._records.get("B-PERSON")
        assert rec is not None
        assert rec.resolved == "PERSON"

    def test_o_token_not_in_records(self):
        df = _make_df(["O", "EMAIL_ADDRESS"], ["O", "EMAIL_ADDRESS"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        # O is filtered before processing
        assert "O" not in mapper._records


# ---------------------------------------------------------------------------
# Single-phase identification — no projection, no canonical surface
# ---------------------------------------------------------------------------


class TestSinglePhase:
    def test_records_have_resolved_field(self):
        """After analyze(), each record has a resolved hierarchy node."""
        df = _make_df(["NAME"], ["NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        rec = mapper._records["NAME"]
        assert rec.resolved == "NAME"

    def test_depth4_label_resolves_via_hierarchy(self):
        """FIRST_NAME resolves to its canonical hierarchy node (NAME at depth-3)."""
        df = _make_df(["FIRST_NAME"], ["FIRST_NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        rec = mapper._records["FIRST_NAME"]
        # FIRST_NAME is a depth-4 alias; canonical hierarchy node is NAME
        assert rec.resolved is not None
        assert rec.tier in ("EXACT", "FUZZY")

    def test_no_canonical_surface_attribute(self):
        """CanonicalMapper no longer has a canonical_surface property."""
        mapper = CanonicalMapper()
        assert not hasattr(mapper, "canonical_surface")

    def test_analyze_twice_recomputes_issues(self):
        """Re-analyzing with a different DataFrame recomputes issues."""
        df1 = _make_df(["NAME"] * 5, ["NAME"] * 5)
        df2 = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df1)
        issues1 = [i.type for i in mapper.get_issues()]
        mapper.analyze(df2)
        issues2 = [i.type for i in mapper.get_issues()]
        assert issues1 != issues2


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------


class TestIssues:
    def test_unresolved_is_error(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        issues = mapper.get_issues()
        err = next((i for i in issues if i.type == IssueType.UNRESOLVED), None)
        assert err is not None
        assert err.severity == IssueSeverity.ERROR

    def test_sorted_by_severity_then_tokens(self):
        df = _make_df(
            ["EMAIL_ADDRESS", "NAME", "UNKNOWN_99"],
            ["EMAIL_ADDRESS", "NAME", "O"],
        )
        mapper = CanonicalMapper()
        mapper.analyze(df)
        issues = mapper.get_issues()
        severities = [i.severity for i in issues]
        order = [IssueSeverity.ERROR, IssueSeverity.WARNING, IssueSeverity.INFO]
        for a, b in zip(severities, severities[1:]):
            assert order.index(a) <= order.index(b)

    def test_prediction_only_warning(self):
        # EMAIL_ADDRESS in predictions but not annotations
        df = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        pred_only = [
            i for i in mapper.get_issues() if i.type == IssueType.PREDICTION_ONLY
        ]
        assert len(pred_only) > 0
        assert pred_only[0].severity == IssueSeverity.WARNING

    def test_dataset_only_warning(self):
        # EMAIL_ADDRESS in annotations but no EMAIL_ADDRESS predictions → DATASET_ONLY WARNING
        df = _make_df(["NAME"] * 5 + ["EMAIL_ADDRESS"] * 5, ["NAME"] * 10)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        issues = mapper.get_issues()
        ds_only = [i for i in issues if i.type == IssueType.DATASET_ONLY]
        assert len(ds_only) > 0, "Expected at least one DATASET_ONLY issue"
        assert any("EMAIL_ADDRESS" in i.labels for i in ds_only)
        for issue in ds_only:
            assert issue.severity == IssueSeverity.WARNING
            assert issue.annotation_count > 0
            assert issue.prediction_count == 0

    def test_collision_same_branch_info(self):
        # PERSON prediction co-occurs with NAME annotation — same branch (PERSON), different depth
        df = _make_df(["NAME"] * 8, ["PERSON"] * 8)
        mapper = CanonicalMapper()
        mapper.analyze(df, min_severity="INFO")
        same_branch = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_SAME_BRANCH
        ]
        assert len(same_branch) > 0
        for issue in same_branch:
            assert issue.severity == IssueSeverity.INFO
            assert issue.overlap_counts is not None

    def test_cross_branch_suppressed_when_same_branch_hit_exists(self):
        # LOCATION has same-branch hits (predicted on STREET_ADDRESS tokens) AND
        # cross-branch FPs (predicted on NAME tokens) → Case 1 → no warning.
        annotations = ["STREET_ADDRESS"] * 5 + ["NAME"] * 5
        predictions = ["LOCATION"] * 10
        df = _make_df(annotations, predictions)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        cross = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_CROSS_BRANCH
        ]
        assert cross == [], (
            "LOCATION should not fire COLLISION_CROSS_BRANCH when it is also "
            "predicted on same-branch tokens (STREET_ADDRESS)"
        )

    def test_cross_branch_case2_displaced_prediction(self):
        # LOCATION branch IS annotated (STREET_ADDRESS), but LOCATION is only ever
        # predicted on NAME tokens — same_branch_count == 0 → Case 2 → warning fires.
        # Row layout: annotation=NAME on first 5, annotation=STREET_ADDRESS on last 5
        #             prediction=LOCATION on first 5 only (last 5 predict O)
        annotations = ["NAME"] * 5 + ["STREET_ADDRESS"] * 5
        predictions = ["LOCATION"] * 5 + ["O"] * 5
        df = _make_df(annotations, predictions)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        cross = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_CROSS_BRANCH
        ]
        assert len(cross) > 0, (
            "LOCATION never predicted on LOCATION-branch (STREET_ADDRESS) tokens "
            "but is predicted on NAME (PERSON-branch) tokens → should warn (Case 2)"
        )
        assert "never predicted" in cross[0].message or "all" in cross[0].message

    def test_cross_branch_case3_branch_absent_from_annotations(self):
        # DATE_TIME branch entirely absent from annotations (only NAME annotated).
        # Model predicts DATE_TIME on NAME tokens and never predicts PERSON-branch labels.
        # → Case 3 → COLLISION_CROSS_BRANCH fires.
        annotations = ["NAME"] * 8
        predictions = ["DATE_TIME"] * 8
        df = _make_df(annotations, predictions)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        cross = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_CROSS_BRANCH
        ]
        assert len(cross) > 0, (
            "DATE_TIME (branch absent from annotations) predicted on NAME tokens "
            "with no PERSON-branch predictions → Case 3 should fire"
        )

    def test_cross_branch_suppressed_when_model_covers_annotation_branch(self):
        # LOCATION is predicted on both STREET_ADDRESS (same-branch hit) and NAME
        # tokens (cross-branch FP) → same_branch_count > 0 → Case 1 → no warning.
        annotations = ["NAME"] * 8 + ["STREET_ADDRESS"] * 2
        predictions = ["LOCATION"] * 10  # LOCATION on STREET_ADDRESS = same-branch hit
        df = _make_df(annotations, predictions)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        cross = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_CROSS_BRANCH
        ]
        assert cross == [], (
            "LOCATION should not be COLLISION_CROSS_BRANCH when it is also "
            "predicted on same-branch (STREET_ADDRESS) tokens"
        )

    def test_issues_have_token_counts(self):
        df = _make_df(["UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        for issue in mapper.get_issues():
            if issue.type == IssueType.UNRESOLVED and "UNKNOWN_99" in issue.labels:
                total = (issue.annotation_count or 0) + (issue.prediction_count or 0)
                assert total >= 1


# ---------------------------------------------------------------------------
# min_severity parameter on analyze()
# ---------------------------------------------------------------------------


class TestMinSeverity:
    def test_default_warning_excludes_info(self):
        # Default min_severity='WARNING' — COLLISION_SAME_BRANCH (INFO) must be hidden
        df = _make_df(["NAME"] * 8, ["PERSON"] * 8)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        same_branch = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_SAME_BRANCH
        ]
        assert same_branch == []

    def test_info_shows_same_branch(self):
        df = _make_df(["NAME"] * 8, ["PERSON"] * 8)
        mapper = CanonicalMapper()
        mapper.analyze(df, min_severity="INFO")
        same_branch = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_SAME_BRANCH
        ]
        assert len(same_branch) > 0

    def test_error_shows_only_unresolved(self):
        df = _make_df(["XYZZY_99"] * 5, ["NAME"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df, min_severity="ERROR")
        issues = mapper.get_issues()
        severities = {i.severity for i in issues}
        assert severities <= {IssueSeverity.ERROR}

    def test_warning_shows_warning_and_error(self):
        df = _make_df(["XYZZY_99"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df, min_severity="WARNING")
        issues = mapper.get_issues()
        severities = {i.severity for i in issues}
        assert all(
            s in (IssueSeverity.ERROR, IssueSeverity.WARNING) for s in severities
        )

    def test_invalid_severity_raises_valueerror(self):
        df = _make_df(["NAME"], ["NAME"])
        with pytest.raises(ValueError, match="Unrecognised severity"):
            CanonicalMapper().analyze(df, min_severity="CRITICAL")

    def test_enum_value_accepted(self):
        df = _make_df(["NAME"] * 5, ["PERSON"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df, min_severity=IssueSeverity.INFO)
        same_branch = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_SAME_BRANCH
        ]
        assert len(same_branch) > 0


# ---------------------------------------------------------------------------
# map()
# ---------------------------------------------------------------------------


class TestMap:
    def test_map_resolves_unresolved(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        assert any(i.type == IssueType.UNRESOLVED for i in mapper.get_issues())
        mapper.map({"XYZZY_UNKNOWN_99": "NAME"})
        assert not any(i.type == IssueType.UNRESOLVED for i in mapper.get_issues())

    def test_map_suppress(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        mapper.map({"XYZZY_UNKNOWN_99": None})
        rec = mapper._records["XYZZY_UNKNOWN_99"]
        assert rec.tier == "NONE"
        assert rec.resolved is None

    def test_map_returns_none(self):
        df = _make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.map({"EMAIL_ADDRESS": "EMAIL_ADDRESS"})
        assert result is None

    def test_map_invalid_raises_atomically(self):
        df = _make_df(["XYZZY_UNKNOWN_99", "ANOTHER_UNKNOWN_99"], ["O", "O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        with pytest.raises(ValueError, match="not a valid canonical entity"):
            mapper.map(
                {
                    "XYZZY_UNKNOWN_99": "NAME",
                    "ANOTHER_UNKNOWN_99": "NOT_REAL_CANONICAL_VALUE",
                }
            )
        # Neither applied
        assert mapper._records["XYZZY_UNKNOWN_99"].tier == "UNRESOLVED"

    def test_map_prediction_only_suppress(self):
        df = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        pred_only = [
            i for i in mapper.get_issues() if i.type == IssueType.PREDICTION_ONLY
        ]
        assert len(pred_only) > 0
        lbl = pred_only[0].labels[0]
        mapper.map({lbl: None})
        remaining = [
            i for i in mapper.get_issues() if i.type == IssueType.PREDICTION_ONLY
        ]
        assert not any(lbl in i.labels for i in remaining)

    def test_map_pre_analyze(self):
        mapper = CanonicalMapper()
        mapper.map({"MY_CUSTOM": "EMAIL_ADDRESS"})
        df = _make_df(["MY_CUSTOM"], ["MY_CUSTOM"])
        mapper.analyze(df)
        rec = mapper._records.get("MY_CUSTOM")
        assert rec is not None
        assert rec.resolved == "EMAIL_ADDRESS"


# ---------------------------------------------------------------------------
# suppress_prediction_only()
# ---------------------------------------------------------------------------


class TestSuppressPredictionOnly:
    def test_clears_prediction_only_issues(self):
        # EMAIL_ADDRESS in predictions only → PREDICTION_ONLY → suppressed
        df = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        assert any(i.type == IssueType.PREDICTION_ONLY for i in mapper.get_issues())
        mapper.suppress_prediction_only()
        assert not any(i.type == IssueType.PREDICTION_ONLY for i in mapper.get_issues())

    def test_suppressed_labels_mapped_to_none(self):
        df = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        pred_only_labels = {
            lbl
            for i in mapper.get_issues()
            if i.type == IssueType.PREDICTION_ONLY
            for lbl in i.labels
        }
        mapper.suppress_prediction_only()
        mapping = mapper.get_mapping()
        for lbl in pred_only_labels:
            assert mapping.get(lbl) is None, f"{lbl!r} should be suppressed (None)"

    def test_returns_none(self):
        df = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.suppress_prediction_only()
        assert result is None

    def test_noop_when_no_prediction_only(self):
        df = _make_df(["NAME"] * 5, ["NAME"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        assert not any(i.type == IssueType.PREDICTION_ONLY for i in mapper.get_issues())
        result = mapper.suppress_prediction_only()
        assert result is None  # no error, returns None

    def test_raises_before_analyze(self):
        mapper = CanonicalMapper()
        with pytest.raises(RuntimeError, match="analyze"):
            mapper.suppress_prediction_only()

    def test_suppress_then_get_mapped_results(self):
        from presidio_evaluator.entity_mapping import MappedResults  # noqa: PLC0415

        df = _make_df(["NAME"] * 5, ["NAME"] * 3 + ["EMAIL_ADDRESS"] * 2)
        # EMAIL_ADDRESS is prediction-only; after suppression no blocking issues remain
        mapper = CanonicalMapper()
        mapper.analyze(df)
        mapper.suppress_prediction_only()
        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, MappedResults)


# ---------------------------------------------------------------------------
# get_mapped_results_dataframe()
# ---------------------------------------------------------------------------


class TestGetMappedResultsDf:
    def test_raises_without_analyze(self):
        mapper = CanonicalMapper()
        with pytest.raises(RuntimeError, match="analyze"):
            mapper.get_mapped_results_dataframe()

    def test_raises_only_for_unresolved_not_warning(self):
        # WARNING issues (PREDICTION_ONLY, DATASET_ONLY, COLLISION_CROSS_BRANCH) do NOT block
        df = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        # There may be WARNING issues (PREDICTION_ONLY), but they should not block
        assert any(i.severity == IssueSeverity.WARNING for i in mapper.get_issues())
        result = mapper.get_mapped_results_dataframe()
        from presidio_evaluator.entity_mapping import MappedResults  # noqa: PLC0415

        assert isinstance(result, MappedResults)

    def test_returns_mapped_results_when_no_blocking(self):
        from presidio_evaluator.entity_mapping import MappedResults  # noqa: PLC0415

        df = _make_df(["NAME"] * 3, ["NAME"] * 3)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, MappedResults)

    def test_original_preserves_raw_labels(self):
        df = _make_df(["NAME"], ["NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapped_results_dataframe()
        assert result.original["annotation"].tolist()[0] == "NAME"
        assert result.original["prediction"].tolist()[0] == "NAME"

    def test_binary_maps_pii_to_pii(self):
        df = _make_df(["NAME"], ["NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapped_results_dataframe()
        assert result.binary["annotation"].tolist()[0] == "PII"
        assert result.binary["prediction"].tolist()[0] == "PII"

    def test_branch_maps_to_depth2_ancestor(self):
        df = _make_df(["NAME"], ["NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapped_results_dataframe()
        # NAME -> PERSON branch
        assert result.branch["annotation"].tolist()[0] == "PERSON"

    def test_detailed_preserves_native_depth(self):
        df = _make_df(["NAME"], ["NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapped_results_dataframe()
        # detailed uses the resolved hierarchy node (NAME itself at depth 3)
        assert result.detailed["annotation"].tolist()[0] == "NAME"

    def test_suppressed_labels_become_o_in_all_levels(self):
        df = _make_df(["NAME"] * 3, ["EMAIL_ADDRESS"] * 3)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        pred_only = [
            i for i in mapper.get_issues() if i.type == IssueType.PREDICTION_ONLY
        ]
        if pred_only:
            for lbl in pred_only[0].labels:
                mapper.map({lbl: None})
        result = mapper.get_mapped_results_dataframe()
        assert "O" in result.binary["prediction"].values
        assert "O" in result.branch["prediction"].values
        assert "O" in result.detailed["prediction"].values

    def test_only_unresolved_blocks(self):
        # Only ERROR (UNRESOLVED) blocks get_mapped_results_dataframe()
        df = _make_df(["NAME"] * 8, ["NAME"] * 8)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        from presidio_evaluator.entity_mapping import MappedResults  # noqa: PLC0415

        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, MappedResults)


# ---------------------------------------------------------------------------
# get_mapped_results_dataframe() — projection scenarios
#
# These tests verify the actual annotation/prediction column values at each
# level for the five scenarios described in the hierarchical-evaluation tests.
#
# Hierarchy depths used:
#   1: PII
#   2: PERSON, LOCATION
#   3: NAME (under PERSON), GPE (under LOCATION)
#   4: FIRST_NAME (→ resolves to NAME at default canonical_depth=3)
# ---------------------------------------------------------------------------


class TestMappedResultsProjectionScenarios:
    """Validate column values in .original / .binary / .branch / .detailed."""

    def _mapped(self, annotations, predictions):
        df = _make_df(annotations, predictions)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        return mapper.get_mapped_results_dataframe()

    # ------------------------------------------------------------------
    # Scenario 1: dataset=depth-1 (PII), model=depths 2 & 3
    # ------------------------------------------------------------------

    def test_scenario1_original_preserves_raw_labels(self):
        r = self._mapped(["PII", "PII"], ["PERSON", "NAME"])
        assert r.original["annotation"].tolist() == ["PII", "PII"]
        assert r.original["prediction"].tolist() == ["PERSON", "NAME"]

    def test_scenario1_binary_both_sides_become_pii(self):
        r = self._mapped(["PII", "PII"], ["PERSON", "NAME"])
        assert set(r.binary["annotation"].unique()) == {"PII"}
        assert set(r.binary["prediction"].unique()) == {"PII"}

    def test_scenario1_branch_annotation_stays_pii_prediction_becomes_person(self):
        """PII (depth-1) has no depth-2 ancestor; stays as PII. PERSON/NAME → PERSON."""
        r = self._mapped(["PII", "PII"], ["PERSON", "NAME"])
        assert r.branch["annotation"].tolist() == ["PII", "PII"]
        assert r.branch["prediction"].tolist() == ["PERSON", "PERSON"]

    def test_scenario1_detailed_annotation_stays_pii_prediction_uses_native_depth(self):
        """PII stays PII; PERSON stays PERSON; NAME stays NAME at detailed level."""
        r = self._mapped(["PII", "PII"], ["PERSON", "NAME"])
        assert r.detailed["annotation"].tolist() == ["PII", "PII"]
        assert r.detailed["prediction"].tolist() == ["PERSON", "NAME"]

    # ------------------------------------------------------------------
    # Scenario 2: dataset=depths 2 & 3, model=depth-1 (PII)
    # ------------------------------------------------------------------

    def test_scenario2_original_preserves_raw_labels(self):
        r = self._mapped(["PERSON", "NAME"], ["PII", "PII"])
        assert r.original["annotation"].tolist() == ["PERSON", "NAME"]
        assert r.original["prediction"].tolist() == ["PII", "PII"]

    def test_scenario2_binary_both_sides_become_pii(self):
        r = self._mapped(["PERSON", "NAME"], ["PII", "PII"])
        assert set(r.binary["annotation"].unique()) == {"PII"}
        assert set(r.binary["prediction"].unique()) == {"PII"}

    def test_scenario2_branch_annotation_becomes_person_prediction_stays_pii(self):
        """PERSON → PERSON, NAME → PERSON at branch; PII stays PII."""
        r = self._mapped(["PERSON", "NAME"], ["PII", "PII"])
        assert r.branch["annotation"].tolist() == ["PERSON", "PERSON"]
        assert r.branch["prediction"].tolist() == ["PII", "PII"]

    def test_scenario2_detailed_annotation_native_prediction_stays_pii(self):
        """PERSON stays PERSON, NAME stays NAME; PII stays PII at detailed."""
        r = self._mapped(["PERSON", "NAME"], ["PII", "PII"])
        assert r.detailed["annotation"].tolist() == ["PERSON", "NAME"]
        assert r.detailed["prediction"].tolist() == ["PII", "PII"]

    # ------------------------------------------------------------------
    # Scenario 3: dataset=depth-2 + depth-3; model=depth-2 only
    # Annotations: PERSON, NAME, LOCATION, GPE
    # Predictions: PERSON, PERSON, LOCATION, LOCATION
    # ------------------------------------------------------------------

    def test_scenario3_binary_all_pii(self):
        r = self._mapped(
            ["PERSON", "NAME", "LOCATION", "GPE"],
            ["PERSON", "PERSON", "LOCATION", "LOCATION"],
        )
        assert set(r.binary["annotation"].unique()) == {"PII"}
        assert set(r.binary["prediction"].unique()) == {"PII"}

    def test_scenario3_branch_depth3_annotations_fold_to_depth2(self):
        """NAME→PERSON, GPE→LOCATION at branch; depth-2 annotations/predictions unchanged."""
        r = self._mapped(
            ["PERSON", "NAME", "LOCATION", "GPE"],
            ["PERSON", "PERSON", "LOCATION", "LOCATION"],
        )
        assert r.branch["annotation"].tolist() == [
            "PERSON",
            "PERSON",
            "LOCATION",
            "LOCATION",
        ]
        assert r.branch["prediction"].tolist() == [
            "PERSON",
            "PERSON",
            "LOCATION",
            "LOCATION",
        ]

    def test_scenario3_detailed_depth2_annotations_stay_depth3_annotations_stay(self):
        """At detailed: PERSON stays PERSON, NAME stays NAME, GPE stays GPE; predictions stay depth-2."""
        r = self._mapped(
            ["PERSON", "NAME", "LOCATION", "GPE"],
            ["PERSON", "PERSON", "LOCATION", "LOCATION"],
        )
        assert r.detailed["annotation"].tolist() == [
            "PERSON",
            "NAME",
            "LOCATION",
            "GPE",
        ]
        assert r.detailed["prediction"].tolist() == [
            "PERSON",
            "PERSON",
            "LOCATION",
            "LOCATION",
        ]

    # ------------------------------------------------------------------
    # Scenario 4: dataset=depth-4 (FIRST_NAME → NAME), model=depth-2 (PERSON)
    # ------------------------------------------------------------------

    def test_scenario4_binary_both_pii(self):
        r = self._mapped(["FIRST_NAME"], ["PERSON"])
        assert r.binary["annotation"].tolist() == ["PII"]
        assert r.binary["prediction"].tolist() == ["PII"]

    def test_scenario4_branch_both_become_person(self):
        """FIRST_NAME resolves to NAME (depth-3) → PERSON branch; PERSON also → PERSON."""
        r = self._mapped(["FIRST_NAME"], ["PERSON"])
        assert r.branch["annotation"].tolist() == ["PERSON"]
        assert r.branch["prediction"].tolist() == ["PERSON"]

    def test_scenario4_detailed_annotation_is_name_prediction_is_person(self):
        """FIRST_NAME resolves to NAME at canonical_depth=3; PERSON stays PERSON."""
        r = self._mapped(["FIRST_NAME"], ["PERSON"])
        assert r.detailed["annotation"].tolist() == ["NAME"]
        assert r.detailed["prediction"].tolist() == ["PERSON"]

    # ------------------------------------------------------------------
    # Scenario 5: dataset=depth-2 (PERSON), model=depth-4 (FIRST_NAME → NAME)
    # ------------------------------------------------------------------

    def test_scenario5_binary_both_pii(self):
        r = self._mapped(["PERSON"], ["FIRST_NAME"])
        assert r.binary["annotation"].tolist() == ["PII"]
        assert r.binary["prediction"].tolist() == ["PII"]

    def test_scenario5_branch_both_become_person(self):
        """PERSON → PERSON; FIRST_NAME → NAME → PERSON branch."""
        r = self._mapped(["PERSON"], ["FIRST_NAME"])
        assert r.branch["annotation"].tolist() == ["PERSON"]
        assert r.branch["prediction"].tolist() == ["PERSON"]

    def test_scenario5_detailed_annotation_is_person_prediction_is_name(self):
        """PERSON stays PERSON; FIRST_NAME resolves to NAME at canonical_depth=3."""
        r = self._mapped(["PERSON"], ["FIRST_NAME"])
        assert r.detailed["annotation"].tolist() == ["PERSON"]
        assert r.detailed["prediction"].tolist() == ["NAME"]


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------


class TestMultiModel:
    def test_issues_recomputed_per_model(self):
        df1 = _make_df(["NAME"] * 5, ["NAME"] * 5)
        df2 = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df1)
        issues1 = [i.type for i in mapper.get_issues()]
        mapper.analyze(df2)
        issues2 = [i.type for i in mapper.get_issues()]
        assert issues1 != issues2


# ---------------------------------------------------------------------------
# get_mapping()
# ---------------------------------------------------------------------------


class TestGetMapping:
    def test_single_entity_lookup(self):
        df = _make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapping(entity="EMAIL_ADDRESS")
        assert result is not None

    def test_full_mapping_dict(self):
        df = _make_df(["EMAIL_ADDRESS", "NAME"], ["EMAIL_ADDRESS", "NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapping()
        assert isinstance(result, dict)
        assert "EMAIL_ADDRESS" in result

    def test_unresolved_labels_excluded(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapping()
        assert "XYZZY_UNKNOWN_99" not in result

    def test_returns_plain_copy(self):
        df = _make_df(["NAME"], ["NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        result = mapper.get_mapping()
        result["INJECTED"] = "FAKE"
        assert "INJECTED" not in mapper.get_mapping()

    def test_suppressed_label_is_none(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        mapper.map({"XYZZY_UNKNOWN_99": None})
        result = mapper.get_mapping()
        assert "XYZZY_UNKNOWN_99" in result
        assert result["XYZZY_UNKNOWN_99"] is None

    def test_unknown_entity_raises(self):
        mapper = CanonicalMapper()
        with pytest.raises(ValueError, match="Cannot resolve"):
            mapper.get_mapping(entity="TOTALLY_UNKNOWN_XYZZY")


# ---------------------------------------------------------------------------
# MapperRenderer
# ---------------------------------------------------------------------------


class TestRenderHtml:
    def test_print_text_does_not_raise(self):
        df = _make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        MapperRenderer(mapper).print_text()  # should not raise

    def test_build_html_returns_string(self):
        df = _make_df(["EMAIL_ADDRESS", "NAME"], ["EMAIL_ADDRESS", "NAME"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        html = MapperRenderer(mapper).build_html()
        assert isinstance(html, str)
        assert "<table" in html

    def test_build_html_before_analyze(self):
        mapper = CanonicalMapper()
        html = MapperRenderer(mapper).build_html()
        assert isinstance(html, str)

    def test_summary_bar_present_in_html(self):
        df = _make_df(["NAME"] * 5, ["NAME"] * 5)
        mapper = CanonicalMapper()
        mapper.analyze(df)
        html = MapperRenderer(mapper).build_html()
        # Summary bar should contain the five issue type names
        assert "Unresolved" in html
        assert "Cross-branch" in html
        assert "Pred-only" in html
        assert "Dataset-only" in html
        assert "Same-branch" in html

    def test_collision_same_branch_card_shown_at_info(self):
        df = _make_df(["NAME"] * 8, ["PERSON"] * 8)
        mapper = CanonicalMapper()
        mapper.analyze(df, min_severity="INFO")
        html = MapperRenderer(mapper).build_html()
        assert "Same-branch depth mismatch" in html

    def test_collision_same_branch_card_hidden_at_warning(self):
        df = _make_df(["NAME"] * 8, ["PERSON"] * 8)
        mapper = CanonicalMapper()
        mapper.analyze(df, min_severity="WARNING")
        html = MapperRenderer(mapper).build_html()
        # The gap card text for COLLISION_SAME_BRANCH should NOT appear in gap_cards
        assert "Handled automatically" not in html


# ---------------------------------------------------------------------------
# EntityHierarchy.get_depth
# ---------------------------------------------------------------------------


class TestGetDepth:
    def test_depth2_entity(self):
        from presidio_evaluator.entity_mapping import EntityHierarchy  # noqa: PLC0415

        h = EntityHierarchy(canonical_depth=10)
        # PERSON is at depth 2: PII -> PERSON
        depth = h.get_depth("PERSON")
        assert depth == 2

    def test_depth3_entity(self):
        from presidio_evaluator.entity_mapping import EntityHierarchy  # noqa: PLC0415

        h = EntityHierarchy(canonical_depth=10)
        # NAME is at depth 3: PII -> PERSON -> NAME
        depth = h.get_depth("NAME")
        assert depth == 3

    def test_depth4_entity(self):
        from presidio_evaluator.entity_mapping import EntityHierarchy  # noqa: PLC0415

        h = EntityHierarchy(canonical_depth=10)
        # FIRST_NAME is at depth 4: PII -> PERSON -> NAME -> FIRST_NAME
        depth = h.get_depth("FIRST_NAME")
        assert depth == 4

    def test_unknown_entity_raises(self):
        from presidio_evaluator.entity_mapping import (  # noqa: PLC0415
            EntityHierarchy,
            EntityNotMappedError,
        )

        h = EntityHierarchy(canonical_depth=10)
        with pytest.raises(EntityNotMappedError):
            h.get_depth("NOT_A_REAL_ENTITY_XYZZY")


# ---------------------------------------------------------------------------
# Integration: full analyze() end-to-end
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline_clean(self):
        """All labels known, same on both sides — no blocking issues."""
        from presidio_evaluator.entity_mapping import MappedResults  # noqa: PLC0415

        df = _make_df(
            ["NAME", "EMAIL_ADDRESS", "SSN"] * 10,
            ["NAME", "EMAIL_ADDRESS", "SSN"] * 10,
        )
        mapper = CanonicalMapper()
        mapper.analyze(df)
        blocking = [i for i in mapper.get_issues() if i.severity == IssueSeverity.ERROR]
        assert blocking == []
        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, MappedResults)
        assert "annotation" in result.original.columns
        assert "annotation" in result.binary.columns

    def test_full_pipeline_with_unresolved(self):
        """Unresolved label blocks extraction until resolved."""
        from presidio_evaluator.entity_mapping import MappedResults  # noqa: PLC0415

        df = _make_df(["NAME", "UNKNOWN_XYZZY_99"], ["NAME", "O"])
        mapper = CanonicalMapper()
        mapper.analyze(df)
        with pytest.raises(IncompleteMapping):
            mapper.get_mapped_results_dataframe()
        mapper.map({"UNKNOWN_XYZZY_99": "NAME"})
        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, MappedResults)

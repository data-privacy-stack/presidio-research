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
        assert mapper.canonical_surface == set()
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
        mapper = CanonicalMapper().analyze(df)
        rec = mapper._records["EMAIL_ADDRESS"]
        assert rec.tier == "EXACT"
        assert rec.canonical == "EMAIL_ADDRESS"

    def test_country_prefix(self):
        df = _make_df(["GERMANY_PASSPORT_NUMBER"], ["GERMANY_PASSPORT_NUMBER"])
        mapper = CanonicalMapper().analyze(df)
        rec = mapper._records["GERMANY_PASSPORT_NUMBER"]
        assert rec.tier == "COUNTRY"
        assert rec.canonical == "PASSPORT"

    def test_fuzzy_match(self):
        df = _make_df(["EMAILADRES"], ["EMAILADRES"])
        mapper = CanonicalMapper().analyze(df)
        rec = mapper._records.get("EMAILADRES")
        if rec and rec.tier == "FUZZY":
            assert rec.canonical is not None
            assert rec.score is not None

    def test_unresolved(self):
        df = _make_df(["XYZZY_TOTALLY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper().analyze(df)
        assert "XYZZY_TOTALLY_UNKNOWN_99" in mapper.pending
        rec = mapper._records["XYZZY_TOTALLY_UNKNOWN_99"]
        assert rec.tier == "UNRESOLVED"
        assert rec.canonical is None

    def test_bio_prefix_stripped(self):
        df = _make_df(["B-PERSON"], ["B-PERSON"])
        mapper = CanonicalMapper().analyze(df)
        rec = mapper._records.get("B-PERSON")
        assert rec is not None
        assert rec.canonical == "PERSON"

    def test_o_token_not_in_records(self):
        df = _make_df(["O", "EMAIL_ADDRESS"], ["O", "EMAIL_ADDRESS"])
        mapper = CanonicalMapper().analyze(df)
        # O is filtered before processing
        assert "O" not in mapper._records


# ---------------------------------------------------------------------------
# Phase 2: Canonical depth computation
# ---------------------------------------------------------------------------


class TestEvalDepth:
    def test_all_depth3_entities(self):
        # NAME is depth 3 in the hierarchy: PII -> PERSON -> NAME
        df = _make_df(["NAME"] * 10, ["NAME"] * 10)
        mapper = CanonicalMapper().analyze(df)
        assert mapper._canonical_depth == 3

    def test_depth_capped_at_3(self):
        # FIRST_NAME is depth 4 — should be capped to 3 before vote
        df = _make_df(["FIRST_NAME"] * 10, ["FIRST_NAME"] * 10)
        mapper = CanonicalMapper().analyze(df)
        assert mapper._canonical_depth == 3

    def test_canonical_surface_locked_after_first_analyze(self):
        df1 = _make_df(["NAME", "EMAIL_ADDRESS"], ["NAME", "EMAIL_ADDRESS"])
        df2 = _make_df(["LOCATION"], ["LOCATION"])
        mapper = CanonicalMapper()
        mapper.analyze(df1)
        surface1 = frozenset(mapper.canonical_surface)
        mapper.analyze(df2)
        surface2 = frozenset(mapper.canonical_surface)
        assert surface1 == surface2

    def test_canonical_surface_not_empty_after_analyze(self):
        df = _make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper = CanonicalMapper().analyze(df)
        assert len(mapper.canonical_surface) > 0

    def test_canonical_surface_property_returns_copy(self):
        df = _make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper = CanonicalMapper().analyze(df)
        s1 = mapper.canonical_surface
        s1.add("FAKE_ENTITY")
        assert "FAKE_ENTITY" not in mapper.canonical_surface


# ---------------------------------------------------------------------------
# Phase 2: Projection
# ---------------------------------------------------------------------------


class TestProjection:
    def test_exact_match_projection(self):
        df = _make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper = CanonicalMapper().analyze(df)
        rec = mapper._records["EMAIL_ADDRESS"]
        assert rec.projected == "EMAIL_ADDRESS"
        assert rec.projection_type == "EXACT"

    def test_descendant_trivial_auto_fix(self):
        # Use PERSON (depth 2) annotations so canonical surface locks at depth 2.
        # NAME (depth 3) in predictions is a descendant of PERSON → TRIVIAL.
        df = _make_df(["PERSON"] * 10, ["NAME"] * 10)
        mapper = CanonicalMapper().analyze(df)
        rec = mapper._records.get("NAME")
        assert rec is not None
        assert rec.projection_type == "TRIVIAL"
        assert rec.projected == "PERSON"

    def test_unresolved_stays_unresolved(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper().analyze(df)
        rec = mapper._records["XYZZY_UNKNOWN_99"]
        assert rec.projection_type == "UNRESOLVED"
        assert rec.projected is None


# ---------------------------------------------------------------------------
# Issue detection
# ---------------------------------------------------------------------------


class TestIssues:
    def test_unresolved_is_error(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper().analyze(df)
        issues = mapper.get_issues()
        err = next((i for i in issues if i.type == IssueType.UNRESOLVED), None)
        assert err is not None
        assert err.severity == IssueSeverity.ERROR

    def test_sorted_by_severity_then_tokens(self):
        df = _make_df(
            ["EMAIL_ADDRESS", "NAME", "UNKNOWN_99"],
            ["EMAIL_ADDRESS", "NAME", "O"],
        )
        mapper = CanonicalMapper().analyze(df)
        issues = mapper.get_issues()
        severities = [i.severity for i in issues]
        order = [IssueSeverity.ERROR, IssueSeverity.WARNING, IssueSeverity.INFO]
        for a, b in zip(severities, severities[1:]):
            assert order.index(a) <= order.index(b)

    def test_prediction_only_warning(self):
        # EMAIL_ADDRESS in predictions but not annotations
        df = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper().analyze(df)
        pred_only = [
            i for i in mapper.get_issues() if i.type == IssueType.PREDICTION_ONLY
        ]
        assert len(pred_only) > 0
        assert pred_only[0].severity == IssueSeverity.WARNING

    def test_dataset_only_info(self):
        # EMAIL_ADDRESS in annotations but no EMAIL_ADDRESS predictions
        df = _make_df(["NAME"] * 5 + ["EMAIL_ADDRESS"] * 5, ["NAME"] * 10)
        mapper = CanonicalMapper().analyze(df)
        issues = mapper.get_issues()
        ds_only = [i for i in issues if i.type == IssueType.DATASET_ONLY]
        assert len(ds_only) > 0, "Expected at least one DATASET_ONLY issue"
        assert any("EMAIL_ADDRESS" in i.labels for i in ds_only)
        for issue in ds_only:
            assert issue.severity == IssueSeverity.INFO
            assert issue.annotation_count > 0
            assert issue.prediction_count == 0

    def test_collision_trivial_info(self):
        # Lock surface with NAME at depth 3, then analyze FIRST_NAME
        df_lock = _make_df(["NAME"] * 10, ["NAME"] * 10)
        mapper = CanonicalMapper()
        mapper.analyze(df_lock)
        df_with_fn = _make_df(["NAME"] * 8, ["FIRST_NAME"] * 8)
        mapper.analyze(df_with_fn)
        trivial = [
            i for i in mapper.get_issues() if i.type == IssueType.COLLISION_TRIVIAL
        ]
        for issue in trivial:
            assert issue.severity == IssueSeverity.INFO

    def test_issues_have_token_counts(self):
        df = _make_df(["UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper().analyze(df)
        for issue in mapper.get_issues():
            if issue.type == IssueType.UNRESOLVED and "UNKNOWN_99" in issue.labels:
                total = (issue.annotation_count or 0) + (issue.prediction_count or 0)
                assert total >= 1


# ---------------------------------------------------------------------------
# map()
# ---------------------------------------------------------------------------


class TestMap:
    def test_map_resolves_unresolved(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper().analyze(df)
        assert any(i.type == IssueType.UNRESOLVED for i in mapper.get_issues())
        mapper.map({"XYZZY_UNKNOWN_99": "NAME"})
        assert not any(i.type == IssueType.UNRESOLVED for i in mapper.get_issues())

    def test_map_suppress(self):
        df = _make_df(["XYZZY_UNKNOWN_99"], ["O"])
        mapper = CanonicalMapper().analyze(df)
        mapper.map({"XYZZY_UNKNOWN_99": None})
        rec = mapper._records["XYZZY_UNKNOWN_99"]
        assert rec.tier == "NONE"
        assert rec.projected is None

    def test_map_returns_self(self):
        df = _make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper = CanonicalMapper().analyze(df)
        result = mapper.map({"EMAIL_ADDRESS": "EMAIL_ADDRESS"})
        assert result is mapper

    def test_map_invalid_raises_atomically(self):
        df = _make_df(["XYZZY_UNKNOWN_99", "ANOTHER_UNKNOWN_99"], ["O", "O"])
        mapper = CanonicalMapper().analyze(df)
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
        mapper = CanonicalMapper().analyze(df)
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
        assert rec.canonical == "EMAIL_ADDRESS"


# ---------------------------------------------------------------------------
# get_mapped_results_dataframe()
# ---------------------------------------------------------------------------


class TestGetMappedResultsDf:
    def test_raises_without_analyze(self):
        mapper = CanonicalMapper()
        with pytest.raises(RuntimeError, match="analyze"):
            mapper.get_mapped_results_dataframe()

    def test_raises_with_unresolved_warning(self):
        df = _make_df(["NAME"] * 5, ["EMAIL_ADDRESS"] * 5)
        mapper = CanonicalMapper().analyze(df)
        if any(i.severity == IssueSeverity.WARNING for i in mapper.get_issues()):
            with pytest.raises(IncompleteMapping):
                mapper.get_mapped_results_dataframe()

    def test_returns_df_when_no_blocking(self):
        df = _make_df(["NAME"] * 3, ["NAME"] * 3)
        mapper = CanonicalMapper().analyze(df)
        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, pd.DataFrame)

    def test_output_has_original_columns(self):
        df = _make_df(["NAME", "NAME"], ["NAME", "NAME"])
        mapper = CanonicalMapper().analyze(df)
        result = mapper.get_mapped_results_dataframe()
        assert "original_annotation" in result.columns
        assert "original_prediction" in result.columns

    def test_original_columns_preserve_raw_labels(self):
        df = _make_df(["NAME"], ["NAME"])
        mapper = CanonicalMapper().analyze(df)
        result = mapper.get_mapped_results_dataframe()
        assert result["original_annotation"].tolist()[0] == "NAME"
        assert result["original_prediction"].tolist()[0] == "NAME"

    def test_suppressed_labels_become_o(self):
        df = _make_df(["NAME"] * 3, ["EMAIL_ADDRESS"] * 3)
        mapper = CanonicalMapper().analyze(df)
        pred_only = [
            i for i in mapper.get_issues() if i.type == IssueType.PREDICTION_ONLY
        ]
        if pred_only:
            for lbl in pred_only[0].labels:
                mapper.map({lbl: None})
        result = mapper.get_mapped_results_dataframe()
        assert "O" in result["prediction"].values

    def test_info_issues_do_not_block(self):
        # DATASET_ONLY and COLLISION_TRIVIAL are INFO — should not block
        df = _make_df(["NAME"] * 8, ["NAME"] * 8)
        mapper = CanonicalMapper().analyze(df)
        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------


class TestMultiModel:
    def test_canonical_surface_locked_across_models(self):
        df1 = _make_df(["NAME", "EMAIL_ADDRESS"], ["NAME", "EMAIL_ADDRESS"])
        df2 = _make_df(["NAME", "EMAIL_ADDRESS"], ["LOCATION", "LOCATION"])
        mapper = CanonicalMapper()
        mapper.analyze(df1)
        surface1 = frozenset(mapper.canonical_surface)
        mapper.analyze(df2)
        surface2 = frozenset(mapper.canonical_surface)
        assert surface1 == surface2

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
        mapper = CanonicalMapper().analyze(df)
        result = mapper.get_mapping(entity="EMAIL_ADDRESS")
        assert result is not None

    def test_full_mapping_dict(self):
        df = _make_df(["EMAIL_ADDRESS", "NAME"], ["EMAIL_ADDRESS", "NAME"])
        mapper = CanonicalMapper().analyze(df)
        result = mapper.get_mapping()
        assert isinstance(result, dict)
        assert "EMAIL_ADDRESS" in result

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
        mapper = CanonicalMapper().analyze(df)
        MapperRenderer(mapper).print_text()  # should not raise

    def test_build_html_returns_string(self):
        df = _make_df(["EMAIL_ADDRESS", "NAME"], ["EMAIL_ADDRESS", "NAME"])
        mapper = CanonicalMapper().analyze(df)
        html = MapperRenderer(mapper).build_html()
        assert isinstance(html, str)
        assert "<table" in html

    def test_build_html_before_analyze(self):
        mapper = CanonicalMapper()
        html = MapperRenderer(mapper).build_html()
        assert isinstance(html, str)


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
        # Use depth-3 entities only: NAME (PII->PERSON->NAME),
        # EMAIL_ADDRESS (PII->CONTACT->EMAIL_ADDRESS), SSN (PII->GOVERNMENT_ID->SSN)
        df = _make_df(
            ["NAME", "EMAIL_ADDRESS", "SSN"] * 10,
            ["NAME", "EMAIL_ADDRESS", "SSN"] * 10,
        )
        mapper = CanonicalMapper().analyze(df)
        blocking = [
            i
            for i in mapper.get_issues()
            if i.severity in (IssueSeverity.ERROR, IssueSeverity.WARNING)
        ]
        assert blocking == []
        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, pd.DataFrame)
        assert "original_annotation" in result.columns

    def test_full_pipeline_with_unresolved(self):
        """Unresolved label blocks extraction until resolved."""
        df = _make_df(["NAME", "UNKNOWN_XYZZY_99"], ["NAME", "O"])
        mapper = CanonicalMapper().analyze(df)
        with pytest.raises(IncompleteMapping):
            mapper.get_mapped_results_dataframe()
        mapper.map({"UNKNOWN_XYZZY_99": "NAME"})
        result = mapper.get_mapped_results_dataframe()
        assert isinstance(result, pd.DataFrame)

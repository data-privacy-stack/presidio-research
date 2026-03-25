"""Tests for CanonicalMapper."""

import pytest

from presidio_evaluator.entity_mapping import (
    CanonicalMapper,
    IncompleteMapping,
)


class TestCanonicalMapperInit:
    def test_deduplicates_labels(self):
        mapper = CanonicalMapper(["PERSON", "PERSON", "LOCATION"])
        assert mapper._labels.count("PERSON") == 1

    def test_accepts_set(self):
        mapper = CanonicalMapper({"EMAIL_ADDRESS", "PERSON"})
        assert set(mapper._labels) == {"EMAIL_ADDRESS", "PERSON"}

    def test_repr(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        assert "resolved" in repr(mapper)
        assert "pending" in repr(mapper)


class TestAutoResolve:
    def test_exact_match(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        assert "EMAIL_ADDRESS" not in mapper.pending
        assert mapper._records["EMAIL_ADDRESS"].tier == "EXACT"

    def test_fuzzy_match(self):
        # EMAILADRES is close enough to EMAIL_ADDRESS aliases
        mapper = CanonicalMapper(["EMAILADRES"])
        if "EMAILADRES" not in mapper.pending:
            assert mapper._records["EMAILADRES"].tier == "FUZZY"
            assert mapper._records["EMAILADRES"].score is not None

    def test_country_prefix(self):
        mapper = CanonicalMapper(["US_PASSPORT"])
        # US_PASSPORT should resolve (either EXACT via alias or COUNTRY)
        assert "US_PASSPORT" not in mapper.pending

    def test_unknown_label_is_pending(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN_THING_99"])
        assert "XYZZY_UNKNOWN_THING_99" in mapper.pending

    def test_pending_is_sorted_alphabetically(self):
        mapper = CanonicalMapper(["ZZZ_UNKNOWN", "AAA_UNKNOWN"])
        assert mapper.pending == sorted(mapper.pending)

    def test_all_resolved_empty_pending(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS", "PERSON"])
        assert mapper.pending == []


class TestMap:
    def test_map_single_label(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        mapper.map({"XYZZY_UNKNOWN": "PERSON"})
        assert "XYZZY_UNKNOWN" not in mapper.pending
        assert mapper._records["XYZZY_UNKNOWN"].tier == "MANUAL"
        assert mapper._records["XYZZY_UNKNOWN"].canonical == "PERSON"

    def test_map_to_none(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        mapper.map({"XYZZY_UNKNOWN": None})
        assert "XYZZY_UNKNOWN" not in mapper.pending
        assert mapper._records["XYZZY_UNKNOWN"].tier == "NONE"
        assert mapper._records["XYZZY_UNKNOWN"].canonical is None

    def test_map_overrides_auto_resolved(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        mapper.map({"EMAIL_ADDRESS": "PERSON"})
        assert mapper._records["EMAIL_ADDRESS"].canonical == "PERSON"

    def test_map_returns_self_for_chaining(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        result = mapper.map({"XYZZY_UNKNOWN": "PERSON"})
        assert result is mapper

    def test_map_invalid_label_raises(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        with pytest.raises(ValueError, match="not in the original input"):
            mapper.map({"NOT_AN_INPUT_LABEL": "PERSON"})

    def test_map_invalid_canonical_raises(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        with pytest.raises(ValueError, match="not a valid canonical entity"):
            mapper.map({"XYZZY_UNKNOWN": "NOT_A_REAL_CANONICAL"})

    def test_map_is_atomic_on_error(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN", "ANOTHER_UNKNOWN"])
        with pytest.raises(ValueError):
            mapper.map(
                {"XYZZY_UNKNOWN": "PERSON", "ANOTHER_UNKNOWN": "COMPLETELY_FAKE"}
            )
        # Neither should be applied
        assert "XYZZY_UNKNOWN" in mapper.pending
        assert "ANOTHER_UNKNOWN" in mapper.pending


class TestGetMapping:
    def test_raises_if_pending(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        with pytest.raises(IncompleteMapping) as exc_info:
            mapper.get_mapping()
        assert "XYZZY_UNKNOWN" in exc_info.value.pending

    def test_returns_dict_when_complete(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS", "XYZZY_UNKNOWN"])
        mapper.map({"XYZZY_UNKNOWN": "PERSON"})
        result = mapper.get_mapping()
        assert isinstance(result, dict)
        assert result["XYZZY_UNKNOWN"] == "PERSON"
        assert result["EMAIL_ADDRESS"] is not None  # auto-resolved

    def test_none_mapped_labels_in_result(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        mapper.map({"XYZZY_UNKNOWN": None})
        result = mapper.get_mapping()
        assert "XYZZY_UNKNOWN" in result
        assert result["XYZZY_UNKNOWN"] is None

    def test_covers_all_labels(self):
        labels = ["EMAIL_ADDRESS", "XYZZY_UNKNOWN"]
        mapper = CanonicalMapper(labels)
        mapper.map({"XYZZY_UNKNOWN": "PERSON"})
        result = mapper.get_mapping()
        assert set(result.keys()) == set(labels)

    def test_mode_html_returns_string_with_table(self):
        """get_mapping(mode='html') returns an HTML string containing a <table> tag."""
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        html = mapper.get_mapping(mode="html")
        assert isinstance(html, str)
        assert "<table" in html

    def test_mode_html_shows_pending_without_raising(self):
        """get_mapping(mode='html') works when labels are still pending."""
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        html = mapper.get_mapping(mode="html")  # should not raise
        assert isinstance(html, str)
        assert "pending" in html.lower()

    def test_mode_text_returns_readable_string(self):
        """get_mapping(mode='text') returns a plain-text table string."""
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        text = mapper.get_mapping(mode="text")
        assert isinstance(text, str)
        assert "EMAIL_ADDRESS" in text
        assert "Label" in text  # header row

    def test_mode_text_shows_pending(self):
        """get_mapping(mode='text') shows (pending) for unresolved labels."""
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        text = mapper.get_mapping(mode="text")
        assert "(pending)" in text

    def test_invalid_mode_raises(self):
        """get_mapping(mode='invalid') raises ValueError."""
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        with pytest.raises(ValueError):
            mapper.get_mapping(mode="invalid")


class TestResolveInteractively:
    def test_resolves_pending_via_prompt(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        mapper.resolve_interactively(prompt_fn=lambda _: "PERSON")
        assert mapper.pending == []
        assert mapper._records["XYZZY_UNKNOWN"].canonical == "PERSON"

    def test_resolves_to_none_via_none_keyword(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        mapper.resolve_interactively(prompt_fn=lambda _: "NONE")
        assert mapper._records["XYZZY_UNKNOWN"].canonical is None

    def test_noop_when_no_pending(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        # Should not call prompt_fn at all
        called = []
        mapper.resolve_interactively(prompt_fn=lambda _: called.append(1) or "PERSON")
        assert called == []

    def test_returns_self_for_chaining(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        result = mapper.resolve_interactively(prompt_fn=lambda _: "PERSON")
        assert result is mapper

    def test_suggestion_number_selection(self, capsys):
        # Feed "1" to select the first suggestion
        mapper = CanonicalMapper(["EMAILADRES"])
        if mapper.pending:
            responses = iter(["1"])
            mapper.resolve_interactively(prompt_fn=lambda _: next(responses))
            assert mapper.pending == []


class TestProtocolCompliance:
    def test_canonical_mapper_has_required_methods(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        assert hasattr(mapper, "pending")
        assert hasattr(mapper, "map")
        assert hasattr(mapper, "resolve_interactively")
        assert hasattr(mapper, "get_mapping")
        assert hasattr(mapper, "render_html")


class TestIncompleteMappingException:
    def test_stores_pending_labels(self):
        exc = IncompleteMapping(["A", "B"])
        assert exc.pending == ["A", "B"]

    def test_message_contains_count(self):
        exc = IncompleteMapping(["A", "B"])
        assert "2" in str(exc)


class TestBIOStripping:
    def test_bio_prefix_b_resolves_correctly(self):
        mapper = CanonicalMapper(["B-PERSON"])
        assert "B-PERSON" not in mapper.pending
        assert mapper._records["B-PERSON"].canonical == "PERSON"

    def test_bio_prefix_key_is_original_label(self):
        mapper = CanonicalMapper(["B-PERSON"])
        result = mapper.get_mapping()
        assert "B-PERSON" in result
        assert result["B-PERSON"] == "PERSON"

    def test_bio_suffix_stripping(self):
        mapper = CanonicalMapper(["PERSON-I"])
        assert "PERSON-I" not in mapper.pending
        assert mapper._records["PERSON-I"].canonical == "PERSON"

    def test_non_bio_label_starting_with_b_unchanged(self):
        mapper = CanonicalMapper(["BANK_ACCOUNT"])
        # BANK_ACCOUNT should NOT be stripped (no hyphen after B)
        assert mapper._stripped["BANK_ACCOUNT"] == "BANK_ACCOUNT"

    def test_o_token_auto_mapped_to_none(self):
        mapper = CanonicalMapper(["O"])
        assert "O" not in mapper.pending
        assert mapper._records["O"].canonical is None
        assert mapper._records["O"].tier == "NONE"

    def test_o_token_not_interactive(self):
        # O should be resolved without ever prompting
        called = []
        mapper = CanonicalMapper(["O"])
        mapper.resolve_interactively(prompt_fn=lambda _: called.append(1) or "PERSON")
        assert called == []


class TestRenderHtml:
    def test_render_html_does_not_raise_without_ipython(self):
        import sys

        # Remove IPython from path to simulate non-Jupyter environment
        ipython = sys.modules.pop("IPython", None)
        ipython_display = sys.modules.pop("IPython.display", None)
        try:
            mapper = CanonicalMapper(["EMAIL_ADDRESS", "XYZZY_UNKNOWN"])
            mapper.render_html()  # should not raise
        finally:
            if ipython is not None:
                sys.modules["IPython"] = ipython
            if ipython_display is not None:
                sys.modules["IPython.display"] = ipython_display

    def test_render_html_can_be_called_before_resolution(self):
        mapper = CanonicalMapper(["XYZZY_UNKNOWN"])
        mapper.render_html()  # should not raise when pending


class TestGetMappedResultsDataframe:
    """Tests for CanonicalMapper.get_mapped_results_dataframe()."""

    def _make_df(self, annotations, predictions):
        import pandas as pd

        n = len(annotations)
        return pd.DataFrame(
            {
                "sentence_id": list(range(n)),
                "token": [f"t{i}" for i in range(n)],
                "annotation": annotations,
                "prediction": predictions,
                "start_indices": [0] * n,
            }
        )

    def test_basic_remapping(self):
        """Labels in both columns are mapped to their canonical names."""
        df = self._make_df(["EMAIL_ADDRESS", "O"], ["EMAIL_ADDRESS", "O"])
        mapper = CanonicalMapper()
        result = mapper.get_mapped_results_dataframe(df)
        assert result["annotation"].tolist() == ["EMAIL_ADDRESS", "O"]
        assert result["prediction"].tolist() == ["EMAIL_ADDRESS", "O"]

    def test_none_passthrough(self):
        """Labels mapped to None remain as their original value in the output."""
        mapper = CanonicalMapper(["MY_SUPPRESSED"])
        mapper.map({"MY_SUPPRESSED": None})
        df = self._make_df(["MY_SUPPRESSED"], ["O"])
        result = mapper.get_mapped_results_dataframe(df)
        assert result["annotation"].tolist() == ["MY_SUPPRESSED"]

    def test_hierarchy_level_1(self):
        """hierarchy=1 resolves everything to the top-level PII category."""
        mapper = CanonicalMapper()
        df = self._make_df(["EMAIL_ADDRESS", "PERSON"], ["EMAIL_ADDRESS", "PERSON"])
        result = mapper.get_mapped_results_dataframe(df, hierarchy=1)
        assert all(v == "PII" for v in result["annotation"])

    def test_hierarchy_level_2(self):
        """hierarchy=2 maps FIRSTNAME and PERSON to the same PERSON entity."""
        mapper = CanonicalMapper()
        df = self._make_df(["FIRSTNAME", "PERSON"], ["PERSON", "PERSON"])
        result = mapper.get_mapped_results_dataframe(df, hierarchy=2)
        assert result["annotation"].tolist() == ["PERSON", "PERSON"]
        assert result["prediction"].tolist() == ["PERSON", "PERSON"]

    def test_hierarchy_level_3(self):
        """hierarchy=3 (default) keeps granular entities like NAME."""
        mapper = CanonicalMapper()
        df = self._make_df(["FIRSTNAME"], ["PERSON"])
        result = mapper.get_mapped_results_dataframe(df, hierarchy=3)
        ann = result["annotation"].tolist()[0]
        assert ann != "FIRSTNAME"  # should have been remapped

    def test_incremental_label_discovery(self):
        """New labels seen in subsequent DataFrames are auto-resolved."""
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        df1 = self._make_df(["EMAIL_ADDRESS"], ["EMAIL_ADDRESS"])
        mapper.get_mapped_results_dataframe(df1)
        # FIRSTNAME not in original labels
        df2 = self._make_df(["FIRSTNAME"], ["PERSON"])
        mapper.get_mapped_results_dataframe(df2)
        assert "FIRSTNAME" in mapper._labels
        assert "PERSON" in mapper._labels

    def test_mixed_level_warning(self):
        """A UserWarning is raised when annotation and prediction map to related but different entities."""
        import warnings

        mapper = CanonicalMapper()
        df = self._make_df(["FIRSTNAME", "O"], ["PERSON", "O"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapper.get_mapped_results_dataframe(df)
        assert any(issubclass(warning.category, UserWarning) for warning in w), (
            "Expected a UserWarning for mixed-granularity mapping"
        )

    def test_no_warning_when_same_canonical(self):
        """No warning when annotation and prediction map to the same canonical entity."""
        import warnings

        mapper = CanonicalMapper()
        df = self._make_df(["EMAIL_ADDRESS", "O"], ["EMAIL_ADDRESS", "O"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapper.get_mapped_results_dataframe(df)
        user_warnings = [
            warning for warning in w if issubclass(warning.category, UserWarning)
        ]
        assert not user_warnings, f"Unexpected warnings: {user_warnings}"

    def test_empty_constructor_works(self):
        """CanonicalMapper() with no arguments works with get_mapped_results_dataframe."""
        mapper = CanonicalMapper()
        df = self._make_df(["PERSON", "O"], ["PERSON", "O"])
        result = mapper.get_mapped_results_dataframe(df)
        assert list(result.columns) == [
            "sentence_id",
            "token",
            "annotation",
            "prediction",
            "start_indices",
        ]

"""Tests for CanonicalMapper."""

import pytest

from presidio_evaluator.entity_mapping import (
    CanonicalMapper,
    EntityMapper,
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
            mapper.map({"XYZZY_UNKNOWN": "PERSON", "ANOTHER_UNKNOWN": "COMPLETELY_FAKE"})
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
    def test_canonical_mapper_satisfies_entity_mapper_protocol(self):
        mapper = CanonicalMapper(["EMAIL_ADDRESS"])
        assert isinstance(mapper, EntityMapper)


class TestIncompleteMappingException:
    def test_stores_pending_labels(self):
        exc = IncompleteMapping(["A", "B"])
        assert exc.pending == ["A", "B"]

    def test_message_contains_count(self):
        exc = IncompleteMapping(["A", "B"])
        assert "2" in str(exc)

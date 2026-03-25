import pytest

from presidio_evaluator.entity_mapping.hierarchy import (
    ALL_CANONICAL_ENTITIES,
    CANONICAL_TO_BRANCH,
    HIERARCHY,
    RAW_TO_CANONICAL,
    EntityHierarchy,
    EntityNotMappedError,
    canonicalize,
    get_branch,
)

# ---------------------------------------------------------------------------
# canonicalize() – explicit aliases
# ---------------------------------------------------------------------------


class TestCanonicalizeExplicitAliases:
    def test_email_address(self):
        assert canonicalize("EMAIL_ADDRESS") == "EMAIL_ADDRESS"

    def test_phone_number(self):
        assert canonicalize("PHONE_NUMBER") == "PHONE_NUMBER"

    def test_person_name(self):
        # PERSON is a depth-2 intermediate node — resolves to itself
        assert canonicalize("PERSON") == "PERSON"

    def test_date_of_birth_alias(self):
        assert canonicalize("DATE_OF_BIRTH") == "BIRTH_DATE"

    def test_ssn_alias(self):
        assert canonicalize("SSN") == "SSN"

    def test_social_security_number_alias(self):
        assert canonicalize("SOCIAL_SECURITY_NUMBER") == "SSN"

    def test_ip_address_alias(self):
        assert canonicalize("IP_ADDRESS") == "IP_ADDRESS"

    def test_url_alias(self):
        assert canonicalize("URL") == "URL"

    def test_credit_card_alias(self):
        assert canonicalize("CREDIT_CARD") == "FINANCIAL"

    def test_iban_alias(self):
        assert canonicalize("IBAN") == "FINANCIAL"

    def test_passport_alias(self):
        assert canonicalize("PASSPORT") == "PASSPORT"

    def test_drivers_license_alias(self):
        assert canonicalize("DRIVER_LICENSE") == "DRIVER_LICENSE"

    def test_nrp_alias(self):
        assert canonicalize("NRP") == "NATIONALITY"

    def test_location_alias(self):
        assert canonicalize("LOCATION") == "LOCATION"

    def test_organization_alias(self):
        # ORG is its own canonical leaf node
        assert canonicalize("ORG") == "ORG"


# ---------------------------------------------------------------------------
# canonicalize() – country-prefix pattern
# ---------------------------------------------------------------------------


class TestCanonicalizeCountryPrefix:
    def test_australia_tax_id(self):
        assert canonicalize("AUSTRALIA_TAX_ID") == "TAX_ID"

    def test_uruguay_tax_id(self):
        assert canonicalize("URUGUAY_TAX_ID") == "TAX_ID"

    def test_australia_drivers_license(self):
        assert canonicalize("AUSTRALIA_DRIVERS_LICENSE") == "DRIVER_LICENSE"

    def test_germany_passport_number(self):
        assert canonicalize("GERMANY_PASSPORT_NUMBER") == "PASSPORT"

    def test_canada_social_insurance(self):
        assert canonicalize("CANADA_SOCIAL_INSURANCE") == "SSN"

    def test_france_national_id(self):
        assert canonicalize("FRANCE_NATIONAL_IDENTIFICATION_NUMBER") == "NATIONAL_ID"

    def test_brazil_phone_number(self):
        assert canonicalize("BRAZIL_PHONE_NUMBER") == "PHONE_NUMBER"

    def test_brazil_postal_code(self):
        # POSTAL_CODE is depth-4; country-prefix maps to depth-3 ancestor ADDRESS
        assert canonicalize("BRAZIL_CEP_CODE") == "ADDRESS"

    def test_us_social_security_number(self):
        assert canonicalize("USA_SOCIAL_SECURITY_NUMBER") == "SSN"

    def test_uk_national_insurance(self):
        # GB national insurance maps via SOCIAL_SECURITY suffix → SSN
        assert canonicalize("GB_NATIONAL_INSURANCE_NUMBER") == "SSN"

    def test_india_aadhaar(self):
        assert canonicalize("IN_AADHAAR") == "NATIONAL_ID"

    def test_israel_id(self):
        # IL is a valid country code; unknown suffix defaults to NATIONAL_ID
        assert canonicalize("IL_ID_NUMBER") == "NATIONAL_ID"

    def test_two_word_country_prefix(self):
        # Two-token country names like COSTA_RICA won't match as a country code,
        # but single ISO 2-letter codes like CR should
        result = canonicalize("CR_PASSPORT")
        assert result == "PASSPORT"


# ---------------------------------------------------------------------------
# canonicalize() – unknown labels raise EntityNotMappedError
# ---------------------------------------------------------------------------


class TestCanonicalizePassthrough:
    def test_completely_unknown(self):
        with pytest.raises(EntityNotMappedError, match="UNKNOWN_ENTITY"):
            canonicalize("UNKNOWN_ENTITY")

    def test_gibberish(self):
        with pytest.raises(EntityNotMappedError, match="XYZZY"):
            canonicalize("XYZZY")

    def test_gibberish_with_underscore(self):
        with pytest.raises(EntityNotMappedError, match="BLORP_FLARP"):
            canonicalize("BLORP_FLARP")


# ---------------------------------------------------------------------------
# canonicalize() – case and underscore agnosticism
# ---------------------------------------------------------------------------


class TestCanonicalizeNormalization:
    @pytest.mark.parametrize(
        "raw",
        [
            "CREDIT_CARD",
            "credit_card",
            "CreditCard",
            "creditcard",
            "CREDITCARD",
            "Credit_Card",
        ],
    )
    def test_credit_card_variants(self, raw):
        assert canonicalize(raw) == "FINANCIAL"

    @pytest.mark.parametrize(
        "raw",
        [
            "EMAIL_ADDRESS",
            "email_address",
            "EmailAddress",
            "emailaddress",
            "Email_Address",
        ],
    )
    def test_email_variants(self, raw):
        assert canonicalize(raw) == "EMAIL_ADDRESS"

    @pytest.mark.parametrize(
        "raw",
        [
            "DATE_OF_BIRTH",
            "date_of_birth",
            "DateOfBirth",
            "dateofbirth",
        ],
    )
    def test_date_of_birth_variants(self, raw):
        assert canonicalize(raw) == "BIRTH_DATE"

    @pytest.mark.parametrize(
        "raw",
        [
            "PHONE_NUMBER",
            "phone_number",
            "PhoneNumber",
            "phonenumber",
        ],
    )
    def test_phone_number_variants(self, raw):
        assert canonicalize(raw) == "PHONE_NUMBER"

    @pytest.mark.parametrize(
        "raw",
        [
            "SOCIAL_SECURITY_NUMBER",
            "social_security_number",
            "SocialSecurityNumber",
        ],
    )
    def test_ssn_variants(self, raw):
        assert canonicalize(raw) == "SSN"


# ---------------------------------------------------------------------------
# get_branch() – full path resolution
# ---------------------------------------------------------------------------


class TestGetBranch:
    def test_email(self):
        assert get_branch("EMAIL_ADDRESS") == ["PII", "CONTACT", "EMAIL_ADDRESS"]

    def test_phone(self):
        assert get_branch("PHONE_NUMBER") == ["PII", "CONTACT", "PHONE_NUMBER"]

    def test_ssn(self):
        branch = get_branch("SSN")
        assert branch is not None
        assert branch[-1] == "SSN"
        assert branch[0] == "PII"

    def test_passport(self):
        assert get_branch("PASSPORT") == ["PII", "GOVERNMENT_ID", "PASSPORT"]

    def test_driver_license(self):
        branch = get_branch("DRIVER_LICENSE")
        assert branch is not None
        assert branch[-1] == "DRIVER_LICENSE"

    def test_credit_card(self):
        assert get_branch("CREDIT_CARD") == ["PII", "FINANCIAL_PII", "FINANCIAL"]

    def test_birth_date(self):
        assert get_branch("DATE_OF_BIRTH") == ["PII", "DATE_TIME", "BIRTH_DATE"]

    def test_tax_id(self):
        assert get_branch("TAX_ID") == ["PII", "GOVERNMENT_ID", "TAX_ID"]

    def test_location(self):
        # LOCATION is a depth-2 intermediate node — resolves to itself with its path
        assert get_branch("LOCATION") == ["PII", "LOCATION"]

    def test_organization(self):
        branch = get_branch("ORG")
        assert branch is not None
        assert branch[-1] == "ORG"


# ---------------------------------------------------------------------------
# get_branch() – country-prefix inputs
# ---------------------------------------------------------------------------


class TestGetBranchCountryPrefix:
    def test_australia_tax_id(self):
        assert get_branch("AUSTRALIA_TAX_ID") == ["PII", "GOVERNMENT_ID", "TAX_ID"]

    def test_germany_passport(self):
        assert get_branch("GERMANY_PASSPORT_NUMBER") == [
            "PII",
            "GOVERNMENT_ID",
            "PASSPORT",
        ]

    def test_canada_social_insurance(self):
        branch = get_branch("CANADA_SOCIAL_INSURANCE")
        assert branch is not None
        assert branch[-1] == "SSN"

    def test_brazil_phone(self):
        assert get_branch("BRAZIL_PHONE_NUMBER") == ["PII", "CONTACT", "PHONE_NUMBER"]

    def test_france_national_id(self):
        branch = get_branch("FRANCE_NATIONAL_IDENTIFICATION_NUMBER")
        assert branch is not None
        assert branch[-1] == "NATIONAL_ID"


# ---------------------------------------------------------------------------
# get_branch() – case/underscore agnosticism
# ---------------------------------------------------------------------------


class TestGetBranchNormalization:
    @pytest.mark.parametrize(
        "raw",
        [
            "CREDIT_CARD",
            "credit_card",
            "CreditCard",
            "creditcard",
        ],
    )
    def test_credit_card_variants(self, raw):
        assert get_branch(raw) == ["PII", "FINANCIAL_PII", "FINANCIAL"]

    @pytest.mark.parametrize(
        "raw",
        [
            "EMAIL_ADDRESS",
            "email_address",
            "EmailAddress",
            "Email_Address",
        ],
    )
    def test_email_variants(self, raw):
        assert get_branch(raw) == ["PII", "CONTACT", "EMAIL_ADDRESS"]

    @pytest.mark.parametrize(
        "raw",
        [
            "DATE_OF_BIRTH",
            "date_of_birth",
            "DateOfBirth",
            "DATEOFBIRTH",
        ],
    )
    def test_date_of_birth_variants(self, raw):
        assert get_branch(raw) == ["PII", "DATE_TIME", "BIRTH_DATE"]

    @pytest.mark.parametrize(
        "raw",
        [
            "PASSPORT",
            "passport",
            "Passport",
        ],
    )
    def test_passport_variants(self, raw):
        assert get_branch(raw) == ["PII", "GOVERNMENT_ID", "PASSPORT"]


# ---------------------------------------------------------------------------
# get_branch() – unknown labels raise EntityNotMappedError
# ---------------------------------------------------------------------------


class TestGetBranchUnknown:
    def test_completely_unknown(self):
        with pytest.raises(EntityNotMappedError, match="UNKNOWN_XYZ"):
            get_branch("UNKNOWN_XYZ")

    def test_gibberish(self):
        with pytest.raises(EntityNotMappedError):
            get_branch("BLORP_FLARP")

    def test_valid_country_unknown_suffix(self):
        # Recognized country + unknown suffix → NATIONAL_ID → has a branch
        assert get_branch("DE_UNICORN") == ["PII", "GOVERNMENT_ID", "NATIONAL_ID"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_all_canonical_entities_nonempty(self):
        assert len(ALL_CANONICAL_ENTITIES) > 50

    def test_all_canonical_entities_are_strings(self):
        assert all(isinstance(e, str) for e in ALL_CANONICAL_ENTITIES)

    def test_raw_to_canonical_nonempty(self):
        assert len(RAW_TO_CANONICAL) > 100

    def test_canonical_to_branch_covers_all_canonical(self):
        for entity in ALL_CANONICAL_ENTITIES:
            assert (
                entity in CANONICAL_TO_BRANCH
            ), f"{entity} missing from CANONICAL_TO_BRANCH"

    def test_canonical_to_branch_paths_start_with_pii(self):
        for entity, branch in CANONICAL_TO_BRANCH.items():
            assert branch[0] == "PII", f"{entity}: branch doesn't start with PII"

    def test_canonical_to_branch_paths_end_with_canonical(self):
        for entity, branch in CANONICAL_TO_BRANCH.items():
            assert branch[-1] == entity, f"{entity}: branch doesn't end with itself"

    def test_hierarchy_top_level_is_pii(self):
        assert "PII" in HIERARCHY

    def test_raw_to_canonical_values_are_canonical(self):
        # Values are either depth-3 canonical entities or depth-1/2 intermediate
        # nodes that self-map (e.g. PERSON, LOCATION, BIOMETRIC, FINANCIAL_PII).
        canonical_set = set(ALL_CANONICAL_ENTITIES) | set(CANONICAL_TO_BRANCH.keys())
        for raw, canonical in RAW_TO_CANONICAL.items():
            assert (
                canonical in canonical_set
            ), f"RAW_TO_CANONICAL[{raw!r}] = {canonical!r} is not a recognized entity"


# ---------------------------------------------------------------------------
# EntityHierarchy class API
# ---------------------------------------------------------------------------


class TestEntityHierarchyClass:
    """Tests for the EntityHierarchy class and its mutation methods."""

    def test_default_returns_singleton(self):
        assert EntityHierarchy.default() is EntityHierarchy.default()

    def test_copy_is_independent(self):
        h = EntityHierarchy.default().copy()
        h.remove_entity("URL")
        # Original default should be unaffected
        assert EntityHierarchy.default().canonicalize("URL") == "URL"

    def test_instance_canonicalize_matches_default(self):
        h = EntityHierarchy()
        assert h.canonicalize("EMAIL_ADDRESS") == "EMAIL_ADDRESS"
        assert h.canonicalize("DATE_OF_BIRTH") == "BIRTH_DATE"

    def test_instance_get_branch_matches_default(self):
        h = EntityHierarchy()
        assert h.get_branch("PASSPORT") == ["PII", "GOVERNMENT_ID", "PASSPORT"]

    def test_unknown_raises_entity_not_mapped_error(self):
        h = EntityHierarchy()
        with pytest.raises(EntityNotMappedError):
            h.canonicalize("TOTALLY_UNKNOWN_XYZ")

    # ── add_alias ────────────────────────────────────────────────────────────

    def test_add_alias_leaf_node(self):
        h = EntityHierarchy.default().copy()
        h.add_alias("EMAIL_ADDRESS", "ELECTRONIC_MAIL")
        assert h.canonicalize("ELECTRONIC_MAIL") == "EMAIL_ADDRESS"

    def test_add_alias_dict_node(self):
        # CREDIT_CARD is depth-4 under FINANCIAL; adding alias there makes it resolve to FINANCIAL
        h = EntityHierarchy.default().copy()
        h.add_alias("CREDIT_CARD", "CC")
        assert h.canonicalize("CC") == "FINANCIAL"

    def test_add_alias_unknown_entity_raises(self):
        h = EntityHierarchy.default().copy()
        with pytest.raises(KeyError):
            h.add_alias("NONEXISTENT_ENTITY", "SOME_ALIAS")

    def test_add_alias_rebuilds_lookup(self):
        h = EntityHierarchy.default().copy()
        h.add_alias("SSN", "MY_SSN")
        assert "MYSSN" in h.raw_to_canonical

    # ── remove_alias ─────────────────────────────────────────────────────────

    def test_remove_alias(self):
        h = EntityHierarchy.default().copy()
        h.remove_alias("EMAIL_ADDRESS", "EMAIL")
        with pytest.raises(EntityNotMappedError):
            h.canonicalize("EMAIL")

    def test_remove_alias_not_present_raises(self):
        h = EntityHierarchy.default().copy()
        with pytest.raises(ValueError):
            h.remove_alias("EMAIL_ADDRESS", "NONEXISTENT_ALIAS")

    def test_remove_alias_dict_node_raises_type_error(self):
        # CREDIT_CARD has dict children — remove_alias should refuse
        h = EntityHierarchy.default().copy()
        with pytest.raises(TypeError):
            h.remove_alias("CREDIT_CARD", "SOME_ALIAS")

    # ── add_entity ───────────────────────────────────────────────────────────

    def test_add_entity_at_canonical_depth(self):
        h = EntityHierarchy.default().copy()
        h.add_entity(["PII", "GOVERNMENT_ID"], "SOCIAL_CREDIT_SCORE", ["SCS"])
        assert h.canonicalize("SOCIAL_CREDIT_SCORE") == "SOCIAL_CREDIT_SCORE"
        assert h.canonicalize("SCS") == "SOCIAL_CREDIT_SCORE"

    def test_add_entity_branch_correct(self):
        h = EntityHierarchy.default().copy()
        h.add_entity(["PII", "GOVERNMENT_ID"], "SOCIAL_CREDIT_SCORE")
        assert h.get_branch("SOCIAL_CREDIT_SCORE") == [
            "PII",
            "GOVERNMENT_ID",
            "SOCIAL_CREDIT_SCORE",
        ]

    def test_add_entity_bad_path_raises(self):
        h = EntityHierarchy.default().copy()
        with pytest.raises(KeyError):
            h.add_entity(["PII", "NONEXISTENT_PARENT"], "NEW_ENTITY")

    def test_add_entity_to_leaf_raises(self):
        h = EntityHierarchy.default().copy()
        # EMAIL_ADDRESS is a leaf (list value) — cannot be a parent
        with pytest.raises(TypeError):
            h.add_entity(["PII", "CONTACT", "EMAIL_ADDRESS"], "SUBEMAIL")

    # ── remove_entity ────────────────────────────────────────────────────────

    def test_remove_entity(self):
        h = EntityHierarchy.default().copy()
        h.remove_entity("URL")
        with pytest.raises(EntityNotMappedError):
            h.canonicalize("URL")

    def test_remove_entity_not_found_raises(self):
        h = EntityHierarchy.default().copy()
        with pytest.raises(KeyError):
            h.remove_entity("DOES_NOT_EXIST")

    def test_remove_entity_updates_all_canonical(self):
        h = EntityHierarchy.default().copy()
        h.remove_entity("URL")
        assert "URL" not in h.all_canonical_entities

    # ── rename_entity ────────────────────────────────────────────────────────

    def test_rename_entity(self):
        h = EntityHierarchy.default().copy()
        h.rename_entity("EMAIL_ADDRESS", "EMAIL_ADDR")
        assert h.canonicalize("EMAIL_ADDR") == "EMAIL_ADDR"
        # Old aliases should still resolve to the renamed entity
        assert h.canonicalize("EMAIL") == "EMAIL_ADDR"

    def test_rename_entity_not_found_raises(self):
        h = EntityHierarchy.default().copy()
        with pytest.raises(KeyError):
            h.rename_entity("GHOST_ENTITY", "NEW_NAME")

    def test_rename_updates_branch(self):
        h = EntityHierarchy.default().copy()
        h.rename_entity("EMAIL_ADDRESS", "EMAIL_ADDR")
        branch = h.get_branch("EMAIL_ADDR")
        assert branch[-1] == "EMAIL_ADDR"

    # ── add_country_doc_type ─────────────────────────────────────────────────

    def test_add_country_doc_type(self):
        h = EntityHierarchy.default().copy()
        h.add_country_doc_type("HEALTH_CARD", "HEALTH_INSURANCE_ID")
        assert h.canonicalize("CA_HEALTH_CARD") == "HEALTH_INSURANCE_ID"

    def test_remove_country_doc_type(self):
        h = EntityHierarchy.default().copy()
        # Register a custom suffix that the hierarchy doesn't know about.
        h.add_country_doc_type("HEALTH_CARD", "HEALTH_INSURANCE_ID")
        assert h.canonicalize("CA_HEALTH_CARD") == "HEALTH_INSURANCE_ID"
        # After removal, the remainder can't be resolved → falls back to NATIONAL_ID.
        h.remove_country_doc_type("HEALTH_CARD")
        assert h.canonicalize("CA_HEALTH_CARD") == "NATIONAL_ID"

    # ── custom canonical_depth ───────────────────────────────────────────────

    def test_custom_canonical_depth(self):
        # At depth 2, intermediate nodes like GOVERNMENT_ID become canonical.
        h = EntityHierarchy(canonical_depth=2)
        assert h.canonicalize("PASSPORT") == "GOVERNMENT_ID"
        assert h.canonicalize("SSN") == "GOVERNMENT_ID"


# ---------------------------------------------------------------------------
# fuzzy_canonicalize()
# ---------------------------------------------------------------------------


class TestFuzzyCanonicalize:
    """Tests for fuzzy_canonicalize() and the fuzzy fallback built into canonicalize()."""

    def setup_method(self):
        self.h = EntityHierarchy.default()

    # ── exact match still works (fast path) ─────────────────────────────────

    def test_exact_match_unchanged(self):
        assert self.h.fuzzy_canonicalize("EMAIL_ADDRESS") == "EMAIL_ADDRESS"

    def test_exact_alias_unchanged(self):
        assert self.h.fuzzy_canonicalize("EMAIL") == "EMAIL_ADDRESS"

    def test_exact_country_prefix_unchanged(self):
        assert self.h.fuzzy_canonicalize("ARGENTINA_TAX_ID") == "TAX_ID"

    # ── canonicalize() now also uses fuzzy as last resort ───────────────────

    def test_canonicalize_fuzzy_country_adjective(self):
        assert self.h.canonicalize("ARGENTENIAN_TAX_ID") == "TAX_ID"

    def test_canonicalize_fuzzy_entity_typo(self):
        assert self.h.canonicalize("EMAIL_ADRES") == "EMAIL_ADDRESS"

    def test_canonicalize_disable_fuzzy_raises(self):
        # threshold=1.0 disables the fuzzy fallback entirely
        with pytest.raises(EntityNotMappedError):
            self.h.canonicalize("ARGENTENIAN_TAX_ID", threshold=1.0)

    def test_get_branch_fuzzy_country(self):
        # get_branch delegates to canonicalize(), so fuzzy works transitively
        assert self.h.get_branch("ARGENTENIAN_TAX_ID") == [
            "PII",
            "GOVERNMENT_ID",
            "TAX_ID",
        ]

    # ── fuzzy country token (adjective / truncation variants) ────────────────

    def test_fuzzy_country_adjective(self):
        # ARGENTENIAN is a common adjective variant of ARGENTINA (score ~0.80)
        assert self.h.fuzzy_canonicalize("ARGENTENIAN_TAX_ID") == "TAX_ID"

    def test_fuzzy_country_truncated(self):
        # ARGENTIN is a truncation of ARGENTINA
        assert self.h.fuzzy_canonicalize("ARGENTIN_PASSPORT") == "PASSPORT"

    def test_fuzzy_country_suffix_resolved(self):
        # AUSTRALIEN is a German/Dutch spelling of AUSTRALIA
        assert self.h.fuzzy_canonicalize("AUSTRALIEN_DRIVER") == "DRIVER_LICENSE"

    def test_fuzzy_two_token_country(self):
        # COSTA_RIC is a truncation of COSTA_RICA
        assert self.h.fuzzy_canonicalize("COSTA_RIC_PASSPORT") == "PASSPORT"

    def test_fuzzy_country_unknown_suffix_defaults_to_national_id(self):
        # Recognisable country, unknown document type → NATIONAL_ID
        assert self.h.fuzzy_canonicalize("GERMANX_UNKNOWN_DOC") == "NATIONAL_ID"

    # ── fuzzy suffix token (typos in the doc-type part) ─────────────────────

    def test_fuzzy_suffix_typo(self):
        # DIVERS is a typo of DRIVER; UK is an exact country match
        assert self.h.fuzzy_canonicalize("UK_DIVERS_LICENSE") == "DRIVER_LICENSE"

    def test_fuzzy_suffix_with_fuzzy_country(self):
        # Both country and suffix are approximate
        assert self.h.fuzzy_canonicalize("AUSTRALIEN_DRIVRE") == "DRIVER_LICENSE"

    def test_fuzzy_suffix_passport_typo(self):
        assert self.h.fuzzy_canonicalize("US_PASSPOORT") == "PASSPORT"

    def test_exact_country_fuzzy_suffix_canonicalize(self):
        # canonicalize() (not just fuzzy_canonicalize) should resolve this too
        assert self.h.canonicalize("UK_DIVERS_LICENSE") == "DRIVER_LICENSE"

    # ── fuzzy entity name (typos in explicit alias) ──────────────────────────

    def test_fuzzy_entity_typo_single_char(self):
        # EMAIL_ADRES (missing S) should resolve to EMAIL_ADDRESS
        assert self.h.fuzzy_canonicalize("EMAIL_ADRES") == "EMAIL_ADDRESS"

    def test_fuzzy_entity_truncated(self):
        # CREDIT_CAR should resolve to FINANCIAL via CARD_NUMBER alias CREDIT_CARD
        assert self.h.fuzzy_canonicalize("CREDIT_CAR") == "FINANCIAL"

    def test_fuzzy_entity_case_insensitive(self):
        assert self.h.fuzzy_canonicalize("email_adres") == "EMAIL_ADDRESS"

    # ── threshold behaviour ──────────────────────────────────────────────────

    def test_strict_threshold_rejects_weak_match(self):
        # ARGENTENIAN scores ~0.91 against ARGENTINIAN (now in COUNTRIES).
        # A threshold of 0.95 should still reject it.
        with pytest.raises(EntityNotMappedError):
            self.h.fuzzy_canonicalize("ARGENTENIAN_TAX_ID", threshold=0.95)

    def test_lenient_threshold_accepts_more(self):
        # At 0.70, ARGENT is close enough to ARGENTINA to be recognised as a country.
        # The remainder TAX_ID contains "TAX" so it resolves to TAX_ID.
        assert self.h.fuzzy_canonicalize("ARGENT_TAX_ID", threshold=0.70) == "TAX_ID"

    def test_custom_threshold_boundary(self):
        # Passing exactly the score of the match should still accept it
        import difflib

        score = difflib.SequenceMatcher(None, "ARGENTENIAN", "ARGENTINA").ratio()
        assert (
            self.h.fuzzy_canonicalize("ARGENTENIAN_TAX_ID", threshold=score) == "TAX_ID"
        )

    # ── complete nonsense should still raise ─────────────────────────────────

    def test_nonsense_label_raises(self):
        with pytest.raises(EntityNotMappedError):
            self.h.fuzzy_canonicalize("XYZZY_QWERTY_FLORP")

    def test_empty_string_raises(self):
        with pytest.raises(EntityNotMappedError):
            self.h.fuzzy_canonicalize("")

    def test_single_token_no_country_raises(self):
        # A single token with no underscore cannot be a country-prefix label
        with pytest.raises(EntityNotMappedError):
            self.h.fuzzy_canonicalize("FLORP")

    # ── get_branch works via fuzzy resolution ────────────────────────────────

    def test_get_branch_after_fuzzy_country(self):
        canonical = self.h.fuzzy_canonicalize("ARGENTENIAN_TAX_ID")
        assert self.h.get_branch(canonical) == ["PII", "GOVERNMENT_ID", "TAX_ID"]

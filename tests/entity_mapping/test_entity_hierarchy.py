import pytest

from presidio_evaluator.entity_mapping import (
    HIERARCHY,
    EntityHierarchy,
    EntityNotMappedError,
)

_h = EntityHierarchy()

# ---------------------------------------------------------------------------
# canonicalize() – explicit aliases
# ---------------------------------------------------------------------------


class TestCanonicalizeExplicitAliases:
    def test_email_address(self):
        assert _h.canonicalize("EMAIL_ADDRESS") == "EMAIL_ADDRESS"

    def test_phone_number(self):
        assert _h.canonicalize("PHONE_NUMBER") == "PHONE_NUMBER"

    def test_person_name(self):
        # PERSON is a depth-2 intermediate node — resolves to itself
        assert _h.canonicalize("PERSON") == "PERSON"

    def test_date_of_birth_alias(self):
        assert _h.canonicalize("DATE_OF_BIRTH") == "BIRTH_DATE"

    def test_ssn_alias(self):
        assert _h.canonicalize("SSN") == "SSN"

    def test_social_security_number_alias(self):
        assert _h.canonicalize("SOCIAL_SECURITY_NUMBER") == "SSN"

    def test_ip_address_alias(self):
        assert _h.canonicalize("IP_ADDRESS") == "IP_ADDRESS"

    def test_url_alias(self):
        assert _h.canonicalize("URL") == "URL"

    def test_credit_card_alias(self):
        assert _h.canonicalize("CREDIT_CARD") == "FINANCIAL"

    def test_iban_alias(self):
        assert _h.canonicalize("IBAN") == "FINANCIAL"

    def test_passport_alias(self):
        assert _h.canonicalize("PASSPORT") == "PASSPORT"

    def test_drivers_license_alias(self):
        assert _h.canonicalize("DRIVER_LICENSE") == "DRIVER_LICENSE"

    def test_nrp_alias(self):
        assert _h.canonicalize("NRP") == "NATIONALITY"

    def test_location_alias(self):
        assert _h.canonicalize("LOCATION") == "LOCATION"

    def test_organization_alias(self):
        # ORG is its own canonical leaf node
        assert _h.canonicalize("ORG") == "ORG"


# ---------------------------------------------------------------------------
# canonicalize() – country-prefix pattern
# ---------------------------------------------------------------------------


class TestCanonicalizeCountryPrefix:
    def test_australia_tax_id(self):
        assert _h.canonicalize("AUSTRALIA_TAX_ID") == "TAX_ID"

    def test_uruguay_tax_id(self):
        assert _h.canonicalize("URUGUAY_TAX_ID") == "TAX_ID"

    def test_australia_drivers_license(self):
        assert _h.canonicalize("AUSTRALIA_DRIVERS_LICENSE") == "DRIVER_LICENSE"

    def test_germany_passport_number(self):
        assert _h.canonicalize("GERMANY_PASSPORT_NUMBER") == "PASSPORT"

    def test_canada_social_insurance(self):
        assert _h.canonicalize("CANADA_SOCIAL_INSURANCE") == "SSN"

    def test_france_national_id(self):
        assert _h.canonicalize("FRANCE_NATIONAL_IDENTIFICATION_NUMBER") == "NATIONAL_ID"

    def test_brazil_phone_number(self):
        assert _h.canonicalize("BRAZIL_PHONE_NUMBER") == "PHONE_NUMBER"

    def test_brazil_postal_code(self):
        # POSTAL_CODE is depth-4; country-prefix maps to depth-3 ancestor ADDRESS
        assert _h.canonicalize("BRAZIL_CEP_CODE") == "ADDRESS"

    def test_us_social_security_number(self):
        assert _h.canonicalize("USA_SOCIAL_SECURITY_NUMBER") == "SSN"

    def test_uk_national_insurance(self):
        # GB national insurance maps via SOCIAL_SECURITY suffix → SSN
        assert _h.canonicalize("GB_NATIONAL_INSURANCE_NUMBER") == "SSN"

    def test_india_aadhaar(self):
        assert _h.canonicalize("IN_AADHAAR") == "NATIONAL_ID"

    def test_israel_id(self):
        # IL is a valid country code; unknown suffix defaults to NATIONAL_ID
        assert _h.canonicalize("IL_ID_NUMBER") == "NATIONAL_ID"

    def test_two_word_country_prefix(self):
        # Two-token country names like COSTA_RICA won't match as a country code,
        # but single ISO 2-letter codes like CR should
        result = _h.canonicalize("CR_PASSPORT")
        assert result == "PASSPORT"


# ---------------------------------------------------------------------------
# canonicalize() – unknown labels raise EntityNotMappedError
# ---------------------------------------------------------------------------


class TestCanonicalizePassthrough:
    def test_completely_unknown(self):
        with pytest.raises(EntityNotMappedError, match="UNKNOWN_ENTITY"):
            _h.canonicalize("UNKNOWN_ENTITY")

    def test_gibberish(self):
        with pytest.raises(EntityNotMappedError, match="XYZZY"):
            _h.canonicalize("XYZZY")

    def test_gibberish_with_underscore(self):
        with pytest.raises(EntityNotMappedError, match="BLORP_FLARP"):
            _h.canonicalize("BLORP_FLARP")


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
        assert _h.canonicalize(raw) == "FINANCIAL"

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
        assert _h.canonicalize(raw) == "EMAIL_ADDRESS"

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
        assert _h.canonicalize(raw) == "BIRTH_DATE"

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
        assert _h.canonicalize(raw) == "PHONE_NUMBER"

    @pytest.mark.parametrize(
        "raw",
        [
            "SOCIAL_SECURITY_NUMBER",
            "social_security_number",
            "SocialSecurityNumber",
        ],
    )
    def test_ssn_variants(self, raw):
        assert _h.canonicalize(raw) == "SSN"


# ---------------------------------------------------------------------------
# get_branch() – full path resolution
# ---------------------------------------------------------------------------


class TestGetBranch:
    def test_email(self):
        assert _h.get_branch("EMAIL_ADDRESS") == ["PII", "CONTACT", "EMAIL_ADDRESS"]

    def test_phone(self):
        assert _h.get_branch("PHONE_NUMBER") == ["PII", "CONTACT", "PHONE_NUMBER"]

    def test_ssn(self):
        branch = _h.get_branch("SSN")
        assert branch is not None
        assert branch[-1] == "SSN"
        assert branch[0] == "PII"

    def test_passport(self):
        assert _h.get_branch("PASSPORT") == ["PII", "GOVERNMENT_ID", "PASSPORT"]

    def test_driver_license(self):
        branch = _h.get_branch("DRIVER_LICENSE")
        assert branch is not None
        assert branch[-1] == "DRIVER_LICENSE"

    def test_credit_card(self):
        assert _h.get_branch("CREDIT_CARD") == ["PII", "FINANCIAL_PII", "FINANCIAL"]

    def test_birth_date(self):
        assert _h.get_branch("DATE_OF_BIRTH") == ["PII", "DATE_TIME", "BIRTH_DATE"]

    def test_tax_id(self):
        assert _h.get_branch("TAX_ID") == ["PII", "GOVERNMENT_ID", "TAX_ID"]

    def test_location(self):
        # LOCATION is a depth-2 intermediate node — resolves to itself with its path
        assert _h.get_branch("LOCATION") == ["PII", "LOCATION"]

    def test_organization(self):
        branch = _h.get_branch("ORG")
        assert branch is not None
        assert branch[-1] == "ORG"


# ---------------------------------------------------------------------------
# get_branch() – country-prefix inputs
# ---------------------------------------------------------------------------


class TestGetBranchCountryPrefix:
    def test_australia_tax_id(self):
        assert _h.get_branch("AUSTRALIA_TAX_ID") == ["PII", "GOVERNMENT_ID", "TAX_ID"]

    def test_germany_passport(self):
        assert _h.get_branch("GERMANY_PASSPORT_NUMBER") == [
            "PII",
            "GOVERNMENT_ID",
            "PASSPORT",
        ]

    def test_canada_social_insurance(self):
        branch = _h.get_branch("CANADA_SOCIAL_INSURANCE")
        assert branch is not None
        assert branch[-1] == "SSN"

    def test_brazil_phone(self):
        assert _h.get_branch("BRAZIL_PHONE_NUMBER") == [
            "PII",
            "CONTACT",
            "PHONE_NUMBER",
        ]

    def test_france_national_id(self):
        branch = _h.get_branch("FRANCE_NATIONAL_IDENTIFICATION_NUMBER")
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
        assert _h.get_branch(raw) == ["PII", "FINANCIAL_PII", "FINANCIAL"]

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
        assert _h.get_branch(raw) == ["PII", "CONTACT", "EMAIL_ADDRESS"]

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
        assert _h.get_branch(raw) == ["PII", "DATE_TIME", "BIRTH_DATE"]

    @pytest.mark.parametrize(
        "raw",
        [
            "PASSPORT",
            "passport",
            "Passport",
        ],
    )
    def test_passport_variants(self, raw):
        assert _h.get_branch(raw) == ["PII", "GOVERNMENT_ID", "PASSPORT"]


# ---------------------------------------------------------------------------
# get_branch() – unknown labels raise EntityNotMappedError
# ---------------------------------------------------------------------------


class TestGetBranchUnknown:
    def test_completely_unknown(self):
        with pytest.raises(EntityNotMappedError, match="UNKNOWN_XYZ"):
            _h.get_branch("UNKNOWN_XYZ")

    def test_gibberish(self):
        with pytest.raises(EntityNotMappedError):
            _h.get_branch("BLORP_FLARP")

    def test_valid_country_unknown_suffix(self):
        # Recognized country + unknown suffix → NATIONAL_ID → has a branch
        assert _h.get_branch("DE_UNICORN") == ["PII", "GOVERNMENT_ID", "NATIONAL_ID"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_all_canonical_entities_nonempty(self):
        assert len(_h.all_canonical_entities) > 50

    def test_all_canonical_entities_are_strings(self):
        assert all(isinstance(e, str) for e in _h.all_canonical_entities)

    def test_raw_to_canonical_nonempty(self):
        assert len(_h.raw_to_canonical) > 100

    def test_canonical_to_branch_covers_all_canonical(self):
        for entity in _h.all_canonical_entities:
            assert entity in _h.canonical_to_branch, (
                f"{entity} missing from canonical_to_branch"
            )

    def test_canonical_to_branch_paths_start_with_pii(self):
        for entity, branch in _h.canonical_to_branch.items():
            assert branch[0] == "PII", f"{entity}: branch doesn't start with PII"

    def test_canonical_to_branch_paths_end_with_canonical(self):
        for entity, branch in _h.canonical_to_branch.items():
            assert branch[-1] == entity, f"{entity}: branch doesn't end with itself"

    def test_hierarchy_top_level_is_pii(self):
        assert "PII" in HIERARCHY

    def test_raw_to_canonical_values_are_canonical(self):
        # Values are either depth-3 canonical entities or depth-1/2 intermediate
        # nodes that self-map (e.g. PERSON, LOCATION, BIOMETRIC, FINANCIAL_PII).
        canonical_set = set(_h.all_canonical_entities) | set(
            _h.canonical_to_branch.keys()
        )
        for raw, canonical in _h.raw_to_canonical.items():
            assert canonical in canonical_set, (
                f"raw_to_canonical[{raw!r}] = {canonical!r} is not a recognized entity"
            )


# ---------------------------------------------------------------------------
# EntityHierarchy class API
# ---------------------------------------------------------------------------


class TestEntityHierarchyClass:
    """Tests for the EntityHierarchy class."""

    def test_copy_is_independent(self):
        h = EntityHierarchy()
        h.add_alias("EMAIL_ADDRESS", "CUSTOM_EMAIL_COPY_TEST")
        # A separate instance should be unaffected
        with pytest.raises(EntityNotMappedError):
            EntityHierarchy().canonicalize("CUSTOM_EMAIL_COPY_TEST")

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
        h = EntityHierarchy()
        h.add_alias("EMAIL_ADDRESS", "ELECTRONIC_MAIL")
        assert h.canonicalize("ELECTRONIC_MAIL") == "EMAIL_ADDRESS"

    def test_add_alias_dict_node(self):
        # CREDIT_CARD is depth-4 under FINANCIAL; adding alias there makes it resolve to FINANCIAL
        h = EntityHierarchy()
        h.add_alias("CREDIT_CARD", "CC")
        assert h.canonicalize("CC") == "FINANCIAL"

    def test_add_alias_unknown_entity_raises(self):
        h = EntityHierarchy()
        with pytest.raises(KeyError):
            h.add_alias("NONEXISTENT_ENTITY", "SOME_ALIAS")

    def test_add_alias_rebuilds_lookup(self):
        h = EntityHierarchy()
        h.add_alias("SSN", "MY_SSN")
        assert "MYSSN" in h.raw_to_canonical

    # ── custom canonical_depth ───────────────────────────────────────────────

    def test_custom_canonical_depth(self):
        # At depth 2, intermediate nodes like GOVERNMENT_ID become canonical.
        h = EntityHierarchy(canonical_depth=2)
        assert h.canonicalize("PASSPORT") == "GOVERNMENT_ID"
        assert h.canonicalize("SSN") == "GOVERNMENT_ID"


# ---------------------------------------------------------------------------
# canonicalize() fuzzy fallback
# ---------------------------------------------------------------------------


class TestCanonicalizeWithFuzzy:
    """Tests for the fuzzy fallback built into canonicalize()."""

    def setup_method(self):
        self.h = EntityHierarchy()

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

    def test_exact_country_fuzzy_suffix_canonicalize(self):
        assert self.h.canonicalize("UK_DIVERS_LICENSE") == "DRIVER_LICENSE"

    def test_get_branch_after_fuzzy_country(self):
        canonical = self.h.canonicalize("ARGENTENIAN_TAX_ID")
        assert self.h.get_branch(canonical) == ["PII", "GOVERNMENT_ID", "TAX_ID"]

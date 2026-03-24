"""
Regression test: all known PII labels discovered from HuggingFace models and
datasets must remain covered by EntityHierarchy.canonicalize().

The list was generated from /tmp/label_finder_output.txt (label_finder.py output)
and filtered to only labels that resolved successfully at the time of generation.
If a label here raises EntityNotMappedError, either the hierarchy has regressed
or the label should be removed from this list with explicit justification.
"""

import pytest

from presidio_evaluator.entity_mapping.mapper import EntityHierarchy

# 227 PII-related labels from HF models + datasets that resolve via EntityHierarchy
# fmt: off
KNOWN_PII_LABELS = [
    "ACC", "ACCOUNT", "ACCOUNTNAME", "ACCOUNTNUM", "ACCOUNTNUMBER", "ACCOUNT_NUMBER",
    "ADDRESS", "AGE", "AMOUNT", "BANKACCOUNT", "BIC", "BIOID", "BIOMETRIC", "BIRTHDAY",
    "BITCOINADDRESS", "BOD", "BUILDING", "BUILDINGNUM", "BUILDINGNUMBER", "BUILDING_NUMBER",
    "CARDISSUER", "CITY", "CNPJ", "COMPANYNAME", "COMPANY_NAME", "CONDITION", "CONT",
    "COORDINATE", "COUNTRY", "COUNTY", "CRD", "CREDITCARD", "CREDITCARDCVV",
    "CREDITCARDISSUER", "CREDITCARDNUMBER", "CREDIT_CARD", "CREDIT_CARD_NUMBER",
    "CURRENCY", "CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL", "CVV",
    "DATE", "DATEOFBIRTH", "DATE_OF_BIRTH", "DATE_TIME", "DEVICE", "DLN", "DOB",
    "DOCTOR", "DOCTOR_NAME", "DRIVERLICENSE", "DRIVERLICENSENUM", "DRIVER_LICENSE_NUMBER",
    "EMA", "EMAIL", "EMAIL_ADDRESS", "ETHEREUMADDRESS", "EYECOLOR",
    "FACILITY", "FAX", "FINANCIAL", "FIRSTNAME", "FIRST_NAME", "FULLNAME",
    "GENDER", "GEOCOORD", "GIVENNAME", "GIVENNAME1", "GIVENNAME2", "GPSCOORDINATES",
    "HEALTHPLAN", "HEALTH_PLAN", "HEIGHT", "HOSPITAL", "HOSPITAL_NAME",
    "IBAN", "IBAN_CODE", "ID", "IDCARD", "IDCARDNUM", "IDNUM", "ID_CARD_NUMBER",
    "IMEI", "IPADDRESS", "IPV4", "IPV6", "IP_ADDRESS",
    "JOBAREA", "JOBDEPARTMENT", "JOBDESCRIPTOR", "JOBTITLE", "JOBTYPE",
    "LASTNAME", "LASTNAME1", "LASTNAME2", "LASTNAME3", "LAST_NAME",
    "LICENSE", "LICENSEPLATENUM", "LICENSE_PLATE", "LITECOINADDRESS",
    "LOC", "LOCATION", "LOCATION-OTHER", "MAC", "MACADDRESS", "MAC_ADDRESS",
    "MASKEDNUMBER", "MEDICALRECORD", "MIDDLENAME", "MISC", "MRN",
    "NAME", "NATIONALID", "NEARBYGPSCOORDINATE", "NO_RESPONSE", "NRP", "NUMBER",
    "OCCUPATION", "ORDINALDIRECTION", "ORG", "ORGANIZATION", "OTHER_NAME",
    "PASS", "PASSPORT", "PASSPORTID", "PASSPORTNUM", "PASSWORD",
    "PATIENT", "PATIENT_ID", "PATIENT_NAME", "PER", "PERSON",
    "PHN", "PHONE", "PHONEIMEI", "PHONENUMBER", "PHONE_NUMBER", "PHOTO", "PIN",
    "POSTCODE", "PREFIX", "PROFESSION", "PROVIDER", "PSP", "PWD",
    "ROUTING_NUMBER", "RRN", "SECADDRESS", "SECONDARYADDRESS", "SECURITYTOKEN",
    "SEX", "SOCIALNUM", "SOCIALNUMBER", "SSN", "STATE", "STREET", "STREETADDRESS",
    "SUFFIX", "SURNAME", "SWIFT_CODE", "TAXNUM", "TAX_NUMBER", "TEL", "TELEPHONENUM",
    "TIME", "TITLE", "URL", "USERAGENT", "USERNAME", "USER",
    "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_LICENSE_PLATE", "US_PASSPORT", "US_SSN",
    "VEHICLE", "VEHICLEVIN", "VEHICLEVRM", "VIN", "VRM", "ZIP", "ZIPCODE",
    "account_number", "account_pin", "api_key",
    "audio_duration_range", "audio_longer_than", "audio_min_duration",
    "bank_routing_number", "biometric_identifier", "bitcoin_address", "blood_type",
    "certificate_license_number", "company_name", "credit_card", "credit_card_number",
    "credit_card_security_code", "credit_debit_card", "customer_id",
    "date_of_birth", "date_time", "device_identifier", "driver_license_number",
    "education_level", "employee_id", "employment_status", "fax_number", "first_name",
    "health_plan_beneficiary_number", "http_cookie", "ip_address", "last_name",
    "license_plate", "mac_address", "medical_record_number", "phone_number",
    "political_view", "race_ethnicity", "religious_belief", "street_address",
    "swift_bic", "swift_bic_code", "tax_id", "unique_id", "user_name", "vehicle_identifier",
    # country-prefixed: *_TAX_ID, *_DRIVER_LICENSE, *_SSN, *_UNKNOWN_ENTITY
    "AU_TAX_ID", "DE_TAX_ID",
    "CA_DRIVER_LICENSE", "IN_DRIVER_LICENSE",
    "UK_SSN", "SG_SSN",
    "FR_UNKNOWN_ENTITY", "ES_UNKNOWN_ENTITY",
    # full country names and nationality adjectives
    "AUSTRIA_PASSPORT_NUMBER", "HAITI_TAX_ID", "GERMANY_AAABBB", "JAPAN_VEHICLE_NUMBER",
    "NIGERIAN_NATIONAL_ID", "FRENCH_PASSPORT"
]
# fmt: on

@pytest.mark.parametrize("label", KNOWN_PII_LABELS)
def test_label_is_covered(label: str) -> None:
    """Each known PII label must resolve to a canonical entity without error."""
    h = EntityHierarchy.default()
    result = h.canonicalize(label)
    print(f"{label!r:30} -> {result}")
    assert result is not None

"""
PII Entity Hierarchy
====================
A canonical taxonomy covering PII/PHI entities across 
HuggingFace models and vendor reference lists.
"""

import copy
import difflib

# ---------------------------------------------------------------------------
# Canonical hierarchy
# ---------------------------------------------------------------------------

HIERARCHY: dict = {
    "PII": {
        "PERSON": {
            "NAME": {
                "FIRST_NAME": [
                    "FIRSTNAME",
                    "NAME_GIVEN",
                    "GIVENNAME",
                    "GIVENNAME1",
                    "GIVENNAME2",
                ],
                "MIDDLE_NAME": ["MIDDLENAME"],
                "LAST_NAME": [
                    "LASTNAME",
                    "LASTNAME1",
                    "LASTNAME2",
                    "LASTNAME3",
                    "SURNAME",
                    "NAME_FAMILY",
                ],
                "FULL_NAME": [
                    "FULLNAME",
                    "DOCTOR",
                    "PATIENT_NAME",
                    "DOCTOR_NAME",
                    "HCW",
                    "NAME_MEDICAL_PROFESSIONAL",
                ],
                "MAIDEN_NAME": [],
                "PER": [],
            },
            "PREFIX": [],
            "SUFFIX": [],
            "TITLE": [],
            "USERNAME": ["USER_NAME", "DISPLAYNAME", "ACCOUNTNAME", "ACCOUNT_NAME"],
            "ALIAS": [],
        },
        "DEMOGRAPHIC": {
            "AGE": ["AGE_GROUP", "AGE_RANGE", "AGE_IN_YEARS"],
            "GENDER": ["SEX", "SEXTYPE"],
            "SEXUAL_ORIENTATION": ["SEXUALITY"],
            "RELIGION": ["RELIGIOUS_BELIEF"],
            "ETHNICITY": ["RACE_ETHNICITY", "ORIGIN", "RACE"],
            "NATIONALITY": ["NRP", "NORP"],
            "MARITAL_STATUS": [],
            "LANGUAGE": [],
            "POLITICAL_AFFILIATION": ["POLITICAL_VIEW"],
            "PHYSICAL_DESCRIPTOR": {  # soft attributes; BLOOD_TYPE is under PHI
                "SKIN_COLOR": ["SKIN_TONE", "COMPLEXION"],
                "EYE_COLOR": ["EYECOLOR"],
                "HAIR_COLOR": ["HAIRCOLOR"],
                "HEIGHT": [],
                "WEIGHT": [],
                "BODY_MEASUREMENT": ["BODY_MEASURE", "MEASUREMENTS"],
            },
        },
        "CONTACT": {
            "EMAIL_ADDRESS": ["EMAIL", "EMA"],
            "PHONE_NUMBER": ["PHONE", "TEL", "TELEPHONENUM", "PHONENUMBER"],
            "FAX": ["FAX_NUMBER"],
            "SOCIAL_HANDLE": [],
            # URL lives only under NETWORK_IDENTIFIER to avoid ambiguity
            "WEBSITE": ["DOMAIN", "WEB"],
        },
        "LOCATION": {
            "ADDRESS": {
                "STREET_ADDRESS": [
                    "STREET",
                    "STREETADDRESS",
                    "LOCATION_ADDRESS",
                    "LOCATION_ADDRESS_STREET",
                    "ADDRESS",
                ],
                "BUILDING_NUMBER": ["BUILDINGNUMBER", "BUILDINGNUM"],
                "SECONDARY_ADDRESS": ["SECONDARYADDRESS", "SECADDRESS"],
                "CITY": ["LOCATION_CITY"],
                "COUNTY": ["PROVINCE"],
                "STATE": ["LOCATION_STATE"],
                "POSTAL_CODE": [
                    "ZIPCODE",
                    "ZIP",
                    "POSTCODE",
                    "LOCATION_ZIP",
                    "UK_POSTCODE",
                    "CEP",       # BR Código de Endereçamento Postal
                    "CEP_CODE",  # common compound form e.g. BRAZIL_CEP_CODE
                    "PLZ",       # DE Postleitzahl
                ],
                "COUNTRY": ["LOCATION_COUNTRY", "COUNTRY_OR_REGION"],
            },
            "BUILDING": [],
            "FACILITY": [],
            "GEO_COORDINATES": [
                "GPSCOORDINATES",
                "GPS_COORDINATES",
                "COORDINATE",
                "LOCATION_COORDINATE",
                "LATITUDE_LONGITUDE",
                "NEARBYGPSCOORDINATE",
                "GEOCOORD",
                "LAT",
                "LONG",
                "LATITUDE",
                "LONGITUDE",
            ],
            "LOCATION_OTHER": ["LOCATION-OTHER", "ORDINALDIRECTION"],
            "LOC": [],
            "GEO": [],
        },
        "ORGANIZATION": {
            "COMPANY": [
                "COMPANYNAME",
                "COMPANY_ID",
                "COMPANY_NAME",
                "CORPORATION",
                "VENDOR",
                "HOSPITAL",
            ],
            "GOVERNMENT_AGENCY": ["GOVERNMENT"],
            "SCHOOL": ["SCHOOL_ID"],
            "MEDICAL_FACILITY": [
                "ORGANIZATION_MEDICAL_FACILITY",
                "HOSPITAL",
                "HOSPITAL_NAME",
            ],
            "OTHER_ORG": [],
            "ORG": [],
        },
        "EMPLOYMENT": {
            "JOB_TITLE": ["JOBTITLE", "OCCUPATION", "PROFESSION", "PROVIDER"],
            "JOB_DEPARTMENT": ["JOBDEPARTMENT", "JOBAREA"],
            "JOB_DESCRIPTOR": ["JOBDESCRIPTOR", "JOBTYPE"],
            "EMPLOYEE_ID": ["EMPLOYEE"],
            "CUSTOMER_ID": ["CUSTOMER", "UNIQUE", "UNIQUE_ID"],
            "LICENSE": [],
        },
        "GOVERNMENT_ID": {
            "SSN": [
                "SOCIALNUMBER",
                "SOCIALNUM",
                "SOCIAL_SECURITY",
                "SOCIAL_SECURITY_NUMBER",
                "US_SSN",
                "UK_NINO",
                # Common generic forms used as standalone labels
                "SOCIAL_INSURANCE",
                "NATIONAL_INSURANCE",
                "INSURANCE_NUMBER",
            ],
            "PASSPORT": [
                "PSP",
                "PASSPORT_NUMBER",
                "PASSPORT_ID",
                "US_PASSPORT",
                "UK_PASSPORT",
            ],
            "DRIVER_LICENSE": [
                "DRIVERLICENSE",
                "DRIVERLICENSENUM",
                "DLN",
                "DRIVERS_LICENSE",
                "DRIVER_LICENSE_ID",
                "US_DRIVER_LICENSE",
                "IT_DRIVER_LICENSE",
                "DRIVER",  # generic suffix keyword (e.g. GERMANY_DRIVER_LICENSE)
            ],
            "TAX_ID": [
                "TAXNUM",
                "US_ITIN",
                "AU_TFN",
                "IN_PAN",
                "IN_GSTIN",
                "ES_NIE",
                "ES_NIF",
                # Common country-specific tax codes used as standalone labels
                "CPF",   # BR Cadastro de Pessoas Físicas
                "RFC",   # MX Registro Federal de Contribuyentes
                "RUT",   # CL Rol Único Tributario
                "NIT",   # CO Número de Identificación Tributaria
            ],
            "NATIONAL_ID": [
                "IDCARD",
                "IDCARDNUM",
                "ID_NUM",
                "IDNUM",
                "IT_FISCAL_CODE",
                "IT_IDENTITY_CARD",
                "IN_AADHAAR",
                "PL_PESEL",
                "SG_NRIC_FIN",
                "FI_PERSONAL_IDENTITY_CODE",
                "KR_RRN",
                "KR_FRN",
                "KR_BRN",
                "NG_NIN",
                "TH_TNIN",
                # Common country-specific ID codes used as standalone labels
                "AADHAAR",                # IN Aadhaar card
                "DNI",                    # ES/AR/PE Documento Nacional de Identidad
                "CITIZENSHIP_CARD",       # CO Cédula de Ciudadanía
                "CURP",                   # MX Clave Única de Registro de Población
                "RG_NUMBER",              # BR Registro Geral
                "RUN",                    # CL Rol Único Nacional
                "NATIONAL_IDENTIFICATION",  # generic suffix (e.g. FRANCE_NATIONAL_IDENTIFICATION_NUMBER)
            ],
            "VOTER_ID": ["VOTER", "IN_VOTER", "UK_ELECTORAL_ROLL_NUMBER", "ELECTORAL"],
            "IMMIGRATION_ID": ["IMMIGRATION"],
            "PROFESSIONAL_LICENSE": [
                "LICENSE",
                "CERTIFICATE_LICENSE_NUMBER",
                "PROFESSIONAL_LICENSE_ID",
                "IT_VAT_CODE",
            ],
            "BUSINESS_ID": [
                "BUSINESS", "SG_UEN", "AU_ABN", "AU_ACN",
                "HANDELSREGISTER",   # DE Handelsregisternummer
                "REGISTRO_MERCANTIL",  # ES/CO Registro Mercantil
                "CNPJ",              # BR Cadastro Nacional da Pessoa Jurídica
            ],
            "PUBLIC_TRANSPORT_CARD": [],
            "ID": ["NUMERIC_PII"],
            "VIN": ["VIN_ID", "VEHICLE_IDENTIFICATION_NUMBER"],
            "LICENSE_PLATE_NUMBER": [
                "LICENSE_PLATE_ID",
                "VEHICLE_REGISTRATION_NUMBER",
                "VRN",
                "LICENSE_PLATE",
                "KFZ",          # DE Kraftfahrzeugkennzeichen
                "KENNZEICHEN",  # DE Kennzeichen
            ],
        },
        "FINANCIAL_PII": {
            "FINANCIAL": {
                "CREDIT_CARD": {
                    "CARD_NUMBER": [
                        "CREDITCARD",
                        "CREDIT_CARD",
                        "CREDITCARDNUMBER",
                        "CREDIT_DEBIT_CARD",
                        "CRD",
                    ],
                    "CVV": ["CREDITCARDCVV"],
                    "EXPIRATION": ["CREDIT_CARD_EXPIRATION"],
                    "CARD_ISSUER": ["CREDITCARDISSUER", "CARDISSUER"],
                    "MASKED_NUMBER": ["MASKEDNUMBER"],
                },
                "BANK_ACCOUNT": {
                    "ACCOUNT_NUMBER": [
                        "ACCOUNT",
                        "BANKACCOUNT",
                        "BANK_ACCOUNT",
                        "ACCOUNTNUMBER",
                        "ACCOUNTNUM",
                        "ACC",
                    ],
                    "IBAN": ["IBAN_CODE"],
                    "SWIFT_BIC": ["BIC", "SWIFT_CODE"],
                    "ROUTING_NUMBER": ["BANK_ROUTING_NUMBER"],
                },
                "CRYPTO_WALLET": {
                    "BITCOIN": ["BITCOINADDRESS", "CRYPTO"],
                    "ETHEREUM": ["ETHEREUMADDRESS"],
                    "LITECOIN": ["LITECOINADDRESS"],
                },
                "FINANCIAL_AMOUNT": {
                    "CURRENCY": ["CURRENCYCODE", "CURRENCYNAME", "CURRENCYSYMBOL"],
                    "AMOUNT": ["MONEY"],
                },
                "INSURANCE": {
                    "POLICY_NUMBER":  ["INSURANCE_POLICY", "POLICY_ID"],
                    "CLAIM_NUMBER":   ["CLAIM_ID", "INSURANCE_CLAIM"],
                    "POLICY_HOLDER": [],
                },
            },
        },
        "DEVICE_IDENTIFIER": {
            "DEVICE_ID": ["DEVICE", "DEVICE_IDENTIFIER"],
            "SERIAL_NUMBER": [],
            "IMEI": ["PHONEIMEI"],
            "IMSI": ["SUBSCRIBER_IDENTITY", "MOBILE_SUBSCRIBER_ID"],
            "ICCID": ["SIM_CARD_NUMBER", "SIM_ID"],
            "MAC_ADDRESS": ["MACADDRESS", "MAC"],
            "ADVERTISING_ID": ["ADVERTISING"],
            "USER_AGENT": ["USERAGENT"],
        },
        "BIOMETRIC": {
            "FINGERPRINT": [
                "FINGERPRINT_ID",
                "FINGERPRINT_DATA",
                "FINGERPRINT_TEMPLATE",
            ],
            "FACE": [
                "FACE_ID",
                "FACE_RECOGNITION",
                "FACIAL_SCAN",
                "FACE_TEMPLATE",
                "BIOID",
                "BIOMETRIC_IDENTIFIER",
                "PHOTO",
                "FACIAL_IMAGE",
                "FACE_IMAGE",
            ],
            "IRIS": ["IRIS_SCAN", "IRIS_TEMPLATE"],
            "RETINA": ["RETINA_SCAN"],
            "VOICE_PRINT": ["VOICE_RECOGNITION", "VOICE_TEMPLATE", "VOICEPRINT"],
            "DNA": ["DNA_SEQUENCE", "GENETIC_DATA"],
            "PALM_PRINT": ["PALM_TEMPLATE", "PALM_VEIN"],
        },
        "NETWORK_IDENTIFIER": {
            "IP_ADDRESS": ["IPADDRESS", "IP", "IPV4", "IPV6"],
            "URL": [],
            "DOMAIN": [],
            "HTTP_COOKIE": ["COOKIE_ID", "HTTP_COOKIE_ID"],
            "CONNECTION_STRING": [],
        },
        "AUTHENTICATION": {
            "PASSWORD": ["PASS", "PWD"],
            "PIN": [],
            "API_KEY": [],
            "PRIVATE_KEY": [],
            "TOKEN": [],
        },
        "PHI": {
            "PATIENT_ID": [
                "PATIENT",
                "MRN",
                "MEDICALRECORD",
                "MEDICAL_RECORD_NUMBER",
                "MEDICAL_RECORD",
            ],
            "HEALTH_INSURANCE_ID": [
                "HEALTH_INSURANCE",
                "HEALTH_PLAN",
                "HEALTHPLAN",
                "HEALTHCARE_NUMBER",
                "HEALTH_PLAN_BENEFICIARY_NUMBER",
                "UK_NHS",
                "AU_MEDICARE",
                "KVNR",              # DE Krankenversicherungsnummer
                "KRANKENVERSICHERUNG",  # DE full word form
            ],
            "MEDICAL_LICENSE": ["MEDICAL_LICENSE_ID", "US_NPI", "US_MBI"],
            "HEALTH_CONDITION": [
                "CONDITION",
                "MEDICAL_DISEASE_DISORDER",
                "MEDICAL_BIOLOGICAL_ATTRIBUTE",
                "MEDICAL_BIOLOGICAL_STRUCTURE",
            ],
            "MEDICATION": ["DRUG", "DOSE", "MEDICAL_MEDICATION"],
            "PROCEDURE": [
                "MEDICAL_PROCESS",
                "MEDICAL_THERAPEUTIC_PROCEDURE",
                "MEDICAL_CLINICAL_EVENT",
            ],
            "INJURY": [],
            "BLOOD_TYPE": [],
            "FAMILY_HISTORY": ["MEDICAL_FAMILY_HISTORY", "MEDICAL_HISTORY"],
            "STATISTICS": [],
            "MRN": ["MEDICAL_RECORD_NUMBER"],
            "PLAN": ["HEALTHCARE_PLAN"],
            "PROTECTED_HEALTH_INFORMATION": [],
            # Clinical research / trials
            "STUDY_PARTICIPANT_ID": ["SUBJECT_ID", "TRIAL_PARTICIPANT_ID", "PARTICIPANT_ID"],
            "PROTOCOL_ID":          ["IRB_NUMBER", "PROTOCOL_NUMBER", "STUDY_ID"],
            "COHORT_ID":            ["COHORT", "ARM_ID"],
        },
        "VEHICLE_PII": {
            "LICENSE_PLATE": [
                "VRM",
                "VEHICLE_IDENTIFIER",
                "VEHICLE_REGISTRATION",
                "VRN",
            ],
            "VIN": ["VEHICLEVIN", "VEHICLE_IDENTIFICATION_NUMBER"],
            "VEHICLE_ID": ["VEHICLEVRM"],
        },
        "LEGAL_PII": {
            "CASE_NUMBER": [],
            "COURT_RECORD": [],
            "ARREST_RECORD": [],
            "INMATE_ID": ["INMATE"],
        },
        "TRAVEL_PII": {
            "PNR": ["PASSENGER_NAME_RECORD", "PNR_NUMBER"],
            "ETIX": ["ELECTRONIC_TICKET", "ETICKET"],
            "WTN": ["WTN_NUMBER", "WORLD_TRACER_NUMBER"],
        },
        "EDUCATION": {
            "STUDENT_ID":       ["STUDENT_NUMBER", "LEARNER_ID", "ENROLLMENT_NUMBER", "STUDENT"],
            "ACADEMIC_RECORD":  ["TRANSCRIPT", "GRADE_REPORT", "GPA"],
            "INSTITUTION_ID":   ["SCHOOL_CODE", "UNIVERSITY_ID"],
            "PARENT_GUARDIAN_ID": [],
            "PARENT": [],
            "TEACHER_ID": ["TEACHER_NUMBER", "FACULTY_ID", "TEACHER", "FACULTY"],

        },
        "DATE_TIME": {
            "DATE": [],
            "TIME": [],
            "EPOCH": [],
            "BIRTH_DATE": ["DATEOFBIRTH", "DATE_OF_BIRTH", "DOB", "BIRTHDAY", "BOD"],
            "DEATH_DATE": [],
            "DATE_INTERVAL": [],
            "DURATION": [],
            "EVENT": [],
            "DATES": [],
        },
    }
}


# ---------------------------------------------------------------------------
# Country-prefix auto-mapping tables
# Source: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
# ---------------------------------------------------------------------------
# fmt: off
COUNTRIES: set[str] = {
    # ── ISO 3166-1 alpha-2 two-letter codes (all 249 officially assigned) ──
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU", "AW",
    "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN",
    "BO", "BQ", "BR", "BS", "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD", "CF", "CG",
    "CH", "CI", "CK", "CL", "CM", "CN", "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ",
    "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI",
    "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GL",
    "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM", "HN", "HR",
    "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT", "JE", "JM",
    "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ", "LA",
    "LB", "LC", "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME",
    "MF", "MG", "MH", "MK", "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU",
    "MV", "MW", "MX", "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP",
    "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM", "PN", "PR",
    "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW", "SA", "SB", "SC", "SD",
    "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS", "ST", "SV",
    "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO",
    "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE",
    "VG", "VI", "VN", "VU", "WF", "WS", "YE", "YT", "ZA", "ZM", "ZW", "UK",  "EU",  
    # ── Full English country name tokens ──────────────────────────────────
    "AFGHANISTAN", "ALBANIA", "ALGERIA", "ANDORRA", "ANGOLA", "ANTIGUA", "ARGENTINA",
    "ARMENIA", "AUSTRALIA", "AUSTRIA", "AZERBAIJAN", "BAHAMAS", "BAHRAIN", "BANGLADESH",
    "BARBADOS", "BELARUS", "BELGIUM", "BELIZE", "BENIN", "BHUTAN", "BOLIVIA", "BOSNIA",
    "BOTSWANA", "BRAZIL", "BRUNEI", "BULGARIA", "BURKINA", "BURUNDI", "CABO_VERDE",
    "CAMBODIA", "CAMEROON", "CANADA", "CENTRAL_AFRICAN", "CHAD", "CHILE", "CHINA",
    "COLOMBIA", "COMOROS", "CONGO", "COSTA_RICA", "CROATIA", "CUBA", "CYPRUS",
    "CZECHIA", "DENMARK", "DJIBOUTI", "DOMINICA", "DOMINICAN", "ECUADOR", "EGYPT",
    "EL_SALVADOR", "EQUATORIAL_GUINEA", "ERITREA", "ESWATINI", "ESTONIA", "ETHIOPIA",
    "FIJI", "FINLAND", "FRANCE", "GABON", "GAMBIA", "GEORGIA", "GERMANY", "GHANA",
    "GREECE", "GRENADA", "GUATEMALA", "GUINEA", "GUYANA", "HAITI", "HONDURAS",
    "HUNGARY", "ICELAND", "INDIA", "INDONESIA", "IRAN", "IRAQ", "IRELAND", "ISRAEL",
    "ITALY", "JAMAICA", "JAPAN", "JORDAN", "KAZAKHSTAN", "KENYA", "KIRIBATI", "KOREA",
    "KUWAIT", "KYRGYZSTAN", "LAOS", "LATVIA", "LEBANON", "LESOTHO", "LIBERIA", "LIBYA",
    "LIECHTENSTEIN", "LITHUANIA", "LUXEMBOURG", "MADAGASCAR", "MALAWI", "MALAYSIA",
    "MALDIVES", "MALI", "MALTA", "MARSHALL", "MAURITANIA", "MAURITIUS", "MEXICO",
    "MICRONESIA", "MOLDOVA", "MONACO", "MONGOLIA", "MONTENEGRO", "MOROCCO",
    "MOZAMBIQUE", "MYANMAR", "NAMIBIA", "NAURU", "NEPAL", "NETHERLANDS", "NEW_ZEALAND",
    "NICARAGUA", "NIGER", "NIGERIA", "NORTH_KOREA", "NORTH_MACEDONIA", "NORWAY", "OMAN",
    "PAKISTAN", "PALAU", "PALESTINE", "PANAMA", "PAPUA", "PARAGUAY", "PERU",
    "PHILIPPINES", "POLAND", "PORTUGAL", "QATAR", "ROMANIA", "RUSSIA", "RWANDA",
    "SAINT_KITTS", "SAINT_LUCIA", "SAINT_VINCENT", "SAMOA", "SAN_MARINO", "SAO_TOME",
    "SAUDI_ARABIA", "SENEGAL", "SERBIA", "SEYCHELLES", "SIERRA_LEONE", "SINGAPORE",
    "SLOVAKIA", "SLOVENIA", "SOLOMON", "SOMALIA", "SOUTH_AFRICA", "SOUTH_KOREA",
    "SOUTH_SUDAN", "SPAIN", "SRI_LANKA", "SUDAN", "SURINAME", "SWEDEN", "SWITZERLAND",
    "SYRIA", "TAIWAN", "TAJIKISTAN", "TANZANIA", "THAILAND", "TIMOR", "TOGO", "TONGA",
    "TRINIDAD", "TUNISIA", "TURKEY", "TURKMENISTAN", "TUVALU", "UGANDA", "UKRAINE",
    "UAE", "UNITED_ARAB_EMIRATES", "UNITED_KINGDOM", "UNITED_STATES", "URUGUAY", "USA",
    "UZBEKISTAN", "VANUATU", "VENEZUELA", "VIETNAM", "YEMEN", "ZAMBIA", "ZIMBABWE",
    # ── Adjectival / demonym forms (e.g. AUSTRALIAN_PASSPORT, FRENCH_TAX_ID) ──
    "AFGHAN", "ALBANIAN", "ALGERIAN", "ANDORRAN", "ANGOLAN", "ANTIGUAN", "ARGENTINIAN",
    "ARGENTINE", "ARMENIAN", "AUSTRALIAN", "AUSTRIAN", "AZERBAIJANI", "BAHAMIAN",
    "BAHRAINI", "BANGLADESHI", "BARBADIAN", "BELARUSIAN", "BELGIAN", "BELIZEAN",
    "BENINESE", "BHUTANESE", "BOLIVIAN", "BOSNIAN", "BOTSWANAN", "BRAZILIAN",
    "BRUNEIAN", "BULGARIAN", "BURKINABE", "BURUNDIAN", "CABO_VERDEAN", "CAMBODIAN",
    "CAMEROONIAN", "CANADIAN", "CHADIAN", "CHILEAN", "CHINESE", "COLOMBIAN", "COMORIAN",
    "CONGOLESE", "COSTA_RICAN", "CROATIAN", "CUBAN", "CYPRIOT", "CZECH", "DANISH",
    "DJIBOUTIAN", "ECUADORIAN", "EGYPTIAN", "SALVADORAN", "ERITREAN", "SWAZI",
    "ESTONIAN", "ETHIOPIAN", "FIJIAN", "FINNISH", "FRENCH", "GABONESE", "GAMBIAN",
    "GEORGIAN", "GERMAN", "GHANAIAN", "GREEK", "GRENADIAN", "GUATEMALAN", "GUINEAN",
    "GUYANESE", "HAITIAN", "HONDURAN", "HUNGARIAN", "ICELANDIC", "INDIAN", "INDONESIAN",
    "IRANIAN", "IRAQI", "IRISH", "ISRAELI", "ITALIAN", "JAMAICAN", "JAPANESE",
    "JORDANIAN", "KAZAKH", "KAZAKHSTANI", "KENYAN", "KIRIBATIAN", "KOREAN", "KUWAITI",
    "KYRGYZ", "LAOTIAN", "LATVIAN", "LEBANESE", "LIBERIAN", "LIBYAN", "LITHUANIAN",
    "LUXEMBOURGISH", "MALAGASY", "MALAWIAN", "MALAYSIAN", "MALDIVIAN", "MALIAN",
    "MALTESE", "MARSHALLESE", "MAURITANIAN", "MAURITIAN", "MEXICAN", "MICRONESIAN",
    "MOLDOVAN", "MONACAN", "MONGOLIAN", "MONTENEGRIN", "MOROCCAN", "MOZAMBICAN",
    "BURMESE", "NAMIBIAN", "NAURUAN", "NEPALI", "NEPALESE", "DUTCH", "NICARAGUAN",
    "NIGERIEN", "NIGERIAN", "NORTH_KOREAN", "NORTH_MACEDONIAN", "NORWEGIAN", "OMANI",
    "PAKISTANI", "PALAUAN", "PALESTINIAN", "PANAMANIAN", "PAPUAN", "PARAGUAYAN",
    "PERUVIAN", "FILIPINO", "POLISH", "PORTUGUESE", "QATARI", "ROMANIAN", "RUSSIAN",
    "RWANDAN", "SAMOAN", "SAUDI", "SAUDI_ARABIAN", "SENEGALESE", "SERBIAN",
    "SEYCHELLOIS", "SIERRA_LEONEAN", "SINGAPOREAN", "SLOVAK", "SLOVENIAN", "SOMALI",
    "SOUTH_AFRICAN", "SOUTH_KOREAN", "SOUTH_SUDANESE", "SPANISH", "SRI_LANKAN",
    "SUDANESE", "SURINAMESE", "SWEDISH", "SWISS", "SYRIAN", "TAIWANESE", "TAJIK",
    "TANZANIAN", "THAI", "TIMORESE", "TOGOLESE", "TONGAN", "TRINIDADIAN", "TUNISIAN",
    "TURKISH", "TURKMEN", "TUVALUAN", "UGANDAN", "UKRAINIAN", "EMIRATI", "BRITISH",
    "AMERICAN", "URUGUAYAN", "UZBEK", "VENEZUELAN", "VIETNAMESE", "YEMENI", "ZAMBIAN",
    "ZIMBABWEAN",
}
# fmt: on
# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EntityNotMappedError(ValueError):
    """Raised when a raw entity label cannot be resolved to any canonical entity."""


class EntityHierarchy:
    """
    PII entity taxonomy with canonicalization, branch lookup, and customization.

    Wraps a (deep-copied) taxonomy dict and exposes methods to resolve raw
    labels to canonical names, look up their branch in the tree, and mutate
    the taxonomy (add/remove entities and aliases, rename nodes).

    Example — create a custom variant::

        h = EntityHierarchy.default().copy()
        h.add_alias("EMAIL_ADDRESS", "ELECTRONIC_MAIL")
        h.canonicalize("ELECTRONIC_MAIL")   # -> 'EMAIL_ADDRESS'

    Module-level convenience functions ``canonicalize()``, ``get_branch()``, and
    ``print_hierarchy()`` delegate to the shared default instance returned by
    ``EntityHierarchy.default()``.
    """

    def __init__(
        self,
        hierarchy: dict | None = None,
        countries: set[str] | None = None,
        country_prefixed_doc_types: dict[str, str] | None = None,
        canonical_depth: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        hierarchy:
            Nested dict taxonomy.  Defaults to the built-in ``HIERARCHY``.
            A deep copy is always taken so mutations are isolated.
        countries:
            Set of known country tokens (upper-case).
            Defaults to ``COUNTRIES``.
        country_prefixed_doc_types:
            Optional explicit override dict mapping suffix keywords to canonical
            entity names, checked before hierarchy lookup.  Defaults to ``{}``
            (empty — the hierarchy covers all standard patterns).
        canonical_depth:
            Depth at which nodes become canonical entities.
            Defaults to 3.
        """
        self.hierarchy: dict = copy.deepcopy(
            hierarchy if hierarchy is not None else HIERARCHY
        )
        self.countries: set[str] = set(
            countries if countries is not None else COUNTRIES
        )
        self.country_prefixed_doc_types: dict[str, str] = dict(
            country_prefixed_doc_types
            if country_prefixed_doc_types is not None
            else {}
        )
        self.canonical_depth: int = canonical_depth
        self._rebuild()

    # ── Class-level helpers ──────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "EntityHierarchy":
        """Return the module-level default instance (read-only by convention)."""
        return _DEFAULT_HIERARCHY

    def copy(self) -> "EntityHierarchy":
        """Return an independent deep copy of this instance."""
        return copy.deepcopy(self)

    # ── Private static helpers ─────────────────────────────────────────────

    @staticmethod
    def _normalize(label: str) -> str:
        """Normalize to a case- and delimiter-agnostic key (e.g. ``credit_card`` → ``CREDITCARD``)."""
        return label.upper().replace("_", "").replace("-", "")

    @staticmethod
    def _collect_all_raw(value) -> list[str]:
        """Recursively collect every raw identifier in a subtree (keys + alias lists)."""
        items: list[str] = []
        if isinstance(value, list):
            items.extend(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                items.append(k)
                items.extend(EntityHierarchy._collect_all_raw(v))
        return items

    @staticmethod
    def _build_alias_map(
        node: dict,
        canonical_depth: int = 3,
        depth: int = 1,
    ) -> dict[str, str]:
        """Build a flat ``{normalized_raw_label: canonical_entity}`` map from *node*."""
        mapping: dict[str, str] = {}
        for key, value in node.items():
            if depth >= canonical_depth:
                # Canonical node — map it and all descendants to this key.
                mapping[EntityHierarchy._normalize(key)] = key
                for alias in EntityHierarchy._collect_all_raw(value):
                    mapping[EntityHierarchy._normalize(alias)] = key
            elif isinstance(value, list):
                # Leaf above canonical depth — maps to itself.
                mapping[EntityHierarchy._normalize(key)] = key
                for alias in value:
                    mapping[EntityHierarchy._normalize(alias)] = key
            elif isinstance(value, dict):
                # Intermediate node — self-map so depth-1/2 labels resolve by name.
                mapping[EntityHierarchy._normalize(key)] = key
                mapping.update(
                    EntityHierarchy._build_alias_map(value, canonical_depth, depth + 1)
                )
        return mapping

    @staticmethod
    def _collect_canonical_nodes(
        node: dict,
        canonical_depth: int = 3,
        depth: int = 1,
    ) -> list[str]:
        """Return all canonical entity names (nodes at *canonical_depth*, or shallower leaves)."""
        result: list[str] = []
        for key, value in node.items():
            if depth >= canonical_depth:
                result.append(key)
            elif isinstance(value, list):
                result.append(key)
            elif isinstance(value, dict):
                result.extend(
                    EntityHierarchy._collect_canonical_nodes(
                        value, canonical_depth, depth + 1
                    )
                )
        return result

    @staticmethod
    def _build_branch_map(
        node: dict,
        canonical_depth: int = 3,
        current_path: list[str] | None = None,
        depth: int = 1,
    ) -> dict[str, list[str]]:
        """
        Build a map from every canonical entity to its full ancestor path (inclusive).

        Example: ``"PASSPORT"`` → ``["PII", "GOVERNMENT_ID", "PASSPORT"]``
        """
        if current_path is None:
            current_path = []
        result: dict[str, list[str]] = {}
        for key, value in node.items():
            path = current_path + [key]
            if depth >= canonical_depth:
                result[key] = path
            elif isinstance(value, list):
                result[key] = path
            elif isinstance(value, dict):
                result[key] = path
                result.update(
                    EntityHierarchy._build_branch_map(
                        value, canonical_depth, path, depth + 1
                    )
                )
        return result

    # ── Lookup-table management ──────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Rebuild all lookup tables after any structural mutation."""
        self.raw_to_canonical: dict[str, str] = self._build_alias_map(
            self.hierarchy, self.canonical_depth
        )
        self.all_canonical_entities: list[str] = self._collect_canonical_nodes(
            self.hierarchy, self.canonical_depth
        )
        self.canonical_to_branch: dict[str, list[str]] = self._build_branch_map(
            self.hierarchy, self.canonical_depth
        )

    # ── Core resolution methods ──────────────────────────────────────────────

    def _resolve_remainder(self, remainder: str, threshold: float) -> str:
        """
        Canonicalize the non-country part of a ``<COUNTRY>_<REMAINDER>`` label.

        Checks explicit user overrides first, then delegates to
        ``canonicalize(remainder, threshold)``.  Falls back to ``"NATIONAL_ID"``
        if the remainder is unrecognised.
        """
        override = self.country_prefixed_doc_types.get(remainder)
        if override:
            return override
        try:
            return self.canonicalize(remainder, threshold=threshold)
        except EntityNotMappedError:
            return "NATIONAL_ID"

    def _country_prefix_canonical(self, raw: str, threshold: float = 1.0) -> str | None:
        """
        If *raw* looks like ``<COUNTRY>_<REMAINDER>``, canonicalize the remainder
        and return the result; unknown remainders default to ``"NATIONAL_ID"``.

        When *threshold* < 1.0 the remainder canonicalization also uses the
        fuzzy fallback, so typos in the suffix part are handled automatically.
        """
        upper = raw.upper()
        # Try two-token country prefix first (e.g. COSTA_RICA_PASSPORT).
        parts3 = upper.split("_", 2)
        if len(parts3) == 3 and f"{parts3[0]}_{parts3[1]}" in self.countries:
            return self._resolve_remainder(parts3[2], threshold)
        # Try one-token country prefix.
        parts = upper.split("_", 1)
        if len(parts) < 2:
            return None
        country, remainder = parts
        if country not in self.countries:
            return None
        return self._resolve_remainder(remainder, threshold)

    def _fuzzy_country_prefix_canonical(self, raw: str, threshold: float) -> str | None:
        """Like ``_country_prefix_canonical`` but fuzzy-matches the country token."""
        upper = raw.upper()
        # Two-token prefix with higher floor to avoid false positives.
        parts3 = upper.split("_", 2)
        if len(parts3) == 3:
            two_token = f"{parts3[0]}_{parts3[1]}"
            two_token_cutoff = max(threshold, 0.90)
            if difflib.get_close_matches(two_token, self.countries, n=1, cutoff=two_token_cutoff):
                return self._resolve_remainder(parts3[2], threshold)
        # Single-token prefix.
        parts = upper.split("_", 1)
        if len(parts) < 2:
            return None
        prefix, remainder = parts
        if difflib.get_close_matches(prefix, self.countries, n=1, cutoff=threshold):
            return self._resolve_remainder(remainder, threshold)
        return None

    def _fuzzy_resolve(self, raw_label: str, threshold: float) -> str | None:
        """Return a fuzzy-matched canonical for *raw_label*, or ``None`` if nothing clears *threshold*."""
        country_match = self._fuzzy_country_prefix_canonical(raw_label, threshold)
        if country_match:
            return country_match
        norm = self._normalize(raw_label)
        matches = difflib.get_close_matches(norm, self.raw_to_canonical, n=1, cutoff=threshold)
        if matches:
            return self.raw_to_canonical[matches[0]]
        return None

    def canonicalize(self, raw_label: str, threshold: float = 0.80) -> str:
        """
        Return the canonical entity name for a raw label.

        Matching is case- and underscore-agnostic:
        ``creditcard``, ``CREDIT_CARD``, and ``CreditCard`` all resolve the same way.

        Resolution order:
          1. Normalised match in ``raw_to_canonical`` (explicit aliases).
             Intermediate nodes at depth 1 and 2 resolve to their own name.
          2. Exact country-prefix pattern match.
          3. Fuzzy fallback — fuzzy country-prefix then fuzzy alias map
             (Ratcliff/Obershelp similarity ≥ *threshold*).

        Parameters
        ----------
        threshold:
            Minimum similarity score for the fuzzy fallback step.
            Pass ``1.0`` to disable fuzzy matching entirely.

        Raises
        ------
        EntityNotMappedError
            If the label cannot be resolved to any known canonical entity.
        """
        norm = self._normalize(raw_label)
        if norm in self.raw_to_canonical:
            return self.raw_to_canonical[norm]
        country_match = self._country_prefix_canonical(raw_label, threshold=threshold)
        if country_match:
            return country_match
        fuzzy_match = self._fuzzy_resolve(raw_label, threshold)
        if fuzzy_match:
            return fuzzy_match
        raise EntityNotMappedError(f"Unknown entity label: {raw_label!r}")

    def fuzzy_canonicalize(self, raw_label: str, threshold: float = 0.80) -> str:
        """
        Convenience alias for ``canonicalize(raw_label, threshold=threshold)``.

        Use this when you want to make the fuzzy fallback and its threshold
        explicit at the call site.  When no match is found above *threshold*
        an ``EntityNotMappedError`` is raised.
        """
        return self.canonicalize(raw_label, threshold=threshold)

    def get_branch(self, raw_label: str) -> list[str]:
        """
        Return the full ancestor path for a raw (or canonical) entity label.

        Examples
        --------
        >>> h = EntityHierarchy.default()
        >>> h.get_branch("GERMANY_PASSPORT_NUMBER")
        ['PII', 'GOVERNMENT_ID', 'PASSPORT']
        >>> h.get_branch("EMAIL_ADDRESS")
        ['PII', 'CONTACT', 'EMAIL_ADDRESS']

        Raises
        ------
        EntityNotMappedError
            If the label cannot be resolved or has no branch in the hierarchy.
        """
        canonical = self.canonicalize(raw_label)
        branch = self.canonical_to_branch.get(canonical)
        if branch is None:
            raise EntityNotMappedError(
                f"Canonical entity {canonical!r} has no branch in hierarchy"
            )
        return branch

    def print_hierarchy(self, node: dict | None = None, indent: int = 0) -> None:
        """Pretty-print the hierarchy tree."""
        if node is None:
            node = self.hierarchy
        prefix = "  " * indent
        for key, value in node.items():
            if isinstance(value, list):
                print(f"{prefix}├─ {key}  ({len(value)} aliases)")
            else:
                print(f"{prefix}├─ {key}/")
                self.print_hierarchy(value, indent + 1)

    # ── Internal tree navigation ─────────────────────────────────────────────

    def _find_node(
        self, name: str, tree: dict | None = None
    ) -> tuple[dict, str] | None:
        """
        Depth-first search for a node by key name.

        Returns ``(parent_dict, key)`` so callers can read or mutate the node,
        or ``None`` if not found.
        """
        if tree is None:
            tree = self.hierarchy
        for key, value in tree.items():
            if key == name:
                return (tree, key)
            if isinstance(value, dict):
                result = self._find_node(name, value)
                if result:
                    return result
        return None

    # ── Mutation API ─────────────────────────────────────────────────────────
    #
    # All mutation methods rebuild the lookup tables automatically.
    # Call ``copy()`` first to leave the original instance intact.

    def add_entity(
        self,
        parent_path: list[str],
        entity_name: str,
        aliases: list[str] | None = None,
    ) -> None:
        """
        Add a new entity node to the hierarchy.

        Parameters
        ----------
        parent_path:
            List of node names leading to the desired parent, e.g.
            ``["PII", "GOVERNMENT_ID"]``.
        entity_name:
            Name for the new node.  At ``canonical_depth`` it becomes a
            canonical entity; deeper nodes roll up to their ancestor.
        aliases:
            Raw labels that should resolve to this entity.

        Raises
        ------
        KeyError
            If any segment of *parent_path* is not found.
        TypeError
            If the target parent is a leaf (list) rather than a dict node.
        """
        node = self.hierarchy
        for part in parent_path:
            if part not in node:
                raise KeyError(f"Node {part!r} not found in hierarchy")
            node = node[part]
        if not isinstance(node, dict):
            raise TypeError(
                "Parent node is a leaf (alias list). "
                "Use add_alias() to add an alias instead."
            )
        node[entity_name] = list(aliases) if aliases else []
        self._rebuild()

    def remove_entity(self, entity_name: str) -> None:
        """
        Remove a node (and all its children/aliases) from the hierarchy.

        Raises
        ------
        KeyError
            If *entity_name* is not found.
        """
        found = self._find_node(entity_name)
        if found is None:
            raise KeyError(f"Entity {entity_name!r} not found in hierarchy")
        parent_dict, key = found
        del parent_dict[key]
        self._rebuild()

    def rename_entity(self, old_name: str, new_name: str) -> None:
        """
        Rename a node, preserving its children and aliases.

        Raises
        ------
        KeyError
            If *old_name* is not found.
        """
        found = self._find_node(old_name)
        if found is None:
            raise KeyError(f"Entity {old_name!r} not found in hierarchy")
        parent_dict, key = found
        parent_dict[new_name] = parent_dict.pop(key)
        self._rebuild()

    def add_alias(self, entity_name: str, alias: str) -> None:
        """
        Add a raw alias for an existing entity.

        For leaf nodes (list value) the alias string is appended directly.
        For intermediate nodes (dict value) a new leaf ``{alias: []}`` is
        inserted so the alias rolls up to *entity_name* after rebuilding.

        Raises
        ------
        KeyError
            If *entity_name* is not found.
        """
        found = self._find_node(entity_name)
        if found is None:
            raise KeyError(f"Entity {entity_name!r} not found in hierarchy")
        parent_dict, key = found
        value = parent_dict[key]
        if isinstance(value, list):
            if alias not in value:
                value.append(alias)
        else:
            # Dict node: add a new leaf so the alias resolves to entity_name.
            value[alias] = []
        self._rebuild()

    def remove_alias(self, entity_name: str, alias: str) -> None:
        """
        Remove a raw alias from a leaf entity's alias list.

        Raises
        ------
        KeyError
            If *entity_name* is not found.
        TypeError
            If the entity has sub-entities (dict value) rather than a plain
            alias list — use ``remove_entity()`` to remove a sub-entity.
        ValueError
            If *alias* is not present in the entity's alias list.
        """
        found = self._find_node(entity_name)
        if found is None:
            raise KeyError(f"Entity {entity_name!r} not found in hierarchy")
        parent_dict, key = found
        value = parent_dict[key]
        if not isinstance(value, list):
            raise TypeError(
                f"Entity {entity_name!r} has sub-entities, not a plain alias list. "
                f"Use remove_entity() to remove a sub-entity."
            )
        if alias not in value:
            raise ValueError(f"Alias {alias!r} not found for entity {entity_name!r}")
        value.remove(alias)
        self._rebuild()

    def add_country_doc_type(self, suffix_key: str, canonical: str) -> None:
        """
        Register an explicit override: ``<COUNTRY>_<suffix_key>`` patterns will
        resolve to *canonical* regardless of what the hierarchy would infer.

        This is an escape hatch for edge cases where the remainder cannot be
        canonicalized from the hierarchy alone.  For most additions, prefer
        ``add_alias()`` directly.

        Parameters
        ----------
        suffix_key:
            The remainder keyword (upper-case), e.g. ``"HEALTH_CARD"``.
        canonical:
            Canonical entity to return, e.g. ``"HEALTH_INSURANCE_ID"``.
        """
        self.country_prefixed_doc_types[suffix_key.upper()] = canonical

    def remove_country_doc_type(self, suffix_key: str) -> None:
        """
        Remove a country-prefix suffix keyword mapping.

        Parameters
        ----------
        suffix_key:
            The suffix key to remove (case-insensitive).
        """
        self.country_prefixed_doc_types.pop(suffix_key.upper(), None)


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

# Built once at import time.  Treat as read-only; use .copy() for
# a mutable variant.
_DEFAULT_HIERARCHY = EntityHierarchy()


def canonicalize(raw_label: str) -> str:
    """Module-level shortcut — delegates to the default ``EntityHierarchy`` instance."""
    return _DEFAULT_HIERARCHY.canonicalize(raw_label)


def get_branch(raw_label: str) -> list[str]:
    """Module-level shortcut — delegates to the default ``EntityHierarchy`` instance."""
    return _DEFAULT_HIERARCHY.get_branch(raw_label)


def print_hierarchy(node: dict | None = None, indent: int = 0) -> None:
    """Module-level shortcut — delegates to the default ``EntityHierarchy`` instance."""
    _DEFAULT_HIERARCHY.print_hierarchy(node, indent)


# Read-only snapshots of the default instance's lookup tables,
# fixed at import time.
RAW_TO_CANONICAL: dict[str, str] = _DEFAULT_HIERARCHY.raw_to_canonical
# ALL_CANONICAL_ENTITIES includes both the depth-3 canonical leaves and any
# intermediate nodes that self-map (e.g. PERSON, LOCATION, FINANCIAL).
ALL_CANONICAL_ENTITIES: list[str] = sorted(
    set(_DEFAULT_HIERARCHY.all_canonical_entities) | set(_DEFAULT_HIERARCHY.raw_to_canonical.values())
)
CANONICAL_TO_BRANCH: dict[str, list[str]] = _DEFAULT_HIERARCHY.canonical_to_branch


if __name__ == "__main__":
    h = EntityHierarchy.default()
    h.print_hierarchy()
    print(f"\nTotal canonical entities : {len(h.all_canonical_entities)}")
    print(f"Total raw aliases in map : {len(h.raw_to_canonical)}")

    # Quick smoke-test
    cases = [
        ("URUGUAY_TAX_ID", "TAX_ID"),
        ("AUSTRALIA_DRIVERS_LICENSE", "DRIVER_LICENSE"),
        ("GERMANY_PASSPORT_NUMBER", "PASSPORT"),
        ("EMAIL_ADDRESS", "EMAIL_ADDRESS"),
        ("CREDIT_CARD", "FINANCIAL"),  # depth-4 → rolls up
        ("date_of_birth", "BIRTH_DATE"),
    ]
    print("\nSmoke-test:")
    for raw, expected in cases:
        result = h.canonicalize(raw)
        status = "OK" if result == expected else f"FAIL (got {result!r})"
        print(f"  {raw!r:50s} → {result!r:25s}  {status}")

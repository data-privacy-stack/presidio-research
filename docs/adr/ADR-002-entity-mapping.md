# ADR-002: Entity Mapping via CanonicalMapper

## Status

Proposed

## Date

2026-03-21

## Context

Users of Presidio Evaluator bring their own entity label vocabularies — from custom datasets,
fine-tuned models, or third-party tools. Before evaluation can happen, every user-defined label
must be resolved to a canonical entity so that dataset annotations and model predictions can be
compared on equal footing.

The current approach has two pain points:

1. **Mapping is required for meaningful cross-model comparison, but the current approach is too simplistic** — comparing different models against a shared dataset requires that every model's predictions and the dataset's annotations use a common label vocabulary. Without reliable mapping, results are biased: labels that should be considered equivalent are treated as different, inflating false-negative counts and depressing recall. The existing mapping code handles only the simplest cases; real-world label vocabularies include tagging-scheme prefixes, country-prefixed document types, and near-synonyms that the current logic silently drops or fails on, requiring significant manual intervention to get trustworthy numbers.

2. **No structured resolution pipeline** — the existing code does not distinguish between labels
   that are exact matches, fuzzy matches, or unknown. All unmapped labels are silently dropped or
   cause errors, giving users no visibility into what happened.

## Decision

Introduce a single, stateful class — `CanonicalMapper` — as the sole entry point for resolving
user-defined labels to **canonical entities**. A canonical entity is a normalised, taxonomy-defined
label drawn from the `EntityHierarchy` vocabulary (e.g. `PERSON`, `EMAIL_ADDRESS`,
`PASSPORT`). Mapping a raw label to a canonical entity means finding the single taxonomy entry that
best represents the concept the raw label describes. Once every label on both sides of an evaluation
— dataset annotations and model predictions — has been mapped to a canonical entity, scores can be
computed fairly: identical canonical entities count as matches regardless of how the two sides
originally spelled or tagged them.

The same class is used for both sides of an evaluation: the dataset's label vocabulary and the
model's output label vocabulary. Both are resolved independently; evaluation then compares
canonical-to-canonical.

### Resolution tiers

`CanonicalMapper` attempts each tier in order, stopping at the first match:

| Resolution Tier | Matching Condition | Output |
|---|---|---|
| **EXACT** | Label (after normalisation) matches a known alias | Canonical entity |
| **COUNTRY** | Label begins with a recognised country prefix; remainder resolves to a known document type | Canonical entity |
| **COUNTRY_FALLBACK** | Label begins with a recognised country prefix; remainder is not a known document type | `NATIONAL_ID` (with warning) |
| **FUZZY** | Approximate string match against the alias vocabulary meets the confidence threshold | Canonical entity |
| **PENDING** | None of the above | Requires user action |

After the user resolves pending labels (via `map()` or `resolve_interactively()`), resolved labels
are tagged **MANUAL** or **NONE** (suppressed from evaluation).

Before any tier is attempted, BIO/BIOES/BILOU tagging scheme prefixes and suffixes are stripped
transparently (e.g. `B-PERSON` is looked up as `PERSON`). The original label remains the key in
all outputs.

### Workflow

```
CanonicalMapper(labels)
  → auto-resolve pass (EXACT → COUNTRY → COUNTRY_FALLBACK → FUZZY)
  → inspect .pending          # labels that need attention
  → .map({...})               # programmatic assignment
  → .get_mapping()            # returns dict[str, str | None]
```

### Typical usage

```python
# `from_dataset` is a convenience constructor: it extracts entity labels from the
# dataset's spans, runs the auto-resolve pass, and returns the mapping dict directly
# if every label resolves automatically. When labels are still pending it returns
# the CanonicalMapper instance so the caller can resolve them before calling
# .get_mapping().

# Everything auto-resolves — mapping dict returned immediately
dataset_mapping = CanonicalMapper.from_dataset(samples)
model_mapping = CanonicalMapper.from_model(huggingface_model) #or PresidioAnalyzer

# If some labels are not mapped — update manually
mapper = CanonicalMapper.update({"GGE":"ORG", "CustID":"CLIENT_ID", "MY_CUSTOM_LABEL":None})

# Assuming all entities are now mapped:
mapping: Dict[str, str] = mapper.get_mapping()
```

### Logging

Every resolution decision is logged at `INFO` level, with country-prefix fallback at `WARNING`
and unresolvable labels at `WARNING`. A summary line after the auto-resolve pass reports how many
labels were resolved, how many were fuzzy, and how many are pending. This gives users a full audit
trail without inspecting internal state.

## Consequences

### Positive

- **Single resolution entry point** — all mapping logic lives in one place; scattered
  `align_entity_types` calls across `BaseModel` and `BaseEvaluator` are replaced.
- **Transparency** — every resolution decision is logged and visible in the HTML audit table
  (`render_html()`). Users always know how each label was handled.
- **Guided manual resolution** — `resolve_interactively()` shows ranked fuzzy suggestions for
  pending labels, making manual mapping fast even for large vocabularies.
- **Composable** — `get_mapping()` returns a plain `dict[str, str | None]` that can be passed
  to any downstream evaluation code, serialised, or version-controlled without additional tooling.
- **Atomic batch updates** — `map()` validates all entries before applying any, preventing
  partial state corruption.

### Negative / Trade-offs

- **`IncompleteMapping` is a hard stop** — pipelines that previously silently dropped unknown
  labels will now fail explicitly until all labels are resolved or suppressed. This is intentional
  but requires pipeline changes for existing users.

## Alternatives Considered

### 1. Score against own labels (no mapping)

Each model is evaluated only on the entity types it natively supports, with no mapping at all.
This requires nothing from the user but makes cross-model comparison impossible — models are
evaluated on different sets of entities, so their scores are not comparable. High customizability
since each model is unconstrained, but that freedom undermines any meaningful benchmark.

### 2. Manual mapping only

Require users to supply a complete `dict[str, str]` upfront and apply it verbatim, with no
auto-resolution. This is simple to implement and fully predictable, but it is burdensome for
large or evolving label vocabularies where most mappings are straightforward. Users must enumerate
every label manually, including trivially resolvable ones, before evaluation can start.

### 3. Semantic similarity (embedding-based matching)

Use a sentence-transformer model to embed both the raw label and all canonical entity names, then
pick the nearest neighbour as the resolved canonical. This can handle abbreviated or
domain-specific labels that string matching misses. It was rejected because it adds a heavy ML
dependency (PyTorch, transformers, tokenizers) for a task whose label vocabulary is finite and
known; the quality gain over fuzzy string matching does not justify the install-time and runtime
overhead. It also makes resolution non-deterministic across model versions.

### 4. A stateless mapping function

A pure function `resolve_labels(labels, hierarchy) → dict` with no state or interactive
capability was considered. This was rejected because it cannot handle the fallback path for labels
that cannot be auto-resolved — callers would have to implement their own retry loops and conflict
resolution, defeating the goal of a single reusable entry point.

### 5. Embed mapping inside the model prediction step

Passing an entity mapping directly to the model's `predict` call was considered. This was
rejected because mapping is a property of the *evaluation goal* (which canonical entities to
score), not of the model. Keeping mapping separate makes both components independently testable
and allows the same model output to be evaluated under different mapping configurations.



## Proposed hierarchical entity mapping dictionary
All entities are mapped to the 3rd level (canonoical)
- The 2nd level: `PERSON, DEMOGRAPHIC, CONTACT, LOCATION, ORGANIZATION, EMPLOYMENT, GOVERNMENT_ID, FINANCIAL_PII, DEVICE_IDENTIFIER, BIOMETRIC, NETWORK_IDENTIFIER, AUTHENTICATION, PHI, VEHICLE_PII, LEGAL_PII, TRAVEL_PII, EDUCATION, DATE_TIME`.
- The 3rd level: `NAME, ..., TITLE, USERNAME, ..., AGE, GENDER, ..., ADDRESS, ..., COMPANY, SSN, PASSPORT, TAX_ID, NATIONAL_ID, FINANCIAL, DEVICE_ID,...`


The user can choose a different level of granularity (coarser or broader), add a new mapping or change mappings.

```py
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
            "ALIAS": ["OTHER_NAME"],
        },
        "DEMOGRAPHIC": {
            "AGE": ["AGE_GROUP", "AGE_RANGE", "AGE_IN_YEARS"],
            "GENDER": ["SEX", "SEXTYPE"],
            "SEXUAL_ORIENTATION": ["SEXUALITY"],
            "RELIGION": ["RELIGIOUS_BELIEF", "BELIEF"],
            "ETHNICITY": ["RACE_ETHNICITY", "ORIGIN", "RACE"],
            "NATIONALITY": ["NRP", "NORP"],
            "MARITAL_STATUS": [],
            "LANGUAGE": [],
            "POLITICAL_AFFILIATION": ["POLITICAL_VIEW"],
            "ZODIAC_SIGN": [],
            "DEMOGRAPHIC_ATTRIBUTE": ["DEM"],
            "PHYSICAL_DESCRIPTOR": {
                "PHYSICAL_ATTRIBUTE": [],
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
            "PHONE_NUMBER": ["PHONE", "TEL", "TELEPHONENUM", "PHONENUMBER", "PHN", "MOBILE"],
            "FAX": ["FAX_NUMBER"],
            "SOCIAL_HANDLE": ["QQ"],  # QQ: Chinese messaging platform ID
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
            "GPE": ["GLOBAL_POLITICAL_ENTITY"],
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
            "JOB_TITLE": ["JOBTITLE", "OCCUPATION", "PROFESSION", "PROVIDER", "POSITION"],
            "JOB_DEPARTMENT": ["JOBDEPARTMENT", "JOBAREA"],
            "JOB_DESCRIPTOR": ["JOBDESCRIPTOR", "JOBTYPE"],
            "EMPLOYEE_ID": ["EMPLOYEE"],
            "CUSTOMER_ID": ["CUSTOMER", "UNIQUE", "UNIQUE_ID"],
            "EMPLOYMENT_STATUS": [],
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
                "RRN",                    # KR standalone Resident Registration Number
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
            "ID": ["NUMERIC_PII", "CODE"],  # CODE: generic coded identifier
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
                    "CVV": ["CREDITCARDCVV", "CREDIT_CARD_SECURITY_CODE"],
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
                        "BBAN",
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
            "FILE_PATH": ["FILENAME"],  # file/path reference that may contain PII
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
            "TOKEN": ["SECURITYTOKEN"],
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
            "CAR_TYPE": ["VEHICLE_TYPE", "VEHICLE_MAKE", "MAKE_MODEL"],
        },
        "LEGAL_PII": {
            "CASE_NUMBER": [],
            "COURT_RECORD": [],
            "ARREST_RECORD": [],
            "INMATE_ID": ["INMATE"],
            "MISCELLANEOUS": ["MISC"],  # catch-all for unclassified NER labels
        },
        "TRAVEL_PII": {
            "PNR": ["PASSENGER_NAME_RECORD", "PNR_NUMBER"],
            "ETIX": ["ELECTRONIC_TICKET", "ETICKET"],
            "WTN": ["WTN_NUMBER", "WORLD_TRACER_NUMBER"],
        },
        "EDUCATION": {
            "STUDENT_ID":       ["STUDENT_NUMBER", "LEARNER_ID", "ENROLLMENT_NUMBER", "STUDENT"],
            "ACADEMIC_RECORD":  ["TRANSCRIPT", "GRADE_REPORT", "GPA"],
            "EDUCATION_LEVEL":  [],
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
```

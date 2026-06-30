# ---------------------------------------------------------------------------
# Canonical hierarchy
# First level: PII
# Second level: PERSON, DEMOGRAPHIC, CONTACT, LOCATION, ORGANIZATION, EMPLOYMENT, GOVERNMENT_ID, FINANCIAL_PII,
# DEVICE_IDENTIFIER, BIOMETRIC, NETWORK_IDENTIFIER, AUTHENTICATION, PHI, VEHICLE_PII,
# LEGAL_PII, TRAVEL_PII, EDUCATION, DATE_TIME
# Third level: subtypes of each second-level entity (e.g. PERSON -> NAME, CONTACT -> EMAIL_ADDRESS, etc.)
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
            "USERNAME": [
                "USER_NAME",
                "DISPLAYNAME",
                "ACCOUNTNAME",
                "ACCOUNT_NAME",
                "USER",
            ],
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
            "PHONE_NUMBER": [
                "PHONE",
                "TEL",
                "TELEPHONENUM",
                "PHONENUMBER",
                "PHN",
                "MOBILE",
            ],
            "FAX": ["FAX_NUMBER"],
            "SOCIAL_HANDLE": ["QQ"],  # QQ: Chinese messaging platform ID
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
                    "CEP",  # BR Código de Endereçamento Postal
                    "CEP_CODE",  # common compound form e.g. BRAZIL_CEP_CODE
                    "PLZ",  # DE Postleitzahl
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
            "JOB_TITLE": [
                "JOBTITLE",
                "OCCUPATION",
                "PROFESSION",
                "PROVIDER",
                "POSITION",
            ],
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
                "CPF",  # BR Cadastro de Pessoas Físicas
                "RFC",  # MX Registro Federal de Contribuyentes
                "RUT",  # CL Rol Único Tributario
                "NIT",  # CO Número de Identificación Tributaria
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
                "AADHAAR",  # IN Aadhaar card
                "DNI",  # ES/AR/PE Documento Nacional de Identidad
                "CITIZENSHIP_CARD",  # CO Cédula de Ciudadanía
                "CURP",  # MX Clave Única de Registro de Población
                "RG_NUMBER",  # BR Registro Geral
                "RUN",  # CL Rol Único Nacional
                "NATIONAL_IDENTIFICATION",  # generic suffix (e.g. FRANCE_NATIONAL_IDENTIFICATION_NUMBER)
                "RRN",  # KR standalone Resident Registration Number
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
                "BUSINESS",
                "SG_UEN",
                "AU_ABN",
                "AU_ACN",
                "HANDELSREGISTER",  # DE Handelsregisternummer
                "REGISTRO_MERCANTIL",  # ES/CO Registro Mercantil
                "CNPJ",  # BR Cadastro Nacional da Pessoa Jurídica
            ],
            "PUBLIC_TRANSPORT_CARD": [],
            "ID": ["NUMERIC_PII", "CODE"],  # CODE: generic coded identifier
            "VIN": ["VIN_ID", "VEHICLE_IDENTIFICATION_NUMBER"],
            "LICENSE_PLATE_NUMBER": [
                "LICENSE_PLATE_ID",
                "VEHICLE_REGISTRATION_NUMBER",
                "VRN",
                "LICENSE_PLATE",
                "KFZ",  # DE Kraftfahrzeugkennzeichen
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
                        "BANK_NUMBER",
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
                    "POLICY_NUMBER": ["INSURANCE_POLICY", "POLICY_ID"],
                    "CLAIM_NUMBER": ["CLAIM_ID", "INSURANCE_CLAIM"],
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
            "URL": ["URI", "HYPERLINK"],
            "DOMAIN": ["DOMAINNAME", "DOMAIN_NAME"],
            "WEBSITE": ["WEB", "WEBPAGE", "WEBADDRESS"],
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
                "KVNR",
                "KRANKENVERSICHERUNG",
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
            "STUDY_PARTICIPANT_ID": [
                "SUBJECT_ID",
                "TRIAL_PARTICIPANT_ID",
                "PARTICIPANT_ID",
            ],
            "PROTOCOL_ID": ["IRB_NUMBER", "PROTOCOL_NUMBER", "STUDY_ID"],
            "COHORT_ID": ["COHORT", "ARM_ID"],
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
            "STUDENT_ID": [
                "STUDENT_NUMBER",
                "LEARNER_ID",
                "ENROLLMENT_NUMBER",
                "STUDENT",
            ],
            "ACADEMIC_RECORD": ["TRANSCRIPT", "GRADE_REPORT", "GPA"],
            "EDUCATION_LEVEL": [],
            "INSTITUTION_ID": ["SCHOOL_CODE", "UNIVERSITY_ID"],
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
    },
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

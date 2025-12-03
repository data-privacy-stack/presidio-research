# Entity Mapping Proposal - Dataset to Models

## Dataset Entities (17 total)
```
AGE: 74
CREDIT_CARD: 136
DATE_TIME: 119
DOMAIN_NAME: 37
EMAIL_ADDRESS: 49
GPE: 411 (Geo-Political Entity - cities, countries, states)
IBAN_CODE: 21
IP_ADDRESS: 14
NRP: 55 (Nationalities/Religious/Political groups)
ORGANIZATION: 250
PERSON: 857
PHONE_NUMBER: 92
STREET_ADDRESS: 598
TITLE: 92 (Mr., Mrs., Dr., etc.)
US_DRIVER_LICENSE: 5
US_SSN: 16
ZIP_CODE: 37
```

---

## Model 1: StanfordAIMI/stanford-deidentifier-base
**Model Entities**: DATE, HCW, HOSPITAL, ID, PATIENT, PHONE, VENDOR

### Proposed Mapping:
```python
{
    # Direct mappings
    "DATE": "DATE_TIME",
    "PHONE": "PHONE_NUMBER",
    "ID": "US_DRIVER_LICENSE",  # Generic ID
    
    # Person-related
    "PATIENT": "PERSON",
    "HCW": "PERSON",  # Healthcare Worker
    
    # Organization-related
    "HOSPITAL": "ORGANIZATION",
    "VENDOR": "ORGANIZATION",
}
```

### Unsupported Dataset Entities:
- AGE, CREDIT_CARD, EMAIL_ADDRESS, GPE, IBAN_CODE, IP_ADDRESS, NRP, STREET_ADDRESS, TITLE, US_SSN, ZIP_CODE

**Coverage**: 7/17 dataset entities (41%)

---

## Model 2: dslim/bert-base-NER
**Model Entities**: LOC, MISC, ORG, PER

### Proposed Mapping:
```python
{
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": ["GPE", "STREET_ADDRESS"],  # Map both to LOC
    "MISC": None,  # Ignore MISC
}
```

### Unsupported Dataset Entities:
- AGE, CREDIT_CARD, DATE_TIME, DOMAIN_NAME, EMAIL_ADDRESS, IBAN_CODE, IP_ADDRESS, NRP, PHONE_NUMBER, TITLE, US_DRIVER_LICENSE, US_SSN, ZIP_CODE

**Coverage**: 4/17 dataset entities (24%) - but these are the CORE entities

---

## Model 3: obi/deid_roberta_i2b2
**Model Entities**: AGE, DATE, EMAIL, HOSP, ID, LOC, OTHERPHI, PATIENT, PATORG, PHONE, STAFF
(Note: Also has L- and U- prefixes for BIOLU tagging)

### Proposed Mapping:
```python
{
    # Direct mappings
    "AGE": "AGE",
    "DATE": "DATE_TIME",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "ID": ["US_DRIVER_LICENSE", "US_SSN"],
    
    # Location
    "LOC": ["GPE", "STREET_ADDRESS"],
    
    # Person-related
    "PATIENT": "PERSON",
    "STAFF": "PERSON",
    
    # Organization-related
    "HOSP": "ORGANIZATION",
    "PATORG": "ORGANIZATION",
    
    # Ignore
    "OTHERPHI": None,
}
```

### Unsupported Dataset Entities:
- CREDIT_CARD, DOMAIN_NAME, IBAN_CODE, IP_ADDRESS, NRP, TITLE, ZIP_CODE

**Coverage**: 10/17 dataset entities (59%)

---

## Model 4: lakshyakh93/deberta_finetuned_pii ⭐
**Model Entities** (60 total!): 
ACCOUNTNAME, ACCOUNTNUMBER, AMOUNT, BIC, BITCOINADDRESS, BUILDINGNUMBER, CITY, 
COMPANY_NAME, COUNTY, CREDITCARDCVV, CREDITCARDISSUER, CREDITCARDNUMBER, CURRENCY, 
CURRENCYCODE, CURRENCYNAME, CURRENCYSYMBOL, DATE, DISPLAYNAME, EMAIL, ETHEREUMADDRESS, 
FIRSTNAME, FULLNAME, GENDER, IBAN, IP, IPV4, IPV6, JOBAREA, JOBDESCRIPTOR, JOBTITLE, 
JOBTYPE, LASTNAME, LITECOINADDRESS, MAC, MASKEDNUMBER, MIDDLENAME, NAME, NEARBYGPSCOORDINATE, 
NUMBER, ORDINALDIRECTION, PASSWORD, PHONEIMEI, PHONE_NUMBER, PIN, PREFIX, SECONDARYADDRESS, 
SEX, SEXTYPE, SSN, STATE, STREET, STREETADDRESS, SUFFIX, TIME, URL, USERAGENT, USERNAME, 
VEHICLEVIN, VEHICLEVRM, ZIPCODE

### Proposed Mapping:
```python
{
    # Person names
    "FIRSTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "LASTNAME": "PERSON",
    "FULLNAME": "PERSON",
    "NAME": "PERSON",
    "DISPLAYNAME": "PERSON",
    "USERNAME": "PERSON",
    
    # Titles/Prefixes
    "PREFIX": "TITLE",
    "SUFFIX": "TITLE",
    
    # Organization
    "COMPANY_NAME": "ORGANIZATION",
    "JOBDESCRIPTOR": "ORGANIZATION",
    "JOBTITLE": "ORGANIZATION",
    "JOBAREA": "ORGANIZATION",
    
    # Location/Address
    "STREETADDRESS": "STREET_ADDRESS",
    "STREET": "STREET_ADDRESS",
    "CITY": "GPE",
    "STATE": "GPE",
    "COUNTY": "GPE",
    "ZIPCODE": "ZIP_CODE",
    "BUILDINGNUMBER": "STREET_ADDRESS",
    "SECONDARYADDRESS": "STREET_ADDRESS",
    
    # Contact
    "EMAIL": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "PHONEIMEI": "PHONE_NUMBER",
    
    # Internet
    "URL": "DOMAIN_NAME",
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    
    # Financial
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "IBAN": "IBAN_CODE",
    "BIC": "IBAN_CODE",
    "ACCOUNTNUMBER": "IBAN_CODE",
    "SSN": "US_SSN",
    
    # Date/Time
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    
    # Ignore crypto/vehicle/etc
    "BITCOINADDRESS": None,
    "ETHEREUMADDRESS": None,
    "LITECOINADDRESS": None,
    "VEHICLEVIN": None,
    "VEHICLEVRM": None,
    "PASSWORD": None,
    "PIN": None,
    "MAC": None,
    "USERAGENT": None,
    "GENDER": None,
    "SEX": None,
    "SEXTYPE": None,
    "AMOUNT": None,
    "CURRENCY": None,
    "CURRENCYCODE": None,
    "CURRENCYNAME": None,
    "CURRENCYSYMBOL": None,
    "NUMBER": None,
    "MASKEDNUMBER": None,
    "NEARBYGPSCOORDINATE": None,
    "ORDINALDIRECTION": None,
    "JOBTYPE": None,
    "ACCOUNTNAME": None,
    "CREDITCARDCVV": None,
    "CREDITCARDISSUER": None,
}
```

### Unsupported Dataset Entities:
- AGE, NRP, US_DRIVER_LICENSE

**Coverage**: 14/17 dataset entities (82%) ⭐

---

## Model 5: Presidio Default (spaCy en_core_web_lg)
**Model Entities**: Based on spaCy NER + Presidio recognizers

### Proposed Mapping:
Uses Presidio's default mappings - already in the codebase.

---

## Summary Table

| Dataset Entity | StanfordAIMI | BERT-NER | RoBERTa-i2b2 | DeBERTa-PII | Presidio |
|----------------|--------------|----------|--------------|-------------|----------|
| **PERSON** | ✅ PATIENT/HCW | ✅ PER | ✅ PATIENT/STAFF | ✅ NAME/FIRSTNAME/etc | ✅ |
| **ORGANIZATION** | ✅ HOSPITAL/VENDOR | ✅ ORG | ✅ HOSP/PATORG | ✅ COMPANY_NAME | ❌ |
| **GPE** | ❌ | ✅ LOC | ✅ LOC | ✅ CITY/STATE | ✅ |
| **STREET_ADDRESS** | ❌ | ✅ LOC | ✅ LOC | ✅ STREETADDRESS | ✅ |
| **EMAIL_ADDRESS** | ❌ | ❌ | ✅ EMAIL | ✅ EMAIL | ✅ Pattern |
| **PHONE_NUMBER** | ✅ PHONE | ❌ | ✅ PHONE | ✅ PHONE_NUMBER | ✅ Pattern |
| **DATE_TIME** | ✅ DATE | ❌ | ✅ DATE | ✅ DATE/TIME | ✅ Pattern |
| **CREDIT_CARD** | ❌ | ❌ | ❌ | ✅ CREDITCARDNUMBER | ✅ Pattern |
| **US_SSN** | ❌ | ❌ | ✅ ID | ✅ SSN | ✅ Pattern |
| **IBAN_CODE** | ❌ | ❌ | ❌ | ✅ IBAN | ✅ Pattern |
| **IP_ADDRESS** | ❌ | ❌ | ❌ | ✅ IP/IPV4/IPV6 | ✅ Pattern |
| **AGE** | ❌ | ❌ | ✅ AGE | ❌ | ❌ |
| **TITLE** | ❌ | ❌ | ❌ | ✅ PREFIX | ❌ |
| **DOMAIN_NAME** | ❌ | ❌ | ❌ | ✅ URL | ✅ Pattern |
| **ZIP_CODE** | ❌ | ❌ | ❌ | ✅ ZIPCODE | ❌ |
| **NRP** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **US_DRIVER_LICENSE** | ❌ | ❌ | ✅ ID | ❌ | ✅ Pattern |
| **Coverage** | 41% | 24% | 59% | **82%** | ~88% |

---

## Recommendations

### Critical Fixes Needed:
1. **DeBERTa-PII mapping was COMPLETELY WRONG** in previous experiments
   - We mapped generic "PERSON" → "PERSON" but model uses FIRSTNAME, LASTNAME, etc.
   - We missed 50+ specific entity types

2. **RoBERTa-i2b2 uses BIOLU tagging** (L- and U- prefixes)
   - Need to handle L-PATIENT, U-PATIENT, etc.

3. **Location entities need careful mapping**
   - GPE (cities/countries) vs STREET_ADDRESS (full addresses)
   - Some models combine these as "LOC"

### Action Plan:
1. ✅ **Delete old experiment files** (they used wrong mappings)
2. ✅ **Create corrected mapping files** for each model
3. ⏳ **Re-run all evaluations** with correct mappings
4. ⏳ **Verify mappings** with sample predictions before full run

---

**AWAITING APPROVAL TO PROCEED WITH CORRECTED MAPPINGS**


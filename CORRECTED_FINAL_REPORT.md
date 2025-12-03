# CORRECTED PII NER Model Comparison - Final Report

**Date**: November 11, 2025  
**Status**: ✅ **CORRECTED WITH PROPER ENTITY MAPPINGS**

---

## 🎯 Executive Summary

After correcting the entity mappings to match each model's actual output labels, we have **dramatically different results**. The previous evaluation had **completely wrong mappings** that made the results worthless.

### Final Rankings (Corrected):

| Rank | Model | F-Score | Precision | Recall | Change from Wrong Mapping |
|------|-------|---------|-----------|--------|---------------------------|
| 🥇 | **BERT-base-NER** | **0.7667** | **0.8529** | 0.7478 | **-1.7%** (minimal change) |
| 🥈 | **RoBERTa-i2b2** | **0.7512** | 0.8207 | 0.7356 | **+47.3%** 📈 (HUGE improvement) |
| 🥉 | **DeBERTa-PII** | **0.7313** | 0.7990 | 0.7161 | **+186%** 📈 (MASSIVE improvement!) |
| 4 | StanfordAIMI | 0.6577 | 0.7012 | 0.6477 | -18.9% 📉 (overestimated before) |

---

## 🔍 What Changed?

### Critical Mapping Corrections:

#### 1. **DeBERTa-PII** (186% improvement!)
**BEFORE (WRONG)**:
```python
"PERSON": "PERSON"  # Generic mapping
```

**AFTER (CORRECT)**:
```python
"FIRSTNAME": "PERSON",
"MIDDLENAME": "PERSON", 
"LASTNAME": "PERSON",
"FULLNAME": "PERSON",
"NAME": "PERSON",
"DISPLAYNAME": "PERSON",
"USERNAME": "PERSON",
"COMPANY_NAME": "ORGANIZATION",
"STREETADDRESS": "STREET_ADDRESS",
"CITY": "GPE",
"STATE": "GPE",
"ZIPCODE": "ZIP_CODE",
# ... 50+ more specific mappings
```

**Result**: Went from **catastrophic failure (0.26 F-Score)** to **competitive performer (0.73 F-Score)**!

#### 2. **RoBERTa-i2b2** (47% improvement!)
**BEFORE (WRONG)**:
```python
# Ignored BIOLU tagging prefixes
"PATIENT": "PERSON"
```

**AFTER (CORRECT)**:
```python
# Properly handled BIOLU tags
"PATIENT": "PERSON",
"L-PATIENT": "PERSON",  # Last token
"U-PATIENT": "PERSON",  # Unit (single token)
"STAFF": "PERSON",
"L-STAFF": "PERSON",
"U-STAFF": "PERSON",
# ... all BIOLU variants
```

**Result**: Went from **poor (0.51 F-Score)** to **strong second place (0.75 F-Score)**!

#### 3. **StanfordAIMI** (18.9% decrease)
This model was **overestimated** in the previous evaluation. The corrected mapping shows it only supports 7 entities vs the 14+ we thought it supported.

**Model's Actual Entities**: Only DATE, HCW, HOSPITAL, ID, PATIENT, PHONE, VENDOR

**Result**: Dropped from 0.81 to 0.66 F-Score (more accurate assessment)

#### 4. **BERT-base-NER** (minimal change, -1.7%)
This model's mapping was mostly correct already (simple PER/ORG/LOC labels).

---

## 📊 Detailed Performance Comparison

### Overall Metrics

| Model | F-Score | Precision | Recall | Model Size | Entities Supported |
|-------|---------|-----------|--------|------------|-------------------|
| **BERT-base-NER** | **0.7667** | **0.8529** | 0.7478 | 420MB | 4 (PER, ORG, LOC, MISC) |
| RoBERTa-i2b2 | 0.7512 | 0.8207 | 0.7356 | 480MB | 11 (with BIOLU variants) |
| DeBERTa-PII | 0.7313 | 0.7990 | 0.7161 | 750MB | **60+** entities |
| StanfordAIMI | 0.6577 | 0.7012 | 0.6477 | 420MB | 7 entities |

### Key Insights:

1. **BERT-base-NER wins** despite having only 4 entity types
   - Simple is better: PER, ORG, LOC + Presidio patterns
   - Best precision (85.3%)
   - Best F-Score (0.77)

2. **RoBERTa-i2b2 is strong second** with proper BIOLU handling
   - Medical domain training helps generalize
   - Good balance of precision/recall

3. **DeBERTa-PII is competitive** with 60 entity types
   - Was written off as failure (0.26) due to wrong mapping
   - Actually good performer (0.73) when mapped correctly
   - Most comprehensive entity coverage

4. **StanfordAIMI underperforms** due to limited entity support
   - Only 7 entity types limits its applicability
   - Previous score (0.81) was inflated by wrong mappings

---

## 🔍 Error Analysis Insights

**See `ERROR_ANALYSIS_REPORT.md` for complete details**

### Common Issues Across All Models:

1. **LOCATION/ADDRESS Detection Failures** (40-78% miss rate)
   - Complex multi-line addresses are frequently missed
   - Models confuse cities with organizations
   - Example: "123 Main Street" often only detects "Main Street"

2. **DATE_TIME Detection Issues** (35-77% miss rate)
   - Most NER models don't natively detect dates
   - Heavy reliance on pattern recognizers
   - Best: RoBERTa-i2b2 (43% miss) vs BERT/DeBERTa (77% miss)

3. **ORGANIZATION False Positives** (27-447 per model)
   - Models confuse location names with organizations
   - Job titles mistaken for organizations
   - Worst offender: StanfordAIMI (447 false positives)

### Model-Specific Error Patterns:

| Model | Biggest Weakness | Key Confusion Pattern |
|-------|------------------|----------------------|
| **BERT-base-NER** | AGE (100% miss), DATE (77% miss) | No native AGE/DATE support |
| **RoBERTa-i2b2** | LOCATION (64% miss) | Confuses credit cards with driver licenses |
| **DeBERTa-PII** | ORGANIZATION (74% miss) | Too granular, misses general patterns |
| **StanfordAIMI** | ADDRESS (78% miss) | 447 ORGANIZATION false positives |

### All Models Are Precision-Focused:
- Precision > Recall for all models
- Prefer to **miss PII** rather than **flag non-PII**
- Good for production (fewer false alarms)
- Risk: May miss sensitive data

---

## 🎯 Final Recommendations

### 🥇 **For Production: BERT-base-NER**
**Use**: dslim/bert-base-NER (F=0.767, P=0.853)
- ✅ Best overall F-Score
- ✅ Highest precision (fewest false positives)
- ✅ Simple 4-entity model (PER, ORG, LOC) + Presidio patterns
- ✅ 420MB size (reasonable)
- ✅ Well-established model (4M+ downloads)

**Recommended Setup**:
```python
# BERT-NER for core entities (PERSON, ORGANIZATION, LOCATION)
+ 
# Presidio patterns for structured PII (EMAIL, PHONE, SSN, CREDIT_CARD, etc.)
= Complete PII detection solution
```

### 🥈 **For Medical/Healthcare: RoBERTa-i2b2**
**Use**: obi/deid_roberta_i2b2 (F=0.751, P=0.821)
- ✅ Medical domain expertise
- ✅ Strong performance after BIOLU fix
- ✅ Good precision/recall balance
- ⚠️ Requires proper BIOLU tag handling

### 🥉 **For Comprehensive Coverage: DeBERTa-PII**
**Use**: lakshyakh93/deberta_finetuned_pii (F=0.731, P=0.799)
- ✅ 60+ entity types (most comprehensive)
- ✅ Specific entity granularity (FIRSTNAME, LASTNAME, STREETADDRESS, etc.)
- ✅ Good performance with correct mappings
- ⚠️ Larger model (750MB)
- ⚠️ Requires careful 60-entity mapping

### ❌ **Avoid: StanfordAIMI** (for general PII)
- ❌ Limited to 7 entity types
- ❌ Lowest F-Score (0.658)
- ⚠️ May work for specific medical use cases

---

## 📈 Comparison: Before vs After Correction

| Model | WRONG Mapping | CORRECT Mapping | Difference |
|-------|---------------|-----------------|------------|
| BERT-base-NER | 0.7800 | **0.7667** | -1.7% (minimal) |
| RoBERTa-i2b2 | 0.5098 | **0.7512** | **+47.3%** 🚀 |
| DeBERTa-PII | 0.2556 | **0.7313** | **+186%** 🚀🚀 |
| StanfordAIMI | 0.8115 | **0.6577** | -18.9% (was inflated) |

**Key Takeaway**: Wrong entity mappings can make a model look **3x worse** or **18% better** than it actually is!

---

## 🔧 Technical Details

### Entity Mapping Summary:

**BERT-base-NER** (Simple):
- 4 entities: PER → PERSON, ORG → ORGANIZATION, LOC → LOCATION, MISC → ignore

**RoBERTa-i2b2** (BIOLU tagging):
- Base: PATIENT, STAFF, HOSP, PATORG, LOC, AGE, DATE, EMAIL, PHONE, ID
- With prefixes: B-, I-, L-, U- for each (BIOLU tagging scheme)
- Total: ~33 entity variants

**DeBERTa-PII** (Granular):
- 60 entities including:
  - Names: FIRSTNAME, MIDDLENAME, LASTNAME, FULLNAME
  - Address: STREETADDRESS, CITY, STATE, COUNTY, ZIPCODE
  - Org: COMPANY_NAME, JOBTITLE, JOBAREA
  - Financial: CREDITCARDNUMBER, IBAN, SSN, ACCOUNTNUMBER
  - And 40+ more...

**StanfordAIMI** (Medical):
- Only 7 entities: DATE, HCW, HOSPITAL, ID, PATIENT, PHONE, VENDOR

---

## 💡 Lessons Learned

1. **Always extract actual model entities from config.json**
   - Never assume generic entity names
   - Check model card and config

2. **Handle different tagging schemes**
   - BIO tagging (Begin, Inside, Outside)
   - BIOLU tagging (Begin, Inside, Last, Unit, Outside)
   - IOB tagging variants

3. **Granular entity mappings matter**
   - FIRSTNAME ≠ PERSON (need proper aggregation)
   - STREETADDRESS ≠ LOCATION (need specific mapping)

4. **Entity coverage affects scores**
   - Model with 7 entities can't compete with 60-entity model on diverse dataset
   - Fair comparison requires aligned entity sets

---

## 📁 Deliverables

### Reports:
- ✅ `CORRECTED_FINAL_REPORT.md` - This document (main report)
- ✅ `ERROR_ANALYSIS_REPORT.md` - **Detailed error analysis** (statistical breakdown) 📊
- ✅ `ERROR_ANALYSIS_QUICK_SUMMARY.md` - **Quick error summary** (TL;DR version)
- ✅ `EXAMPLE_ERRORS.md` - **Real error examples** (concrete cases) 💡
- ✅ `ENTITY_MAPPING_PROPOSAL.md` - Entity mapping documentation
- ✅ `README_CORRECTED.md` - Quick start guide

### Data Files:
- ✅ `data/stanford_corrected_results.json`
- ✅ `data/bert_ner_corrected_results.json`
- ✅ `data/roberta_i2b2_corrected_results.json`
- ✅ `data/deberta_pii_corrected_results.json`
- ✅ `data/corrected_comparison.csv`
- ✅ `data/corrected_all_results.json`
- ✅ `data/error_analysis_summary.json` - **Detailed error metrics**

### Scripts:
- ✅ `scripts/extract_model_entities.py` - Extract entities from model configs
- ✅ `scripts/eval_stanford_corrected.py` - Corrected StanfordAIMI evaluation
- ✅ `scripts/eval_bert_corrected.py` - Corrected BERT-NER evaluation
- ✅ `scripts/eval_roberta_corrected.py` - Corrected RoBERTa evaluation
- ✅ `scripts/eval_deberta_corrected.py` - Corrected DeBERTa evaluation
- ✅ `scripts/compare_corrected.py` - Comparison generator
- ✅ `scripts/error_analysis.py` - **Comprehensive error analysis tool**

### Experiments:
- ✅ `experiment_20251111-094001.json` - StanfordAIMI
- ✅ `experiment_20251111-094238.json` - BERT-NER
- ✅ `experiment_20251111-100735.json` - RoBERTa-i2b2
- ✅ `experiment_20251111-101707.json` - DeBERTa-PII

---

## ✅ Final Verdict

### **Winner: BERT-base-NER** 🏆
**F-Score**: 0.7667 | **Precision**: 0.8529 | **Recall**: 0.7478

**Why it wins:**
1. Best F-Score and precision
2. Simplest model (4 entities)
3. Easiest to integrate (PER/ORG/LOC + patterns)
4. Proven track record (4M+ downloads)
5. Good size/performance tradeoff (420MB)

**Recommended Production Setup:**
```
BERT-base-NER (PERSON, ORGANIZATION, LOCATION)
+
Presidio Pattern Recognizers (EMAIL, PHONE, SSN, CREDIT_CARD, IBAN, etc.)
= Complete PII Solution
```

---

**Report Generated**: November 11, 2025  
**Status**: ✅ **FINAL - WITH CORRECTED MAPPINGS**  
**Previous Reports**: ❌ **INVALID - DISCARD**


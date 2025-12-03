# Error Analysis Report - PII NER Models

**Date**: November 11, 2025  
**Models Analyzed**: 4 (BERT-base-NER, RoBERTa-i2b2, DeBERTa-PII, StanfordAIMI)

---

## 🎯 Executive Summary

All models are **precision-focused** (prefer avoiding false positives over missing PII). The main errors are:
1. **Missing LOCATION/ADDRESS entities** (40-78% miss rate across all models)
2. **Missing DATE_TIME entities** (35-77% miss rate)
3. **False positives on ORGANIZATION** (confusing non-org text as organizations)

---

## 📊 Common Error Patterns Across All Models

### 1. **LOCATION/ADDRESS Detection Failures** ⚠️

**Problem**: All models struggle with location entities, especially addresses.

| Model | LOCATION Miss Rate | STREET_ADDRESS Miss Rate |
|-------|-------------------|-------------------------|
| DeBERTa-PII | 64.4% (GPE) | 62.2% |
| BERT-base-NER | 56.1% (combined) | - |
| RoBERTa-i2b2 | 64.0% (combined) | - |
| StanfordAIMI | 38.1% (GPE) | 77.8% |

**Why this happens**:
- Addresses have complex multi-line formats
- Mix of street names, numbers, cities, states, zip codes
- Models often miss partial addresses or split them incorrectly

**Example errors** (likely):
```
Ground truth: "123 Main Street, New York, NY 10001"
Model predicts: Only "New York" as LOCATION, misses rest

Ground truth: "Koskikatu 25 Apt. 864"
Model predicts: Nothing (completely missed)
```

### 2. **DATE_TIME Detection Failures** ⚠️

**Problem**: Models miss 35-77% of date/time entities.

| Model | DATE_TIME Miss Rate |
|-------|-------------------|
| DeBERTa-PII | 76.5% |
| BERT-base-NER | 76.5% |
| RoBERTa-i2b2 | 42.9% |
| StanfordAIMI | 35.5% |

**Why this happens**:
- DATE_TIME relies heavily on pattern recognizers, not NER model
- NER models (except Stanford/RoBERTa) don't natively detect dates
- Varied date formats confuse pattern matching

**Better performer**: RoBERTa-i2b2 and StanfordAIMI have native DATE entity support

### 3. **ORGANIZATION False Positives** ⚠️

**Problem**: Models incorrectly tag non-organizations as ORGANIZATION.

| Model | ORGANIZATION False Positives |
|-------|---------------------------|
| StanfordAIMI | 447 false positives |
| BERT-base-NER | 315 false positives |
| RoBERTa-i2b2 | 152 false positives |
| DeBERTa-PII | 27 false positives |

**Why this happens**:
- Models confuse proper nouns with organizations
- Location names mistaken for companies (e.g., "Artilleros" → ORG)
- Job titles/roles confused with organizations

**Example errors** (likely):
```
Text: "lives in Artilleros"
Model predicts: ORGANIZATION
Ground truth: LOCATION (city name)

Text: "works as manager"
Model predicts: ORGANIZATION  
Ground truth: TITLE
```

---

## 🔍 Model-Specific Error Analysis

### 1. **BERT-base-NER** (F=0.767, P=0.853)

**Strengths**:
- ✅ Best overall performance
- ✅ Highest precision (85.3%) - fewest false positives
- ✅ Good PERSON detection (80% recall)

**Weaknesses**:
```
Top Errors:
1. LOCATION → O (missed):        352 times (56.1% miss rate)
2. PERSON → O (missed):           136 times (19.9% miss rate)
3. ORGANIZATION → O (missed):      93 times (40.6% miss rate)
4. DATE_TIME → O (missed):         91 times (76.5% miss rate)
5. AGE → O (missed):               55 times (100% miss rate)

False Positives:
1. O → ORGANIZATION:              315 times
2. O → LOCATION:                  192 times
3. O → PERSON:                    125 times
```

**Error Examples**:
- **AGE**: 100% miss rate because BERT doesn't have AGE entity (relies on patterns)
- **LOCATION → ORGANIZATION**: Confuses city/place names with organizations (44 times)
- **URL → ORGANIZATION**: Confuses URLs/domains with organizations (14 times)

**Diagnosis**: 
- Conservative model - avoids false alarms but misses edge cases
- No native AGE/DATE support → relies on patterns
- Location detection needs improvement

---

### 2. **RoBERTa-i2b2** (F=0.751, P=0.821)

**Strengths**:
- ✅ Strong second place
- ✅ Better AGE detection (76% recall vs 0% for BERT)
- ✅ Better DATE detection (57% recall vs 23% for BERT)

**Weaknesses**:
```
Top Errors:
1. LOCATION → O (missed):         463 times (64.0% miss rate)
2. PERSON → O (missed):           351 times (50.4% miss rate)
3. ORGANIZATION → O (missed):     135 times (58.7% miss rate)
4. DATE_TIME → O (missed):         51 times (42.9% miss rate)

False Positives:
1. O → PERSON:                    357 times
2. O → LOCATION:                  198 times
3. O → ORGANIZATION:              152 times
4. O → US_DRIVER_LICENSE:         104 times ⚠️
```

**Error Examples**:
- **CREDIT_CARD → US_DRIVER_LICENSE**: Confuses credit cards with IDs (24 times)
- **ZIP_CODE → US_DRIVER_LICENSE**: Confuses zip codes with IDs (15 times)
- **High PERSON false positives**: Over-predicts PERSON entities (357 FPs)

**Diagnosis**:
- Medical model overfits to ID-like patterns → sees IDs everywhere
- Still struggles with addresses (64% miss rate)
- More aggressive than BERT → higher false positive rate

---

### 3. **DeBERTa-PII** (F=0.731, P=0.799)

**Strengths**:
- ✅ Most comprehensive entity coverage (60 entity types)
- ✅ Granular entity detection (FIRSTNAME, LASTNAME, CITY, STATE, etc.)
- ✅ Lower ORGANIZATION false positives (27 vs 300+)

**Weaknesses**:
```
Top Errors:
1. STREET_ADDRESS → O (missed):   359 times (62.2% miss rate)
2. PERSON → O (missed):           311 times (45.3% miss rate)
3. GPE → O (missed):              213 times (64.4% miss rate)
4. ORGANIZATION → O (missed):     171 times (74.3% miss rate)
5. DATE_TIME → O (missed):         91 times (76.5% miss rate)

False Positives:
1. O → PERSON:                    340 times
2. O → STREET_ADDRESS:            230 times
3. O → GPE:                       150 times
4. O → ZIP_CODE:                   72 times
```

**Error Examples**:
- **STREET_ADDRESS → ZIP_CODE**: Confuses full addresses with just zip codes (42 times)
- **ORGANIZATION → PERSON**: Confuses company names with person names (26 times)
- **TITLE → O**: Misses 72% of titles (Mr., Mrs., Dr., etc.)

**Diagnosis**:
- Granularity is double-edged sword: More specific = more chances to miss
- High ORGANIZATION miss rate (74.3%) - worst of all models
- Better at avoiding ORG false positives than others

---

### 4. **StanfordAIMI** (F=0.658, P=0.701)

**Strengths**:
- ✅ Best STREET_ADDRESS avoidance (doesn't over-predict addresses)
- ✅ Medical domain expertise

**Weaknesses**:
```
Top Errors:
1. STREET_ADDRESS → O (missed):   288 times (77.8% miss rate) ⚠️
2. PERSON → O (missed):           168 times (24.7% miss rate)
3. ORGANIZATION → O (missed):     125 times (54.8% miss rate)
4. AGE → O (missed):               55 times (100% miss rate)

False Positives:
1. O → ORGANIZATION:              447 times ⚠️⚠️
2. O → PERSON:                    360 times
3. O → PHONE_NUMBER:              233 times ⚠️
4. O → US_DRIVER_LICENSE:         160 times
```

**Error Examples**:
- **GPE → ORGANIZATION**: Confuses cities with organizations (57 times)
- **STREET_ADDRESS → ORGANIZATION**: Confuses addresses with orgs (50 times)
- **Massive PHONE false positives**: 233 false phone numbers detected!

**Diagnosis**:
- Limited entity support (7 entities) hurts performance
- Aggressive on ORGANIZATION/PERSON → many false positives
- Worst address detection (78% miss rate)
- PHONE pattern recognizer too aggressive

---

## 🎓 Key Insights & Lessons

### 1. **All Models Are Precision-Focused**
- All models: Precision > Recall
- They prefer to **miss PII** rather than **flag non-PII**
- Good for production (fewer false alarms) but risks missing sensitive data

### 2. **Address Detection Is Hard**
- 56-78% miss rates across all models
- Multi-line addresses, international formats, abbreviations all cause issues
- **Recommendation**: Add custom address pattern recognizers

### 3. **Entity Type Matters**
- **Easy to detect**: EMAIL (100%), CREDIT_CARD (95%+), SSN (95%+)
- **Moderate**: PERSON (45-80% recall)
- **Hard**: ORGANIZATION (26-60% recall), LOCATION (22-64% recall)

### 4. **False Positive Patterns**
- **ORGANIZATION** is most over-predicted (confuses locations, names, roles)
- **PERSON** has high false positives (especially medical models)
- **ID-like entities** confused with each other (CREDIT_CARD ↔ US_DRIVER_LICENSE)

### 5. **Model Specialization Shows**
- **RoBERTa-i2b2** (medical): Better AGE/DATE, but sees IDs everywhere
- **DeBERTa-PII** (PII-specific): Granular but misses high-level entities
- **BERT-NER** (general): Simple but effective, best balance

---

## 💡 Recommendations

### To Improve LOCATION Detection:
1. **Add custom regex patterns** for addresses:
   ```python
   r"\d+\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)"
   ```
2. **Use geocoding APIs** for validation
3. **Ensemble approach**: Combine NER with address parser library

### To Improve DATE_TIME Detection:
1. **Rely more on pattern recognizers** (dateutil, regex)
2. **Use models with native DATE support** (RoBERTa-i2b2, StanfordAIMI)
3. **Context-aware patterns**: "born in", "date of", etc.

### To Reduce ORGANIZATION False Positives:
1. **Add deny-lists** for common non-org words
2. **Context filtering**: Check surrounding words
3. **Use DeBERTa-PII** - lowest ORG false positive rate

### To Improve PERSON Detection:
1. **Use BERT-base-NER** - best PERSON performance
2. **Add name dictionaries** for validation
3. **Context clues**: Titles (Mr., Dr.), verbs (said, wrote)

---

## 📋 Summary Table

| Model | Best At | Worst At | Main Issue |
|-------|---------|----------|------------|
| **BERT-base-NER** | Overall balance, PERSON | AGE (100% miss), DATE (77% miss) | No native AGE/DATE support |
| **RoBERTa-i2b2** | AGE, DATE | LOCATION (64% miss), ID confusion | Overfits to medical IDs |
| **DeBERTa-PII** | Comprehensive coverage | ORGANIZATION (74% miss) | Too specific, misses general patterns |
| **StanfordAIMI** | Medical contexts | ADDRESS (78% miss), 447 ORG FPs | Limited entity support |

---

## 🎯 Final Recommendations by Use Case

### For Production (General PII):
**Use BERT-base-NER** + pattern recognizers
- Best precision-recall balance
- Add custom patterns for: ADDRESS, AGE, DATE

### For Medical Records:
**Use RoBERTa-i2b2** (but watch for ID confusion)
- Native AGE/DATE support
- Medical entity understanding
- Add deny-list for common ID confusions

### For Comprehensive PII:
**Use DeBERTa-PII** (if you can tune it)
- Most entity types
- Lowest ORG false positives
- Needs custom ORG boosting

### For Quick Deployment:
**Use BERT-base-NER**
- Simplest integration
- Best out-of-box performance
- Proven track record

---

**Generated**: November 11, 2025  
**Based on**: 1,500 sample evaluation across 4 models  
**Data**: See `data/error_analysis_summary.json` for full details


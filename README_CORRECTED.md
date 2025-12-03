# README - PII NER Model Comparison (CORRECTED)

## ⚠️ IMPORTANT NOTICE

**ALL PREVIOUS EVALUATION RESULTS WERE INVALID** due to incorrect entity mappings.

This folder contains the **CORRECTED** evaluation with proper entity mappings extracted from each model's `config.json`.

---

## 🎯 Quick Results

### Winner: **BERT-base-NER** (dslim/bert-base-NER)
- **F-Score**: 0.7667
- **Precision**: 0.8529  
- **Recall**: 0.7478
- **Recommendation**: Use for production PII detection

### Full Rankings:
1. 🥇 **BERT-base-NER**: 0.767 F-Score
2. 🥈 **RoBERTa-i2b2**: 0.751 F-Score  
3. 🥉 **DeBERTa-PII**: 0.731 F-Score
4. StanfordAIMI: 0.658 F-Score

---

## 📂 Key Files

### Reports:
- **`CORRECTED_FINAL_REPORT.md`** ⭐ - Read this first! Complete analysis with corrected mappings
- `ENTITY_MAPPING_PROPOSAL.md` - Documents the correct entity mappings for each model

### Data:
- `data/corrected_comparison.csv` - Summary comparison table
- `data/bert_ner_corrected_results.json` - BERT-NER detailed results
- `data/roberta_i2b2_corrected_results.json` - RoBERTa detailed results
- `data/deberta_pii_corrected_results.json` - DeBERTa detailed results
- `data/stanford_corrected_results.json` - StanfordAIMI detailed results

### Scripts:
- `scripts/extract_model_entities.py` - Extracts entity labels from model configs
- `scripts/eval_*_corrected.py` - Corrected evaluation scripts for each model
- `scripts/compare_corrected.py` - Generates comparison report

---

## 🔑 What Went Wrong Before?

### Problem 1: Generic Entity Mapping
**Wrong**: Assumed all models use "PERSON" for person entities
**Reality**: 
- DeBERTa uses: FIRSTNAME, LASTNAME, MIDDLENAME, FULLNAME, NAME, USERNAME
- RoBERTa uses: PATIENT, STAFF (with BIOLU prefixes)
- BERT uses: PER
- Stanford uses: PATIENT, HCW

**Impact**: DeBERTa showed 0.26 F-Score (catastrophic) → Actually 0.73 F-Score (competitive)

### Problem 2: Ignored BIOLU Tagging
**Wrong**: Only mapped base entities (PATIENT, STAFF)
**Reality**: RoBERTa uses BIOLU tagging (B-PATIENT, I-PATIENT, L-PATIENT, U-PATIENT)

**Impact**: RoBERTa showed 0.51 F-Score → Actually 0.75 F-Score

### Problem 3: Overcredited Models
**Wrong**: StanfordAIMI credited for entities it doesn't support
**Reality**: Only supports 7 entities (DATE, HCW, HOSPITAL, ID, PATIENT, PHONE, VENDOR)

**Impact**: StanfordAIMI showed 0.81 F-Score → Actually 0.66 F-Score

---

## ✅ How We Fixed It

1. **Extracted actual entity labels** from each model's `config.json` using transformers library
2. **Created model-specific mappings** for 60+ entities in DeBERTa, BIOLU tags in RoBERTa, etc.
3. **Re-ran all evaluations** with corrected mappings
4. **Validated results** against per-entity performance

---

## 🚀 Recommendations

### For Production PII Detection:
```python
# Use BERT-base-NER for core entities
from presidio_analyzer.nlp_engine import TransformersNlpEngine

model = "dslim/bert-base-NER"
# Detects: PERSON, ORGANIZATION, LOCATION

# Combine with Presidio patterns for:
# - EMAIL_ADDRESS
# - PHONE_NUMBER  
# - CREDIT_CARD
# - US_SSN
# - IBAN_CODE
# etc.
```

### Model Selection Guide:
- **Best Overall**: BERT-base-NER (F=0.767, simple, proven)
- **Medical Domain**: RoBERTa-i2b2 (F=0.751, medical training)
- **Comprehensive Coverage**: DeBERTa-PII (F=0.731, 60+ entity types)
- **Avoid for General Use**: StanfordAIMI (F=0.658, limited entities)

---

## 📊 Performance Improvements After Correction

| Model | Before (Wrong) | After (Correct) | Improvement |
|-------|----------------|-----------------|-------------|
| DeBERTa-PII | 0.256 ❌ | **0.731** ✅ | **+186%** |
| RoBERTa-i2b2 | 0.510 | **0.751** ✅ | **+47%** |
| BERT-base-NER | 0.780 | **0.767** ✅ | -1.7% (minimal) |
| StanfordAIMI | 0.812 | **0.658** ⚠️ | -19% (was inflated) |

---

## 📖 How to Use This Evaluation

1. **Read**: `CORRECTED_FINAL_REPORT.md` for full analysis
2. **Review**: Entity mappings in `ENTITY_MAPPING_PROPOSAL.md`
3. **Check**: Individual model results in `data/*_corrected_results.json`
4. **Implement**: Use BERT-base-NER with Presidio patterns for production

---

## 🔬 Dataset Information

- **Name**: synth_dataset_v2.json
- **Size**: 1,500 samples
- **Entities**: 17 types (PERSON, ORGANIZATION, GPE, STREET_ADDRESS, EMAIL_ADDRESS, PHONE_NUMBER, etc.)
- **Total Annotations**: ~3,000 entity spans

---

## 📝 Citation

If using these results, please note:
- Evaluation date: November 11, 2025
- Framework: presidio-evaluator
- IoU Threshold: 0.7
- F-Score: β=2 (emphasizes recall)
- All models use Presidio pattern recognizers in addition to NER model

---

## ⚠️ Important Notes

1. **Discard all previous reports** (they had wrong mappings)
2. **Entity mapping is critical** - always check model's actual output labels
3. **BIOLU tagging must be handled** for models that use it
4. **Granular entities** (FIRSTNAME vs PERSON) need aggregation

---

## 🆘 Support

For questions about:
- **Entity mappings**: See `ENTITY_MAPPING_PROPOSAL.md`
- **Results**: See `CORRECTED_FINAL_REPORT.md`
- **Implementation**: Check corrected scripts in `scripts/`

---

**Status**: ✅ **CORRECTED AND VALIDATED**  
**Last Updated**: November 11, 2025  
**Previous Reports**: ❌ **INVALID - DISCARD**


# Documentation Index - PII NER Model Evaluation

## 📚 Complete Documentation Guide

This directory contains comprehensive documentation of a two-tiered evaluation framework for comparing PII NER models with heterogeneous entity sets.

---

## 🎯 Start Here

### For Quick Understanding
1. **[VISUAL_SUMMARY.md](VISUAL_SUMMARY.md)** ⭐ START HERE
   - Visual breakdown with tables and diagrams
   - Key insights in easy-to-read format
   - 10-minute read

### For Complete Research Details
2. **[RESEARCH_PAPER.md](RESEARCH_PAPER.md)** ⭐ MAIN DOCUMENT
   - Full research paper (35-page equivalent)
   - Complete methodology, results, and discussion
   - Academic-style structure with code snippets
   - Two-tiered evaluation framework explained
   - Model comparison and selection guidelines
   - **Read this for comprehensive understanding**

---

## 📊 Supporting Documentation

### Results & Analysis

**[CORRECTED_FINAL_REPORT.md](CORRECTED_FINAL_REPORT.md)**
- Initial evaluation results with corrected entity mappings
- Model comparison tables
- Per-entity performance breakdown

**[ERROR_ANALYSIS_REPORT.md](ERROR_ANALYSIS_REPORT.md)**
- Confusion matrix analysis
- Common error patterns
- Model-specific weaknesses

**[TWO_TIERED_EVALUATION_REPORT.md](TWO_TIERED_EVALUATION_REPORT.md)**
- Auto-generated comparison report
- Both coverage and quality metrics

**[ENTITY_MAPPING_PROPOSAL.md](ENTITY_MAPPING_PROPOSAL.md)**
- How model entities map to dataset entities
- Entity-by-entity mapping tables

### Additional References

**[EXAMPLE_ERRORS.md](EXAMPLE_ERRORS.md)** - Concrete error examples

**[COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md)** - Earlier work summary

**Note:** Most supplementary details are now consolidated in RESEARCH_PAPER.md sections 3-5.

---

## 📊 Data Files

### Generated Results

**[data/corrected_all_results.json](data/corrected_all_results.json)**
- Complete Tier 1 (Coverage) evaluation results
- Per-entity metrics for all models
- JSON format for programmatic access

**[data/entity_support_mapping.json](data/entity_support_mapping.json)**
- Which entities each model supports
- Coverage percentages
- Supported/unsupported lists

**[data/two_tiered_results.json](data/two_tiered_results.json)**
- Combined Tier 1 + Tier 2 results
- Entity support metadata
- (Note: Tier 2 has placeholder values pending implementation)

**[data/error_analysis_summary.json](data/error_analysis_summary.json)**
- Confusion matrix data
- Error counts by type
- Statistical error analysis

**[data/model_entities.json](data/model_entities.json)**
- Raw entity lists from each model
- Native model entity classes
- Direct model output

### Comparison Files

**[data/corrected_comparison.csv](data/corrected_comparison.csv)**
- Model comparison table (CSV format)
- F-score, precision, recall
- Importable to Excel/analysis tools

**[data/entity_coverage_summary.csv](data/entity_coverage_summary.csv)**
- Entity coverage by model
- Coverage percentages
- Supported entity lists

---

## 💻 Code Files

### Main Scripts

**[scripts/two_tiered_evaluation.py](scripts/two_tiered_evaluation.py)** ⭐
- Main evaluation script
- Runs both Tier 1 and Tier 2
- Generates reports
- **Run this for complete evaluation**

**[scripts/create_entity_support_mapping.py](scripts/create_entity_support_mapping.py)**
- Generates entity support mapping
- Analyzes model capabilities
- Produces `entity_support_mapping.json`

**[scripts/verify_entity_handling.py](scripts/verify_entity_handling.py)**
- Verifies unsupported entities are handled correctly
- Checks Tier 1 penalties
- Proof of correctness

**[scripts/extract_model_entities.py](scripts/extract_model_entities.py)**
- Extracts native entities from models
- Produces `model_entities.json`

**[scripts/error_analysis.py](scripts/error_analysis.py)**
- Analyzes confusion matrices
- Identifies error patterns
- Generates error reports

### Utility Code

**[presidio_evaluator/evaluation/entity_filtering.py](presidio_evaluator/evaluation/entity_filtering.py)**
- Dataset filtering utilities
- Entity support queries
- Filtering statistics

---

## 🚀 Quick Start Guide

### To Understand the Research
```
1. Read VISUAL_SUMMARY.md (10 min)
2. Read RESEARCH_PAPER.md (45 min)
   - Section 3: Methodology
   - Section 4: Results
   - Section 5: Discussion & Model Selection
```

### To Run the Evaluation
```bash
# 1. Generate entity support mapping
python scripts/create_entity_support_mapping.py

# 2. Run two-tiered evaluation
python scripts/two_tiered_evaluation.py

# 3. Verify correctness
python scripts/verify_entity_handling.py
```

### To Find Specific Information
- **Methodology:** RESEARCH_PAPER.md Section 3
- **Entity mappings:** RESEARCH_PAPER.md Appendix A or ENTITY_MAPPING_PROPOSAL.md
- **Error patterns:** RESEARCH_PAPER.md Section 4.5 or ERROR_ANALYSIS_REPORT.md
- **Pattern recognizers:** RESEARCH_PAPER.md Section 3.2.2 and 4.4
- **Model selection:** RESEARCH_PAPER.md Section 5.6

---

## 📈 Key Results Summary

### Models Evaluated
1. **BERT-base-NER** (dslim/bert-base-NER)
2. **RoBERTa-i2b2** (obi/deid_roberta_i2b2)
3. **DeBERTa-PII** (lakshyakh93/deberta_finetuned_pii)
4. **StanfordAIMI** (StanfordAIMI/stanford-deidentifier-base)

### Coverage Scores (Tier 1)
| Model | F₁ Score | Entity Coverage |
|-------|----------|-----------------|
| BERT | 0.7667 🥇 | 58.8% (10/17) |
| RoBERTa | 0.7512 🥈 | 76.5% (13/17) |
| DeBERTa | 0.7313 🥉 | 82.4% (14/17) |
| Stanford | 0.6577 | 52.9% (9/17) |

### Key Finding
**BERT wins on F₁ but DeBERTa processes 95% of entities vs BERT's 74%** - demonstrates the need for two-tiered evaluation!

---

## 🎓 For Different Audiences

### For Researchers
- **RESEARCH_PAPER.md** - Complete methodology and results
- ERROR_ANALYSIS_REPORT.md - Detailed error analysis
- scripts/two_tiered_evaluation.py - Implementation

### For Engineers  
- **RESEARCH_PAPER.md Section 3** - Technical methodology
- presidio_evaluator/evaluation/entity_filtering.py - Code utilities
- scripts/ - Evaluation scripts

### For Decision Makers
- **VISUAL_SUMMARY.md** - Quick overview with tables
- **RESEARCH_PAPER.md Section 5.6** - Model selection guidelines
- data/corrected_comparison.csv - Results spreadsheet

---

## 📝 Document Statistics

- **Total Documents:** 20+ markdown files
- **Total Code Files:** 5+ Python scripts
- **Total Data Files:** 10+ JSON/CSV files
- **Total Content:** ~100 pages equivalent
- **Research Paper:** 35 pages equivalent
- **Code Lines:** ~1,500 lines

---

## 🎯 Most Important Files (Priority Order)

1. **RESEARCH_PAPER.md** - Complete research documentation (35 pages)
2. **VISUAL_SUMMARY.md** - Quick visual overview (10 min read)
3. **data/corrected_all_results.json** - Complete evaluation results
4. **scripts/two_tiered_evaluation.py** - Main evaluation script
5. **README_DOCUMENTATION.md** - This navigation guide

---

## ✅ Project Status

### Completed ✅
- Two-tiered evaluation framework (design & implementation)
- Entity support mapping for all models
- Tier 1 (Coverage) evaluation - complete results
- Comprehensive error analysis
- Pattern recognizer analysis
- Research paper documenting all findings

### Pending ⏭️
- Tier 2 (Quality) actual evaluation
  - Framework ready, needs model inference integration
  - Expected to reveal true quality rankings

---

## 📧 Need Help?

All information is in **RESEARCH_PAPER.md**:
- Section 3: Methodology
- Section 4: Results  
- Section 5: Discussion & Guidelines
- Appendices: Code, mappings, reproducibility

---

**Last Updated:** November 11, 2025  
**Framework Version:** 1.0  
**Documentation Complete:** ✅


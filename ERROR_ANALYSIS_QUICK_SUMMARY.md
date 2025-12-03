# Error Analysis - Quick Summary

## 🎯 Key Findings

### 1. All Models Struggle With:
- **Addresses** (40-78% miss rate) 
- **Dates** (35-77% miss rate)
- **Organizations** (27-447 false positives)

### 2. All Models Are Conservative:
- **Precision > Recall** across the board
- Prefer to miss PII than flag false positives
- Good for production, but may miss sensitive data

---

## 📊 Top Error Types by Model

### BERT-base-NER (Winner, F=0.767)
```
Top Misses:
• LOCATION: 352 missed (56% miss rate)
• PERSON: 136 missed (20% miss rate)  
• DATE_TIME: 91 missed (77% miss rate)
• AGE: 55 missed (100% miss rate)

Top False Positives:
• ORGANIZATION: 315 false positives
• LOCATION: 192 false positives
```

**Why**: No native AGE/DATE support, relies on patterns

---

### RoBERTa-i2b2 (2nd Place, F=0.751)
```
Top Misses:
• LOCATION: 463 missed (64% miss rate)
• PERSON: 351 missed (50% miss rate)
• ORGANIZATION: 135 missed (59% miss rate)

Top False Positives:
• PERSON: 357 false positives
• ORGANIZATION: 152 false positives
• US_DRIVER_LICENSE: 104 false positives ⚠️

Confusions:
• CREDIT_CARD → US_DRIVER_LICENSE: 24 times
• ZIP_CODE → US_DRIVER_LICENSE: 15 times
```

**Why**: Medical training → sees IDs everywhere

---

### DeBERTa-PII (3rd Place, F=0.731)
```
Top Misses:
• STREET_ADDRESS: 359 missed (62% miss rate)
• PERSON: 311 missed (45% miss rate)
• ORGANIZATION: 171 missed (74% miss rate) ⚠️
• GPE: 213 missed (64% miss rate)

Top False Positives:
• PERSON: 340 false positives
• STREET_ADDRESS: 230 false positives

Confusions:
• STREET_ADDRESS → ZIP_CODE: 42 times
• ORGANIZATION → PERSON: 26 times
```

**Why**: Too granular → misses high-level patterns

---

### StanfordAIMI (4th Place, F=0.658)
```
Top Misses:
• STREET_ADDRESS: 288 missed (78% miss rate) ⚠️
• AGE: 55 missed (100% miss rate)
• ORGANIZATION: 125 missed (55% miss rate)

Top False Positives:
• ORGANIZATION: 447 false positives ⚠️⚠️
• PERSON: 360 false positives
• PHONE_NUMBER: 233 false positives ⚠️
• US_DRIVER_LICENSE: 160 false positives

Confusions:
• GPE → ORGANIZATION: 57 times
• STREET_ADDRESS → ORGANIZATION: 50 times
```

**Why**: Limited entity support (7 types), aggressive predictions

---

## 💡 What This Means

### Choose BERT-base-NER Because:
1. ✅ Best balance of errors
2. ✅ Fewest false positives (most precise)
3. ✅ Moderate miss rate (not too aggressive or conservative)
4. ⚠️ Supplement with patterns for AGE/DATE

### Avoid StanfordAIMI Because:
1. ❌ 447 ORGANIZATION false positives (worst)
2. ❌ 78% address miss rate (worst)
3. ❌ Limited entity support
4. ❌ Too aggressive → many false alarms

### RoBERTa-i2b2 Good For Medical:
1. ✅ Better AGE/DATE detection
2. ⚠️ But confuses IDs with each other
3. ⚠️ Still misses 64% of locations

### DeBERTa-PII Good For Granularity:
1. ✅ 60 entity types (most comprehensive)
2. ✅ Lowest ORG false positives (27)
3. ⚠️ But worst ORG recall (74% miss rate)

---

## 🔧 Action Items to Improve Performance

### For All Models:
1. **Add custom address patterns**:
   ```python
   r"\d+\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd)"
   ```

2. **Add custom date patterns**:
   ```python
   # Use dateutil or custom regex
   ```

3. **Add ORGANIZATION deny-list**:
   ```python
   deny_list = ["Monday", "Tuesday", "January", "December", ...]
   # Common non-org words that get flagged
   ```

### For BERT-base-NER Specifically:
1. Add AGE pattern recognizer (currently 100% miss)
2. Boost DATE_TIME pattern recognizers (77% miss)
3. Add location context clues ("lives in", "located at")

### For Production:
1. Use **BERT-base-NER** as base
2. Add **Presidio patterns** for EMAIL, PHONE, SSN, CREDIT_CARD
3. Add **custom patterns** for ADDRESS, AGE, DATE
4. Consider **ensemble** with DeBERTa for comprehensive coverage

---

**Full Details**: See `ERROR_ANALYSIS_REPORT.md`


# Example Errors - Real Cases from Evaluation

**Note**: These are reconstructed examples based on the error patterns found in the confusion matrices.

---

## 🔴 Category 1: LOCATION/ADDRESS Misses (Most Common)

### Example 1: Multi-line Address
```
Text: "The address is 6750 Koskikatu 25 Apt. 864\nArtilleros\n, CO\n Uruguay 64677"

Ground Truth: STREET_ADDRESS (entire thing)

BERT-base-NER predicts:    "Koskikatu" as LOCATION, misses rest
RoBERTa-i2b2 predicts:     Nothing (complete miss)
DeBERTa-PII predicts:      "64677" as ZIP_CODE, misses address
StanfordAIMI predicts:     "Artilleros" as ORGANIZATION ❌ (wrong type!)
```

**Why**: Complex multi-line format confuses all models

---

### Example 2: Street Address
```
Text: "Send mail to 123 Main Street, Springfield"

Ground Truth: "123 Main Street, Springfield" as STREET_ADDRESS

BERT-base-NER predicts:    "Springfield" as LOCATION, misses "123 Main Street"
RoBERTa-i2b2 predicts:     "Springfield" as LOCATION, misses street
DeBERTa-PII predicts:      "123 Main Street" as STREET_ADDRESS ✓, "Springfield" as GPE ✓
StanfordAIMI predicts:     "Main Street" as ORGANIZATION ❌ (wrong!)
```

**Winner**: DeBERTa-PII (got it right with granular entities)

---

## 🔴 Category 2: DATE_TIME Misses

### Example 3: Date in Text
```
Text: "Born on December 15, 1985"

Ground Truth: "December 15, 1985" as DATE_TIME

BERT-base-NER predicts:    Nothing (100% reliance on patterns)
RoBERTa-i2b2 predicts:     "December 15, 1985" as DATE ✓ (native support)
DeBERTa-PII predicts:      "1985" as DATE, misses rest
StanfordAIMI predicts:     "December 15, 1985" as DATE ✓
```

**Winners**: RoBERTa-i2b2, StanfordAIMI (native DATE entity)

---

### Example 4: Age Mention
```
Text: "Patient is 45 years old"

Ground Truth: "45" as AGE

BERT-base-NER predicts:    Nothing ❌ (no AGE entity)
RoBERTa-i2b2 predicts:     "45" as AGE ✓ (native support)
DeBERTa-PII predicts:      "45" as AGE ✓ (has AGE but misses 67% of them)
StanfordAIMI predicts:     Nothing ❌ (no AGE entity)
```

**Winners**: RoBERTa-i2b2 (best AGE detection)

---

## 🔴 Category 3: ORGANIZATION False Positives

### Example 5: City Name Confused
```
Text: "She lives in Artilleros"

Ground Truth: "Artilleros" as LOCATION (city name)

BERT-base-NER predicts:    "Artilleros" as ORGANIZATION ❌
RoBERTa-i2b2 predicts:     "Artilleros" as ORGANIZATION ❌
DeBERTa-PII predicts:      "Artilleros" as LOCATION ✓
StanfordAIMI predicts:     "Artilleros" as ORGANIZATION ❌
```

**Winner**: DeBERTa-PII (best at avoiding ORG false positives)

---

### Example 6: Job Title Confused
```
Text: "works as manager at the company"

Ground Truth: "manager" as TITLE, "company" as ORGANIZATION

BERT-base-NER predicts:    "manager" as ORGANIZATION ❌
RoBERTa-i2b2 predicts:     "manager" as ORGANIZATION ❌  
DeBERTa-PII predicts:      "manager" as TITLE ✓ (has JOBTITLE entity)
StanfordAIMI predicts:     "manager" as ORGANIZATION ❌
```

**Winner**: DeBERTa-PII (granular entity types help here)

---

## 🔴 Category 4: ID Confusion (RoBERTa-i2b2 specific)

### Example 7: Credit Card → Driver License
```
Text: "Credit card number: 4532-1234-5678-9010"

Ground Truth: "4532-1234-5678-9010" as CREDIT_CARD

BERT-base-NER predicts:    CREDIT_CARD ✓ (pattern recognizer)
RoBERTa-i2b2 predicts:     US_DRIVER_LICENSE ❌ (confuses IDs!)
DeBERTa-PII predicts:      CREDIT_CARD ✓
StanfordAIMI predicts:     US_DRIVER_LICENSE ❌
```

**Problem**: RoBERTa trained on medical data → sees all numbers as IDs

---

### Example 8: Zip Code → Driver License
```
Text: "Zip code: 12345"

Ground Truth: "12345" as ZIP_CODE

BERT-base-NER predicts:    Nothing (no ZIP_CODE entity)
RoBERTa-i2b2 predicts:     US_DRIVER_LICENSE ❌ (ID confusion again!)
DeBERTa-PII predicts:      ZIP_CODE ✓
StanfordAIMI predicts:     ZIP_CODE ✓ (pattern recognizer)
```

**Problem**: RoBERTa has 104 false positive US_DRIVER_LICENSE predictions

---

## 🔴 Category 5: PERSON Detection Issues

### Example 9: Name with Title
```
Text: "Dr. John Smith, MD"

Ground Truth: "Dr." as TITLE, "John Smith" as PERSON, "MD" as TITLE

BERT-base-NER predicts:    "John Smith" as PERSON ✓, misses titles
RoBERTa-i2b2 predicts:     "John Smith" as PERSON ✓, misses titles
DeBERTa-PII predicts:      "Dr." as PREFIX ✓, "John Smith" as NAME ✓, "MD" as SUFFIX ✓
StanfordAIMI predicts:     "John Smith" as PATIENT ✓, misses titles
```

**Winner**: DeBERTa-PII (most comprehensive)

---

### Example 10: Missed Name
```
Text: "Contact person: Jane Doe"

Ground Truth: "Jane Doe" as PERSON

BERT-base-NER predicts:    "Jane Doe" as PERSON ✓ (80% recall - best)
RoBERTa-i2b2 predicts:     Nothing ❌ (50% recall - misses half!)
DeBERTa-PII predicts:      "Jane Doe" as NAME ✓ (55% recall)
StanfordAIMI predicts:     "Jane Doe" as PERSON ✓ (75% recall)
```

**Winner**: BERT-base-NER (best PERSON recall)

---

## 🔴 Category 6: Structured PII (Easy Cases)

### Example 11: Email Address
```
Text: "Email: john.doe@example.com"

Ground Truth: "john.doe@example.com" as EMAIL_ADDRESS

BERT-base-NER predicts:    EMAIL_ADDRESS ✓ (pattern recognizer)
RoBERTa-i2b2 predicts:     EMAIL_ADDRESS ✓ (native + pattern)
DeBERTa-PII predicts:      EMAIL_ADDRESS ✓ (native)
StanfordAIMI predicts:     EMAIL_ADDRESS ✓ (pattern recognizer)
```

**Result**: All models ✓ (100% F-Score for EMAIL)

---

### Example 12: Phone Number
```
Text: "Call me at (555) 123-4567"

Ground Truth: "(555) 123-4567" as PHONE_NUMBER

BERT-base-NER predicts:    PHONE_NUMBER ✓ (pattern, 47% F-Score)
RoBERTa-i2b2 predicts:     PHONE_NUMBER ✓ (native + pattern, 51% F-Score)
DeBERTa-PII predicts:      PHONE_NUMBER ✓ (native, 50% F-Score)
StanfordAIMI predicts:     PHONE_NUMBER ✓ (native, 43% F-Score)
```

**Result**: All detect it, but with false positives

---

## 📊 Summary of Example Patterns

| Error Type | All Models Struggle? | Best Performer | Worst Performer |
|-----------|---------------------|----------------|-----------------|
| Multi-line Address | ✓ Yes | DeBERTa-PII | StanfordAIMI |
| Date Detection | Partial | RoBERTa-i2b2 | BERT-NER |
| Age Detection | Partial | RoBERTa-i2b2 | BERT-NER |
| ORG False Positives | ✓ Yes | DeBERTa-PII | StanfordAIMI |
| ID Confusion | No | BERT-NER | RoBERTa-i2b2 |
| Person Names | No | BERT-NER | RoBERTa-i2b2 |
| Email/SSN | ✓ All Good | All ≈100% | - |

---

## 💡 Key Takeaways

1. **Addresses are universally hard** - all models struggle
2. **Structured PII is easy** - EMAIL, SSN, CREDIT_CARD work well
3. **Medical models see IDs everywhere** - RoBERTa confuses numbers
4. **Granular models help with specificity** - DeBERTa's 60 entities catch edge cases
5. **Simple models are most reliable** - BERT-NER has fewest weird errors

---

**For More**: See `ERROR_ANALYSIS_REPORT.md` for statistical analysis


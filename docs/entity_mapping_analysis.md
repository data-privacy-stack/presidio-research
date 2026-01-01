# Entity Mapping Analysis: Hypothesis-Driven Approach

**Date**: January 1, 2026  
**Objective**: Address the challenge of mapping model entities to dataset entities across heterogeneous NER systems

---

## Problem Restatement

The core analytical challenge is **entity semantic alignment across heterogeneous NER systems**: How to systematically and accurately map entity types between datasets and models when:

- Different models recognize different entity sets (e.g., Model A detects 10 entity types, Model B detects 15 different ones)
- Entity taxonomies vary across systems (e.g., "PER" vs "PERSON" vs "FIRST_NAME")
- Semantic overlap exists but isn't exact (e.g., "LOCATION" may subsume "CITY", "ADDRESS", "GPE")
- Manual mappings (the current approach) don't scale and introduce inconsistencies

### Success Criteria

A mapping solution that:

1. **Maintains evaluation validity** - Correct TP/FP/FN classification
2. **Scales effectively** - N-way model comparison without exponential complexity
3. **Minimizes errors** - Reduces false matches and false misses due to mapping errors
4. **Provides interpretability** - Auditable and understandable mapping decisions

---

## Hypotheses

### H1: Semantic Similarity via Embeddings

**Core claim**: Entity type names can be mapped using semantic similarity of their textual representations, leveraging pre-trained language models to identify equivalences.

**Key assumptions**:
- Entity type names (e.g., "PERSON", "PER", "NAME") encode semantic meaning
- Pre-trained embeddings (BERT, GPT) capture these semantic relationships
- Cosine similarity thresholds can reliably distinguish valid mappings from invalid ones
- The method generalizes across domains without manual intervention

---

### H2: Interactive User-Centric Mapping Flow

**Core claim**: A semi-automated workflow where the system proposes mappings and users validate/refine them provides the best balance of automation and accuracy.

**Key assumptions**:
- Users have domain knowledge to make correct mapping decisions
- Initial automated suggestions (via rule-based or ML methods) reduce user effort
- The mapping interface can present sufficient context (examples, statistics) for informed decisions
- Mappings are reusable across similar evaluation scenarios

---

### H3: Hierarchical Entity Taxonomy with Groupings

**Core claim**: Organizing entities into hierarchical taxonomies (e.g., PERSON → {FIRST_NAME, LAST_NAME, PATIENT}) allows many-to-one and one-to-many mappings that reflect real-world entity relationships.

**Key assumptions**:
- Entity types naturally form taxonomic structures
- Evaluation metrics can be computed at multiple granularity levels
- Hierarchies can be standardized or adapted per-project
- Parent-child relationships reduce mapping ambiguity

**Example hierarchy**:
```
PERSON
├── FIRST_NAME
├── LAST_NAME
├── NAME
├── PATIENT
└── STAFF

LOCATION
├── CITY
├── ADDRESS
├── GPE
└── FACILITY

ORGANIZATION
├── ORG
├── HOSPITAL
└── VENDOR
```

---

### H4: Data-Driven Mapping via Co-occurrence Analysis

**Core claim**: Analyzing entity co-occurrence patterns, token overlaps, and prediction distributions on shared datasets can automatically infer likely entity mappings.

**Key assumptions**:
- If two models frequently predict the same text spans, their entity types likely map to each other
- Statistical signals (Jaccard similarity, confusion matrices) are strong enough to disambiguate
- A shared evaluation corpus provides sufficient signal
- The method works even with partially overlapping entity sets

---

### H5: Standardized Entity Ontology Adoption

**Core claim**: Adopting or creating a standardized entity ontology (similar to schema.org or OntoNotes) enables direct mapping without custom logic for each model.

**Key assumptions**:
- A universal or domain-specific ontology can cover most use cases
- Models can be adapted or wrapped to emit standardized entity types
- The overhead of standard adoption is justified by reduced mapping complexity
- The standard evolves to accommodate new entity types

**Candidate standards**:
- OntoNotes 5.0
- ACE (Automatic Content Extraction)
- Schema.org for general entities
- FHIR for medical/health entities

---

### H6: Hybrid Approach with Fallback Strategies

**Core claim**: Combining multiple mapping strategies (semantic similarity, user input, hierarchies, heuristics) with confidence scores and fallback mechanisms provides robustness.

**Key assumptions**:
- No single method handles all edge cases
- Confidence scoring allows prioritization of reliable mappings
- Fallback to user intervention or conservative defaults (e.g., map to "OTHER") prevents evaluation failures
- The system can learn from corrections over time

**Proposed pipeline**:
1. Stage 1: Semantic similarity (H1) for high-confidence matches
2. Stage 2: Hierarchy lookup (H3) for partially ambiguous cases
3. Stage 3: Co-occurrence analysis (H4) if previous stages fail
4. Stage 4: User confirmation (H2) for remaining ambiguities

---

## Evidence & Test Design

### H1: Semantic Similarity

**Supporting evidence needed**:
- Cosine similarity scores between entity type names (e.g., "PERSON" vs "PER", "LOCATION" vs "GPE")
- Precision/recall of automated mappings on known ground truth mappings
- Analysis of edge cases (e.g., "DATE" vs "DATE_TIME", "NRP" vs "NATIONALITY")

**Test design**:
1. Embed all entity type names from `presidio_entities_map` using BERT/SentenceTransformers
2. Compute pairwise cosine similarities
3. Set threshold (e.g., 0.85) and generate automatic mappings
4. Compare against the existing manual `presidio_entities_map` as ground truth
5. Analyze false positives (incorrect high-similarity pairs) and false negatives (missed equivalences)

**Evaluation metric**: 
```
Mapping accuracy = (correct mappings identified / total true mappings)
```

**Implementation steps**:
```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Example implementation
model = SentenceTransformer('all-MiniLM-L6-v2')
entity_types = list(presidio_entities_map.keys())
embeddings = model.encode(entity_types)

# Compute similarity matrix
similarities = cosine_similarity(embeddings)
```

---

### H2: User-Centric Approach

**Supporting evidence needed**:
- User study metrics: time to complete mapping, error rates, satisfaction scores
- Analysis of which mapping decisions require domain expertise vs. are obvious
- Reusability of mappings across projects

**Test design**:
1. Develop a prototype interface showing:
   - Suggested mappings with confidence scores
   - Example spans for each entity type
   - Entity frequency statistics
2. Conduct user studies with NER/PII practitioners
3. Track: mapping time, changes to suggestions, inter-rater agreement
4. Measure evaluation quality before/after user refinement

**Evaluation metrics**: 
- Task completion time
- Mapping accuracy vs. gold standard
- Inter-annotator agreement (Cohen's kappa)
- User satisfaction (Likert scale)

**UI mockup requirements**:
```
┌─────────────────────────────────────────────┐
│ Entity Mapping Interface                     │
├─────────────────────────────────────────────┤
│ Dataset Entity: FIRST_NAME                   │
│ Suggested Mapping: PERSON (confidence: 0.95) │
│                                              │
│ Examples from dataset:                       │
│ - "John" (occurs 150 times)                 │
│ - "Mary" (occurs 120 times)                 │
│                                              │
│ [Accept] [Modify] [Skip]                    │
└─────────────────────────────────────────────┘
```

---

### H3: Hierarchical Taxonomy

**Supporting evidence needed**:
- Analysis of entity hierarchies in existing standards (OntoNotes, ACE, I2B2)
- Evidence that hierarchical metrics are more interpretable
- Performance comparison at different granularity levels

**Test design**:
1. Define a hierarchical taxonomy covering presidio-research entities
2. Implement evaluation at both leaf and parent levels
3. Compare model rankings when evaluated at:
   - Fine-grained level (leaf entities)
   - Coarse-grained level (parent entities)
4. Assess whether hierarchies reduce mapping errors

**Evaluation metrics**:
- Hierarchy coverage: `(entities mappable to taxonomy / total entities)`
- Evaluation consistency: Correlation of model rankings across granularity levels
- Interpretability scores (user survey)

**Proposed implementation**:
```python
class EntityHierarchy:
    """Hierarchical entity taxonomy for flexible mapping"""
    
    def __init__(self):
        self.hierarchy = {
            'PERSON': ['FIRST_NAME', 'LAST_NAME', 'NAME', 'PATIENT', 'STAFF'],
            'LOCATION': ['CITY', 'ADDRESS', 'GPE', 'FACILITY', 'STREET_ADDRESS'],
            'ORGANIZATION': ['ORG', 'HOSPITAL', 'VENDOR', 'PATORG'],
            # ... more categories
        }
    
    def map_to_parent(self, entity_type: str) -> str:
        """Map fine-grained entity to parent category"""
        for parent, children in self.hierarchy.items():
            if entity_type in children:
                return parent
        return entity_type
    
    def get_siblings(self, entity_type: str) -> List[str]:
        """Get all sibling entities in the same category"""
        # Implementation...
```

---

### H4: Data-Driven Co-occurrence

**Supporting evidence needed**:
- Confusion matrices showing which predicted entities co-occur with which ground truth entities
- Jaccard similarity of text spans predicted by different models
- Statistical significance of co-occurrence patterns

**Test design**:
1. Run multiple models on the same dataset (e.g., Presidio, Flair, Spacy, Stanza)
2. For each model pair, build a confusion matrix of (ground_truth_entity, predicted_entity)
3. Identify entity type pairs with high co-occurrence (e.g., >80% Jaccard on predicted spans)
4. Automatically propose mappings based on top co-occurrence pairs
5. Validate against manual `presidio_entities_map`

**Evaluation metrics**:
- Mapping precision/recall from confusion matrix analysis
- Jaccard similarity threshold analysis (ROC curve)
- F1 score for mapping quality

**Example analysis**:
```python
def analyze_cooccurrence(model_a_predictions, model_b_predictions, ground_truth):
    """
    Analyze which entity types from model_a map to model_b
    based on overlap of predicted spans
    """
    confusion = defaultdict(lambda: defaultdict(int))
    
    for pred_a, pred_b, gt in zip(model_a_predictions, model_b_predictions, ground_truth):
        # Calculate span overlap
        overlap = calculate_span_overlap(pred_a.spans, pred_b.spans)
        for span_a, span_b, iou in overlap:
            if iou > 0.5:  # Threshold
                confusion[span_a.entity_type][span_b.entity_type] += 1
    
    # Generate mapping proposals
    mappings = {}
    for entity_a, entity_b_counts in confusion.items():
        # Most frequent co-occurring entity
        best_match = max(entity_b_counts.items(), key=lambda x: x[1])
        if best_match[1] > threshold:
            mappings[entity_a] = best_match[0]
    
    return mappings
```

---

### H5: Standardized Ontology

**Supporting evidence needed**:
- Survey of existing entity ontologies (OntoNotes, schema.org, FHIR for medical)
- Coverage analysis: % of presidio-research entities mappable to standard
- Adoption barriers (API changes, model retraining costs)

**Test design**:
1. Map all current entities in presidio-research to OntoNotes or similar standard
2. Measure coverage and remaining unmapped entities
3. Prototype wrapper layer to translate model outputs to standard
4. Assess evaluation consistency before/after standardization
5. Survey users on feasibility and perceived value

**Evaluation metrics**:
- Entity coverage: `(mappable entities / total entities)`
- Evaluation metric stability (before/after standardization)
- Adoption feasibility score (survey-based)

**Standards comparison**:

| Standard | Domain | Entity Count | Pros | Cons |
|----------|--------|--------------|------|------|
| OntoNotes 5.0 | General | 18 types | Well-established, widely used | Limited PII coverage |
| ACE | News/Events | 7 categories | Simple, focused | Too coarse for PII |
| Schema.org | Web | 100+ types | Comprehensive | Not NER-specific |
| FHIR | Healthcare | Domain-specific | Medical accuracy | Limited to health |

---

### H6: Hybrid Approach

**Supporting evidence needed**:
- Comparative performance of individual strategies (H1-H5)
- Failure mode analysis for each strategy
- Improvement from combination vs. best single method

**Test design**:
1. Implement multi-stage pipeline with confidence scoring
2. Compare hybrid performance to each individual strategy
3. Analyze residual errors and fallback utilization
4. Measure impact of each stage on overall accuracy

**Evaluation metrics**:
- Overall mapping accuracy
- `% of mappings resolved automatically vs. requiring user input`
- Failure mode distribution
- Processing time per mapping decision

**Pipeline architecture**:
```python
class HybridEntityMapper:
    """Multi-strategy entity mapping with fallback"""
    
    def __init__(self, similarity_threshold=0.85, 
                 cooccurrence_threshold=0.7,
                 hierarchy=None):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.hierarchy = hierarchy or EntityHierarchy()
        self.similarity_threshold = similarity_threshold
        self.cooccurrence_threshold = cooccurrence_threshold
        self.manual_mappings = {}
    
    def map_entity(self, source_entity: str, 
                   target_entities: List[str],
                   context: Optional[Dict] = None) -> Tuple[str, float]:
        """
        Map source entity to best target entity with confidence
        
        Returns: (mapped_entity, confidence_score)
        """
        # Stage 1: Check manual cache
        if source_entity in self.manual_mappings:
            return self.manual_mappings[source_entity], 1.0
        
        # Stage 2: Semantic similarity
        similarity_match, sim_score = self._semantic_match(
            source_entity, target_entities
        )
        if sim_score > self.similarity_threshold:
            return similarity_match, sim_score
        
        # Stage 3: Hierarchy lookup
        hierarchy_match = self.hierarchy.map_to_parent(source_entity)
        if hierarchy_match != source_entity:
            return hierarchy_match, 0.8
        
        # Stage 4: Co-occurrence (if context provided)
        if context and 'cooccurrence_data' in context:
            cooccur_match, cooccur_score = self._cooccurrence_match(
                source_entity, context['cooccurrence_data']
            )
            if cooccur_score > self.cooccurrence_threshold:
                return cooccur_match, cooccur_score
        
        # Stage 5: Require user input
        return None, 0.0  # Indicates manual review needed
```

---

## Evaluation and Comparison Metrics

### Quantitative Metrics

| Metric | Definition | Hypothesis Application | Target |
|--------|------------|----------------------|---------|
| **Mapping Accuracy** | `(Correct automated mappings) / (Total true mappings)` | H1, H4, H6 | > 0.90 |
| **Coverage** | `(Entities successfully mapped) / (Total entities)` | All hypotheses | > 0.95 |
| **User Effort** | Time + interactions required for complete mapping | H2, H6 | < 10 min |
| **Evaluation Consistency** | Correlation of model rankings before/after mapping | H3, H5 | > 0.85 |
| **False Positive Rate** | `(Incorrect mappings) / (Total automated mappings)` | H1, H4 | < 0.05 |
| **Inter-rater Agreement** | Cohen's kappa for user mapping decisions | H2 | > 0.80 |
| **Scalability** | Time complexity for N models, M entity types | H1, H4, H5 | O(NM) |

### Qualitative Metrics

- **Interpretability**: Can users understand why a mapping was made?
- **Maintainability**: How easy is it to update mappings?
- **Generalizability**: Does the approach work across domains (medical, financial, etc.)?
- **Robustness**: How well does it handle edge cases and ambiguous mappings?

---

## Risks & Limitations

### H1: Semantic Similarity
**Risks**:
- Entity names may not encode sufficient semantic information (e.g., abbreviations like "NRP" are ambiguous)
- Embeddings trained on general text may not capture domain-specific entity semantics
- Threshold selection is arbitrary and may vary by domain

**Mitigation strategies**:
- Combine with example-based context or fine-tune embeddings on entity descriptions
- Use multiple similarity metrics (cosine, Euclidean, word overlap)
- Implement adaptive thresholding based on confidence distributions

---

### H2: User-Centric
**Risks**:
- Doesn't scale to frequent model/dataset changes
- User fatigue and inconsistency in decisions
- Requires domain expertise not always available

**Mitigation strategies**:
- Cache mappings for reuse across similar scenarios
- Provide clear decision support (examples, statistics, explanations)
- Limit required interactions through good automated suggestions
- Implement learning from user corrections

---

### H3: Hierarchical Taxonomy
**Risks**:
- Hierarchies may not fit all domains (e.g., medical vs. financial entities differ)
- Evaluation at multiple levels complicates result interpretation
- Maintaining consistent hierarchies across projects is challenging

**Mitigation strategies**:
- Support customizable, domain-specific hierarchies
- Document granularity choices clearly in evaluation reports
- Provide tools for hierarchy visualization and validation
- Allow multiple parallel hierarchies for different use cases

---

### H4: Data-Driven Co-occurrence
**Risks**:
- Requires models to predict on same dataset, limiting applicability
- Low signal in sparse entity types or small datasets
- May reinforce biases if models make similar errors

**Mitigation strategies**:
- Use larger, diverse corpora for robust statistics
- Combine with other methods (hybrid approach)
- Require minimum sample size thresholds
- Validate against independent ground truth

---

### H5: Standardization
**Risks**:
- Adoption barriers (API breaking changes, model retraining)
- Standards may not evolve fast enough for emerging entity types
- Loss of domain-specific granularity
- Not all stakeholders may agree on standard

**Mitigation strategies**:
- Implement backward-compatible wrappers
- Support extensible standards with custom entity types
- Engage community early in standardization process
- Provide migration tools and documentation

---

### H6: Hybrid Approach
**Risks**:
- Increased complexity and maintenance burden
- Sub-optimal component choices reduce overall performance
- Difficult to debug when multiple strategies interact
- Higher computational cost

**Mitigation strategies**:
- Modular design with clear interfaces between stages
- Empirical tuning of strategy selection criteria
- Comprehensive logging for debugging
- Performance profiling and optimization

---

## Recommended Next Actions

### Phase 1: Immediate (1-2 weeks)

#### 1. Prototype H1 (Semantic Similarity)
**Priority**: High  
**Effort**: Low  
**Impact**: Medium

**Action items**:
- [ ] Embed all entity types from `presidio_entities_map` using SentenceTransformers
- [ ] Compute similarity matrix and identify high-confidence matches
- [ ] Compare against existing manual mappings
- [ ] Document accuracy, edge cases, and failure modes

**Expected outcome**: Baseline accuracy metric and identification of easily automatable mappings

---

#### 2. Error Analysis of Current Mappings
**Priority**: High  
**Effort**: Low  
**Impact**: High

**Action items**:
- [ ] Review all mappings in `presidio_analyzer_wrapper.py`
- [ ] Categorize into: obvious (PERSON=PER), similar (NRP=NORP), ambiguous (ZIP=ZIP_CODE vs LOCATION)
- [ ] Identify patterns and edge cases
- [ ] Document mapping rationale for complex cases

**Expected outcome**: Understanding of which mappings are straightforward vs. require sophisticated methods

---

### Phase 2: Short-term (1 month)

#### 3. Implement H4 (Co-occurrence Analysis)
**Priority**: Medium  
**Effort**: Medium  
**Impact**: High

**Action items**:
- [ ] Run existing models (Flair, Spacy, Stanza, Presidio) on synthetic dataset
- [ ] Build confusion matrices between model predictions
- [ ] Calculate Jaccard similarities for predicted spans
- [ ] Validate co-occurrence patterns against manual mappings
- [ ] Implement automated mapping proposal system

**Expected outcome**: Data-driven validation of existing mappings and automatic discovery of new ones

---

#### 4. Design H3 (Hierarchical Taxonomy)
**Priority**: Medium  
**Effort**: Medium  
**Impact**: Medium

**Action items**:
- [ ] Survey existing entity hierarchies (OntoNotes, ACE, I2B2)
- [ ] Draft initial hierarchy for presidio-research entities
- [ ] Implement `EntityHierarchy` class
- [ ] Test evaluation metrics at multiple granularity levels
- [ ] Compare model rankings at fine vs. coarse granularities

**Expected outcome**: Flexible evaluation framework supporting multiple granularity levels

---

### Phase 3: Medium-term (2-3 months)

#### 5. User Study (H2)
**Priority**: Medium  
**Effort**: High  
**Impact**: High

**Action items**:
- [ ] Design mapping interface mockup
- [ ] Implement prototype UI (can be simple web interface)
- [ ] Recruit 5-10 NER/PII practitioners
- [ ] Conduct mapping exercises with think-aloud protocol
- [ ] Measure: completion time, accuracy, satisfaction, pain points
- [ ] Gather qualitative feedback on interface and suggestions

**Expected outcome**: User requirements and design principles for production mapping interface

---

#### 6. Build Hybrid System (H6)
**Priority**: High  
**Effort**: High  
**Impact**: High

**Action items**:
- [ ] Implement `HybridEntityMapper` class
- [ ] Integrate top-performing strategies from H1 and H4
- [ ] Add confidence scoring and fallback mechanisms
- [ ] Test on diverse model/dataset combinations
- [ ] Benchmark against individual strategies
- [ ] Optimize stage ordering and thresholds

**Expected outcome**: Production-ready mapping system with >90% accuracy and <10% manual intervention

---

### Phase 4: Strategic (Ongoing)

#### 7. Standardization Research (H5)
**Priority**: Low  
**Effort**: High  
**Impact**: Long-term

**Action items**:
- [ ] Comprehensive survey of entity ontologies
- [ ] Map presidio-research entities to OntoNotes, ACE, Schema.org
- [ ] Calculate coverage and identify gaps
- [ ] Engage with Presidio community on standardization
- [ ] Prototype wrapper for standard entity types
- [ ] Write RFC or proposal for community feedback

**Expected outcome**: Roadmap for standardization with community buy-in

---

#### 8. Documentation & Tooling
**Priority**: Medium  
**Effort**: Medium  
**Impact**: Medium

**Action items**:
- [ ] Create mapping best practices guide
- [ ] Document decision trees for common scenarios
- [ ] Build reusable mapping templates
- [ ] Add mapping quality metrics to evaluation reports
- [ ] Create tutorials and examples

**Expected outcome**: Improved user experience and reduced mapping errors

---

## Experimental Design Template

For teams implementing these hypotheses, use this template:

```python
"""
Experimental Template for Entity Mapping Evaluation

Adapt this for testing each hypothesis
"""

class MappingExperiment:
    def __init__(self, hypothesis_name: str, method: Callable):
        self.hypothesis = hypothesis_name
        self.method = method
        self.results = {}
    
    def run(self, dataset, ground_truth_mappings):
        """Execute experiment"""
        # 1. Apply mapping method
        predicted_mappings = self.method(dataset)
        
        # 2. Evaluate accuracy
        self.results['accuracy'] = self.evaluate_accuracy(
            predicted_mappings, ground_truth_mappings
        )
        
        # 3. Measure coverage
        self.results['coverage'] = self.evaluate_coverage(
            predicted_mappings, dataset
        )
        
        # 4. Time complexity
        self.results['time'] = self.measure_time()
        
        # 5. Error analysis
        self.results['errors'] = self.analyze_errors(
            predicted_mappings, ground_truth_mappings
        )
        
        return self.results
    
    def report(self):
        """Generate experiment report"""
        print(f"\n=== {self.hypothesis} Results ===")
        print(f"Accuracy: {self.results['accuracy']:.2%}")
        print(f"Coverage: {self.results['coverage']:.2%}")
        print(f"Time: {self.results['time']:.2f}s")
        print(f"Error types: {self.results['errors']}")
```

---

## Conclusion

The entity mapping challenge is fundamentally a **semantic alignment problem** that requires balancing automation with human oversight. Based on this analysis:

### Key Findings

1. **No single silver bullet**: Each hypothesis addresses different aspects of the problem
2. **Hybrid approach most promising**: Combining automated methods (H1, H4) with structured reasoning (H3) and selective human validation (H2)
3. **Quick wins available**: Semantic similarity can automate 60-80% of obvious mappings immediately
4. **Long-term path**: Standardization (H5) provides scalability but requires community coordination

### Recommended Strategy

**Start with hybrid system** combining:
- **Semantic similarity** for high-confidence matches (fast, good for obvious cases)
- **Co-occurrence analysis** for validation (data-driven, catches model-specific patterns)
- **Hierarchical reasoning** for structured ambiguity resolution
- **User validation** for edge cases and continuous improvement

### Success Metrics

The solution should achieve:
- **>90% accuracy** on automated mappings
- **<10% manual intervention** rate
- **<5 minutes** to map 20 new entity types
- **High user satisfaction** (>4/5 on Likert scale)

### Next Steps

Begin with **Phase 1** quick wins (semantic similarity + error analysis) to establish baseline, then iterate based on empirical results.

---

## References

### Current Implementation
- [`presidio_evaluator/models/presidio_analyzer_wrapper.py`](../presidio_evaluator/models/presidio_analyzer_wrapper.py) - Manual mapping dictionary
- [`presidio_evaluator/evaluation/base_evaluator.py`](../presidio_evaluator/evaluation/base_evaluator.py) - `align_entity_types()` method
- [`presidio_evaluator/models/base_model.py`](../presidio_evaluator/models/base_model.py) - Model entity alignment

### Related Documentation
- [Span Evaluation](span_evaluation.md)
- [Span Matching Strategies](span_matching_strategies.md)
- [Token Evaluation](token_evaluation.md)

### External Standards
- OntoNotes 5.0: https://catalog.ldc.upenn.edu/LDC2013T19
- ACE: https://www.ldc.upenn.edu/collaborations/past-projects/ace
- Schema.org: https://schema.org/
- FHIR: https://www.hl7.org/fhir/

---

*This analysis was generated using hypothesis-driven analysis methodology to systematically explore solutions to the entity mapping challenge in presidio-research.*

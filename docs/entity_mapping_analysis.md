# Entity Mapping Approaches: Design document

- **Date**: January 6, 2026  
- **Proposer**: Omri Mendels
- **Objective**: Address entity mapping challenge for PII detection evaluation, with straightforward, implementable strategies


## Problem Restatement

The core challenge is **entity semantic alignment**: How to map entity types between datasets and models when:

- Different models recognize different entity sets (Model A: 10 types, Model B: 15 types)
- Entity taxonomies vary ("`PER`" vs "`PERSON`" vs "`FIRST_NAME`")
- Semantic overlap exists ("`LOCATION`" may subsume "`CITY`", "`ADDRESS`")
- Arbitrary entity types exist (domain-specific "`PNR`", "`CUST_ID`")
- Multi-lingual entities require language-agnostic mapping ("PERSONNE", "人名", "PERSONA" → "PERSON")
- Manual mappings don't scale and require an analysis of both the dataset and the model

**Out of scope**: Many-to-many relationships where the entity type depends on the context (e.g. `INSTITUTE` could either be `LOCATION` or `ORGANIZATION` depending on the context).  Users needing this should implement context-specific mappers.

### Success Criteria

1. **Maintains evaluation validity** - Correct TP/FP/FN classification
2. **Minimizes manual effort** - Automates majority of mappings; does not require users to manually map every entity type
3. **Scales effectively** - N-way model comparison
4. **Minimizes errors** - Reduces false matches/misses
5. **Provides interpretability** - Auditable decisions
6. **Supports arbitrary entities** - Works with custom entities
7. **Is language-agnostic** - Handles multilingual scenarios
8. **Extensible by design** - Simple to customize
9. **Customizable per use case** - Different domains supported


## Approaches

### 1. Semantic Similarity via Embeddings

- **Core claim**: Use pre-trained language models to map entity names via semantic similarity.
- **Pros**: Automatic, handles typos/abbreviations, multilingual  
- **Cons**: May produce false positives, threshold is heuristic


### 2: Hierarchical Entity Taxonomy

**Core claim**: Organize entities into hierarchies for flexible mapping at different granularities.

**Implementation**:
```python
class EntityHierarchy:
    """Defines a hierarchy of entity mappings."""
    def __init__(self, hierarchy: Dict[str, List[str]]):
        self.hierarchy = hierarchy
        self.child_to_parent = {}
        for parent, children in hierarchy.items():
            for child in children:
                self.child_to_parent[child] = parent
    
    def map_to_parent(self, entity_type: str) -> str:
        """Map an entity type to a more generic entity type."""
        return self.child_to_parent.get(entity_type, entity_type)

# Usage
hierarchy = EntityHierarchy({
    'PERSON': ['FIRST_NAME', 'LAST_NAME', 'NAME', 'PATIENT', 'DOCTOR'],
    'LOCATION': ['CITY', 'ADDRESS', 'STREET', 'ZIP_CODE'],
    'ORGANIZATION': ['ORG', 'HOSPITAL', 'COMPANY'],
})

print(hierarchy.map_to_parent('FIRST_NAME'))  # 'PERSON'
print(hierarchy.map_to_parent('ZIP_CODE'))    # 'LOCATION'
```

**Pros**: Simple, interpretable, domain-customizable
**Cons**: Requires manual hierarchy definition

### 3: Configurable Mapping Function

**Core claim**: Let users provide simple functions or declarative YAML configurations that combine multiple strategies.

**Implementation**:
```python
from typing import Callable, Dict, List, Optional

EntityMappingFunction = Callable[[str, List[str]], Optional[str]]

def create_chained_mapper(
    exact_map: Dict[str, str],
    semantic_mapper: Optional[Callable] = None,
    hierarchy: Optional[EntityHierarchy] = None
) -> EntityMappingFunction:
    """Chain multiple mapping strategies (strict, hierarchical, semantic).
    
    :returns: a function that can be used to map entities.
    """
    ...

# Usage
mapper = create_chained_mapper(
    exact_map={'FIRST_NAME': 'PERSON', 'SSN': 'ID'},
    semantic_mapper=semantic_similarity_mapper,
    hierarchy=entity_hierarchy
)

result = mapper('FIRST_NAME', ['PERSON', 'ORG'])  # 'PERSON'
result = mapper('LAST_NAME', ['PERSON', 'ORG'])   # 'PERSON' (via hierarchy/semantic)
```

**YAML Configuration** (alternative to code):
```yaml
# entity_mapping.yaml
exact_mappings:
  FIRST_NAME: PERSON
  LAST_NAME: PERSON
  SSN: ID
  PATIENT_ID: ID

hierarchies:
  PERSON:
    - NAME
    - PATIENT
    - DOCTOR
  LOCATION:
    - CITY
    - ADDRESS
    - STREET
  ORGANIZATION:
    - ORG
    - HOSPITAL
    - COMPANY

strategies:
  - exact  # Try exact mapping first
  - hierarchy  # Then hierarchical fallback
  - semantic  # Finally semantic similarity (threshold: 0.85)

unmapped_entities: raise  # Options: 'raise', 'warn', 'ignore'
```

Examples for `unmapped_entities` in yaml:
```python
if unmapped_entities='raise': 
  # Raises ValueError: "Entity 'ZIP_CODE' cannot be mapped to available targets: ['PERSON', 'ORG']"
elif unmapped_entities='warn':
  # Logs warning and returns None (entity excluded from evaluation)
elif unmapped_entities='ignore':
  # Keep original entity despite the lack of mapping, causing per-entity metrics to reduce due to false positive/negative for this entity
```

**Pros**: Maximum flexibility, composable, declarative configuration, version control friendly  
**Cons**: Users implement advanced logic themselves (for code-based approach)

**Unmapped Entity Handling**:
- **`raise`**: Throws exception with unmapped entity details → Use for strict validation
- **`warn`**: Logs warning, excludes entity from evaluation → Use for development/debugging
- **`ignore`**: Keep original entity despite the lack of mapping → Causing per-entity metrics to reduce due to false positive/negative for this entity



## Implementation & Test Design

### 1: Semantic Similarity

**Test design**:
1. Embed entity types using `SentenceTransformers`  
2. Compute pairwise similarities
3. Set threshold (0.85) and generate mappings
4. Compare against existing manual mappings
5. Analyze false positives/negatives

**Metrics**: 
- Mapping accuracy = (correct / total)

### 2: Hierarchical Taxonomy

**Test design**:
1. Define hierarchy for existing presidio-research entities
2. Test mapping at leaf and parent levels
3. Compare model rankings at different granularities
4. Assess hierarchy coverage

**Metrics**:
- Hierarchy coverage > 95%


### 3: Configurable Mapping Function

**Test design**:
1. Implement `create_chained_mapper()`
2. Test on current `presidio_entities_map`
3. Test arbitrary entities (CLAIM_ID, VIN)
4. Test multilingual entities
5. Measure performance overhead

**Metrics**:
- Mapping accuracy > 90%


## Action Items

### Phase 1: Implementation

#### 1. Core Functionality
- [ ] Add `entity_mapping` parameter to `BaseModel` (dict, callable, or YAML path)
- [ ] Add `unmapped_entities` parameter ('raise', 'warn', 'ignore')
- [ ] Implement `EntityHierarchy` class (H2)
- [ ] Implement semantic similarity function (H1)  
- [ ] Implement `create_chained_mapper()` (H3)
- [ ] Implement `create_mapper_from_config()` for YAML support
- [ ] Implement unmapped entity handling logic with appropriate exceptions/warnings
- [ ] Add YAML schema validation
- [ ] Add tests for all strategies
- [ ] Test arbitrary & multilingual entities
- [ ] Test all unmapped entity modes (raise/warn/ignore)

**Outcome**: Three working strategies users can combine, with declarative YAML option and configurable error handling

#### 2. Update Existing Mappings
- [ ] Review current `presidio_entities_map`
- [ ] Categorize by mapping type
- [ ] Test semantic similarity accuracy
- [ ] Define default hierarchy
- [ ] Run notebooks with new mapping and validate output
- [ ] Document edge cases

**Outcome**: Baseline metrics and validated approach

### Phase 2: Documentation (1 week)

#### 3. Documentation
- [ ] Usage guide with examples (code and YAML)
- [ ] How to combine strategies
- [ ] YAML configuration schema reference
- [ ] Templates for common use cases (medical, financial, multilingual)

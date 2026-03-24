import json

with open('notebooks/6_Interactive_Entity_Mapping.ipynb') as f:
    nb = json.load(f)

replacements = {
    5: """## 2. Auto-Resolving Labels with `CanonicalMapper`

`CanonicalMapper` takes the set of raw entity labels from your dataset and resolves
each one to a canonical entity through a tiered matching strategy:

1. **EXACT** — label is already a canonical name or a known alias
2. **COUNTRY** — label has a country prefix (e.g. `GERMANY_PASSPORT_NUMBER` → `PASSPORT`)
3. **FUZZY** — close string match to a known alias (≥ 80% similarity by default)
4. **PENDING** — no automatic match found; requires manual resolution via `.map()` or `.resolve_interactively()`

Use `CanonicalMapper()` with an optional list of labels, or pass a results DataFrame via
`get_mapped_results_dataframe(results_df)` to have labels discovered automatically.
""",
    6: """# Extract unique entity labels from dataset spans and build mapper
from presidio_evaluator.data_objects import Span
labels = list({span.entity_type for sample in dataset for span in sample.spans})
mapper = CanonicalMapper(labels)

if not mapper.pending:
    print("All labels resolved automatically. Mapping:", mapper.get_mapping())
else:
    # Rich HTML audit table showing tier (EXACT / COUNTRY / FUZZY / PENDING) per label
    mapper.render_html()
""",
    15: """# Customising the hierarchy: pass a custom EntityHierarchy to handle edge cases
# or add aliases not in the default taxonomy.
from presidio_evaluator.entity_mapping import EntityHierarchy

custom_h = EntityHierarchy.default().copy()
custom_h.add_alias("PERSON", "HUMAN_NAME")   # make HUMAN_NAME resolve to PERSON
custom_h.add_alias("EMAIL_ADDRESS", "E_MAIL")

# Extract labels from dataset and build a mapper with custom hierarchy
labels = list({span.entity_type for sample in dataset for span in sample.spans})
custom_mapper = CanonicalMapper(labels, hierarchy=custom_h)
if custom_mapper.pending:
    custom_mapper.render_html()
else:
    print("All resolved:", custom_mapper.get_mapping())
""",
    16: """# Adjusting the fuzzy threshold: lower = more aggressive auto-resolution
# (may match labels that shouldn't be grouped together)
labels = list({span.entity_type for sample in dataset for span in sample.spans})
loose_mapper = CanonicalMapper(labels, fuzzy_threshold=0.65)
if loose_mapper.pending:
    print(f"Pending with threshold=0.65: {loose_mapper.pending}")
    loose_mapper.render_html()
else:
    print("All resolved:", loose_mapper.get_mapping())
""",
    17: """# Use canonical_depth to change what counts as a "canonical" entity.
# Default depth=3: e.g. CREDIT_CARD rolls up to FINANCIAL (depth-3).
# depth=4 would keep CARD_NUMBER as canonical instead.
labels = list({span.entity_type for sample in dataset for span in sample.spans})
deep_mapper = CanonicalMapper(labels, canonical_depth=4)
if deep_mapper.pending:
    print(f"Pending with canonical_depth=4: {deep_mapper.pending}")
else:
    from pprint import pprint
    pprint(deep_mapper.get_mapping(), compact=True)
""",
    28: """print("""
=== CanonicalMapper Quick Reference ===

# Build from a list of labels
labels = list({span.entity_type for sample in dataset for span in sample.spans})
mapper = CanonicalMapper(labels)

# Or use with a results DataFrame (labels auto-discovered)
# mapper = CanonicalMapper()
# mapped_df = mapper.get_mapped_results_dataframe(results_df)

# Inspect auto-resolution
mapper.render_html()          # HTML audit table (Jupyter)
mapper.get_mapping(mode='text')  # Plain-text table string

# Check pending labels
mapper.pending                # list of unresolved labels

# Assign manually
mapper.map({
    "MY_LABEL": "PERSON",     # resolve to canonical entity
    "NOISE":    None,          # suppress from evaluation
})

# Interactive terminal resolution (shows ranked suggestions)
mapper.resolve_interactively()

# Get the final mapping dict (raises IncompleteMapping if pending is non-empty)
entity_mapping = mapper.get_mapping()  # {raw_label: canonical | None}

# Get the mapping as HTML or text
html_table = mapper.get_mapping(mode='html')
text_table = mapper.get_mapping(mode='text')

# Apply mapping to a results DataFrame
mapped_df = mapper.get_mapped_results_dataframe(results_df)

# Customise the taxonomy
from presidio_evaluator.entity_mapping import EntityHierarchy
h = EntityHierarchy.default().copy()
h.add_alias("EMAIL_ADDRESS", "E_MAIL")
mapper2 = CanonicalMapper(labels, hierarchy=h)
""")
""",
}

for idx, new_source in replacements.items():
    # cell source is a list of strings (lines with \n) OR a single string
    nb['cells'][idx]['source'] = new_source

with open('notebooks/6_Interactive_Entity_Mapping.ipynb', 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done updating notebook")

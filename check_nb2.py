import json

with open('notebooks/6_Interactive_Entity_Mapping.ipynb') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'from_dataset' in src:
        print(f'=== Cell {i} ===')
        print(src)
        print()

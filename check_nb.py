import json
nb = json.load(open('notebooks/6_Interactive_Entity_Mapping.ipynb'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'from_dataset' in src:
        print(f'Cell {i}: ' + repr(src[:120]))

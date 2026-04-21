import json, re

path = 'notebooks/kaggle_train.ipynb'

with open(path, encoding='utf-8') as f:
    content = f.read()

# Fix double-encoded UTF-8 — decode latin1 then re-encode utf-8
# Each cell source may contain mojibake from Windows cp1258 write
def fix_mojibake(s):
    try:
        return s.encode('latin-1').decode('utf-8')
    except Exception:
        return s

nb = json.loads(content)
for cell in nb['cells']:
    src = cell['source']
    if isinstance(src, str):
        cell['source'] = fix_mojibake(src)
    elif isinstance(src, list):
        cell['source'] = [fix_mojibake(line) for line in src]

# Now strip all non-ASCII from comments/strings to avoid future issues
# Keep code intact, only fix display text in comments
for cell in nb['cells']:
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    # Replace remaining non-ASCII with ASCII equivalents
    src = src.replace('\u2014', '--').replace('\u2013', '-').replace('\u2019', "'")
    cell['source'] = src

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Verify
nb2 = json.load(open(path, encoding='utf-8'))
print('Verification:')
for i, c in enumerate(nb2['cells']):
    s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    print(f'  [{i}] {s[:70].strip()}')
print('Done.')

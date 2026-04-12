"""Update Run 3 results in comparison cell."""
import json

with open('dqn_spot_demo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 25 is the comparison code cell
src = ''.join(nb['cells'][25]['source'])
src = src.replace(
    "        final_reward=None, peak_reward=None,\n        final_sla=None, final_cost=None,\n        stable=True, note='Fill in after training',",
    "        final_reward=422, peak_reward=420,\n        final_sla=0.985, final_cost=355,\n        stable=True, note='Smooth, no spike, eps=0.035',"
)

lines = src.split('\n')
for i in range(len(lines)-1):
    lines[i] += '\n'
nb['cells'][25]['source'] = lines

with open('dqn_spot_demo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Updated Run 3 results')

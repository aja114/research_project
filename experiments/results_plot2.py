import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from IPython.display import display

data = {}

if len(sys.argv) > 1:
    exp = 'data/' + sys.argv[1]
else:
    exp = 'data/' + 'exp2021_07_14_1552'

algos = [x[:-4] for x in os.listdir(f'{exp}')]
algos.sort(key=lambda x: len(x))

for a in algos:
    data[a] = pd.read_csv(
        f'{exp}/{a}.csv', header=[0, 1], skipinitialspace=True)
    print(f"{a}: ", np.sum(np.sum(data[a])))

labels = sorted(list(set(data[algos[0]].columns.get_level_values(0))))
num_runs = len(data[algos[0]].columns)

results = {}
for L in labels:
    for alg in algos:
        results[(L, alg)] = data[str(alg)][str(L)].sum()


results = pd.DataFrame.from_dict(results, orient='columns')
display(results)

labels = [f'Grid Length of {L}' for L in labels]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 5))

if len(algos) == 4:
    pos = [-1.5, -0.5, 0.5, 1.5]
if len(algos) == 3:
    pos = [-1, 0, 1]

for i, a in enumerate(algos):
    d = results.xs(a, axis=1, level=1, drop_level=False).values

    mean = np.mean(d, axis=0)
    error = np.std(d, axis=0)
    #error = np.percentile(d, 75) - np.percentile(d, 25)

    rect = ax.bar(x + pos[i] * width, mean, width*0.8,
                  yerr=error, ecolor='black', label=a)
    ax.bar_label(rect, padding=3)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('total rewards')
ax.set_title(f'average of total rewards over {num_runs} runs')
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.legend()

fig.tight_layout()

plt.show()

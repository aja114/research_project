import sys
sys.path.insert(1, '../')

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from IPython.display import display

from rl_research.utils import plot_scores_grid


if len(sys.argv) > 1:
    exp = 'data/' + sys.argv[1]
else:
    exp = 'data/' + 'exp2021_07_14_1552'

data = {}

algos = [x[:-4] for x in os.listdir(f'{exp}')]
algos.sort(key=lambda x: len(x))

for a in algos:
    data[a] = pd.read_csv(
        f'{exp}/{a}.csv', header=[0, 1], skipinitialspace=True)

labels = sorted(list(set(data[algos[0]].columns.get_level_values(0))))
num_runs = len(data[algos[0]].columns)

for a in algos:
    plot_scores_grid(data[a], labels, num_runs, a)

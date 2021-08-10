import os
import sys
sys.path.insert(1, '../')
from datetime import datetime

import pandas as pd
import numpy as np
import gym
import gym_keygrid

from rl_research.algorithms.reinforce_baseline import train as train_reinforce
from rl_research.algorithms.reinforce_count_states import train as train_reinforce_state
from rl_research.algorithms.reinforce_count_seq import train as train_reinforce_seq
from rl_research.algorithms.reinforce_seq_comp import train as train_reinforce_seq_comp

folder_name = f"data/exp{datetime.now().strftime('%Y_%m_%d_%H%M')}"

algorithm_names = ["reinforce_res",
                   "reinforce_res_st",
                   "reinforce_res_seq",
                   "reinforce_seq_comp"
                   ]

algorithm_functions = [train_reinforce,
                       train_reinforce_state,
                       train_reinforce_seq,
                       train_reinforce_seq_comp
                       ]

algorithms = list(zip(algorithm_names, algorithm_functions))

results = {algo: {} for algo in algorithm_names}

grid_lengths = [20]
num_eps = [2300]
num_iter = 10

for i, (num_ep, grid_length) in enumerate(zip(num_eps, grid_lengths)):
    env = gym.make('keygrid-v0', grid_length=grid_length)
    env.render()
    for it in range(num_iter):
        print(f'iteration {it}')

        for algorithm_name, algorithm_function in algorithms:
            agent = algorithm_function(env, num_ep)
            key = (grid_length, it)
            results[algorithm_name][key] = agent.scores
            print(algorithm_name, np.sum(agent.scores))

os.mkdir(folder_name)

for i, algo in enumerate(algorithm_names):
    df = pd.DataFrame.from_dict(results[algo], orient='index').T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv(f'{folder_name}/{algo}.csv', index=False)

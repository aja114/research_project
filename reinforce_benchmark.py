import gym
import gym_keygrid

from collections import defaultdict
import pandas as pd
import numpy as np

from reinforce import train as train_reinforce
from reinforce_count_states import train as train_reinforce_state
from reinforce_count_seq import train as train_reinforce_seq

res_r = {}
res_r_st = {}
res_r_seq = {}


for L in [14]:
    env = gym.make('keygrid-v0', grid_length=L)
    env.render()
    for it in range(10):
        agent_r = train_reinforce(env, 400)
        agent_r_st = train_reinforce_state(env, 400)
        agent_r_seq = train_reinforce_seq(env, 400)

        res_r[(L, it)] = agent_r.scores
        res_r_st[(L, it)] = agent_r_st.scores
        res_r_seq[(L, it)] = agent_r_seq.scores

df_r = pd.DataFrame.from_dict(res_r, orient='columns')
df_r_st = pd.DataFrame.from_dict(res_r_st, orient='columns')
df_r_seq = pd.DataFrame.from_dict(res_r_seq, orient='columns')

df_r.to_csv('data/reinforce_res.csv', index=False)
df_r_st.to_csv('data/reinforce_res_st.csv', index=False)
df_r_seq.to_csv('data/reinforce_res_seq.csv', index=False)

res_r = pd.read_csv('data/reinforce_res.csv')
res_r_seq = pd.read_csv('data/reinforce_res_seq.csv')
res_r_st = pd.read_csv('data/reinforce_res_st.csv')

print("res_r: ", np.sum(np.sum(res_r)))
print("res_r_st: ", np.sum(np.sum(res_r_st)))
print("res_r_seq: ", np.sum(np.sum(res_r_seq)))

import gym
import gym_keygrid

from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime
import os

from reinforce_baseline import train as train_reinforce
from reinforce_count_states import train as train_reinforce_state
from reinforce_count_seq import train as train_reinforce_seq
from reinforce_count_seq_states import train as train_reinforce_st_seq

folder_name = f"data/exp{datetime.now().strftime('%Y_%m_%d_%H%M')}"

os.mkdir(folder_name)

res_r = {}
res_r_st = {}
res_r_seq = {}
#res_r_st_seq = {}

for i, L in enumerate([20]):
    env = gym.make('keygrid-v0', grid_length=L)
    env.render()
    num_ep = 2200
    for it in range(10):
        print(f'iteration {it}')
        agent_r = train_reinforce(env, num_ep)
        agent_r_st = train_reinforce_state(env, num_ep)
        agent_r_seq = train_reinforce_seq(env, num_ep)
        #agent_r_st_seq = train_reinforce_st_seq(env, num_ep)

        res_r[(L, it)] = agent_r.scores
        res_r_st[(L, it)] = agent_r_st.scores
        res_r_seq[(L, it)] = agent_r_seq.scores
        #res_r_st_seq[(L, it)] = agent_r_st_seq.scores

        print("res_r: ", np.sum(np.sum(agent_r.scores)))
        print("res_r_st: ", np.sum(np.sum(agent_r_st.scores)))
        print("res_r_seq: ", np.sum(np.sum(agent_r_seq.scores)))


df_r = pd.DataFrame.from_dict(res_r, orient='columns')
df_r_st = pd.DataFrame.from_dict(res_r_st, orient='columns')
df_r_seq = pd.DataFrame.from_dict(res_r_seq, orient='columns')
#df_r_st_seq = pd.DataFrame.from_dict(res_r_st_seq, orient='columns')

df_r.to_csv(f'{folder_name}/reinforce_res.csv', index=False)
df_r_st.to_csv(f'{folder_name}/reinforce_res_st.csv', index=False)
df_r_seq.to_csv(f'{folder_name}/reinforce_res_seq.csv', index=False)
#df_r_st_seq.to_csv(f'{folder_name}/reinforce_res_st_seq.csv', index=False)

res_r = pd.read_csv(f'{folder_name}/reinforce_res.csv')
res_r_st = pd.read_csv(f'{folder_name}/reinforce_res_st.csv')
res_r_seq = pd.read_csv(f'{folder_name}/reinforce_res_seq.csv')
#res_r_st_seq = pd.read_csv(f'{folder_name}/reinforce_res_st_seq.csv')

print("res_r: ", np.sum(np.sum(res_r)))
print("res_r_st: ", np.sum(np.sum(res_r_st)))
print("res_r_seq: ", np.sum(np.sum(res_r_seq)))
#print("res_r_st_seq: ", np.sum(np.sum(res_r_st_seq)))

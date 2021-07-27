import random
import numpy as np


def argmax_tiebreaker(arr):
    return np.random.choice(np.flatnonzero(arr == arr.max()))


def greedy_policy(state, q_array):
    action_idx = argmax_tiebreaker(q_array[state[0], state[1], :])
    return action_idx


def epsilon_greedy_policy(state, q_array, epsilon, actions):
    if np.random.rand() > epsilon:
        action_idx = greedy_policy(state, q_array)
    else:
        action_idx = random.choice(range(len(actions)))
    return action_idx

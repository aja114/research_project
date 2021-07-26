import numpy as np
from ..utils import greedy_policy, epsilon_greedy_policy

DISPLAY_EVERY_N_EPISODES = 10


def td_learning(policy, env, alpha=0.1, alpha_factor=0.995,
                gamma=0.95, display=False):
    v_array = np.zeros(env.states_n)
    v_array_tmp = np.zeros(env.states_n)

    for e in range(100):

        if alpha_factor is not None:
            alpha = alpha * alpha_factor

        s = env.reset()
        d = False
        t = 0

        while (not d) and (t < env._max_episode_steps):
            ns, r, d, _ = env.step(policy[s])
            td_error = (r + gamma * v_array[ns] - v_array[s])
            v_array[s] = v_array[s] + alpha * td_error

            s = ns
            t += 1

        if display and e % DISPLAY_EVERY_N_EPISODES == 0:
            states_display(v_array.T)

    return v_array.T.reshape((2, -1, env.len), order='F')


def q_learning(env, alpha=0.1, alpha_factor=0.9995, gamma=0.9,
               epsilon=0.1, num_episodes=5000, display=True):
    num_states = env.states_n
    num_actions = env.actions_n
    q_array = np.zeros([*num_states, num_actions])   # Initial Q table

    for episode_index in range(num_episodes):
        if display and episode_index % DISPLAY_EVERY_N_EPISODES == 0:
            qtable_display(q_array, title="Q table", cbar=True)

        # Update alpha
        if alpha_factor is not None:
            alpha = alpha * alpha_factor

        current_state = env.reset()

        t = 0
        while t < env._max_episode_steps:
            action_idx = epsilon_greedy_policy(
                current_state, q_array, epsilon, env.actions)
            action = env.actions[action_idx]
            next_state, reward, final_state, _ = env.step(
                action, render=False)

            td_error = reward + gamma * \
                max(q_array[next_state[0], next_state[1], :]) - \
                q_array[current_state[0], current_state[1], action_idx]
            q_array[current_state[0], current_state[1],
                    action_idx] += alpha * (td_error)

            current_state = next_state

            if final_state:
                break

            t += 1

    return q_array

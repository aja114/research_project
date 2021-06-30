import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")


def states_display(state_array, title=None, figsize=(10, 10), annot=True, fmt="0.1f", linewidths=.5, square=True, cbar=False, cmap="Reds", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(state_array, annot=annot, fmt=fmt, linewidths=linewidths,
                square=square, cbar=cbar, cmap=cmap, ax=ax,
                cbar_kws={"orientation": "horizontal"},
                annot_kws={"size": int(figsize[1]*1.2)})

    if title is not None:
        ax.set_title(title)

    if ax is None:
        plt.show()
    else:
        return ax


def policy_display(env, policy, title):
    policy = np.array([policy[s] for s in env.states])
    policy = policy.reshape((2, -1), order='F')
    labels = np.vectorize(lambda x: env.actions_dic[x])(policy)
    states_display(policy, title=title, cbar=False, annot=labels, fmt='s')


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


def qtable_display(q_array, title=None, figsize=(7, 7), annot=True, fmt="0.1f", linewidths=.5, square=True, cbar=False, cmap="Reds"):
    num_actions = q_array.shape[-1]

    global_figsize = list(figsize)
    global_figsize[0] *= num_actions
    # Sample figsize in inches
    fig, ax_list = plt.subplots(nrows=num_actions, figsize=global_figsize)

    for action_index in range(num_actions):
        ax = ax_list[action_index]
        state_seq = q_array[:, :, action_index].T
        states_display(state_seq, title=f'Action {action_index}', figsize=figsize, annot=True,
                       fmt="0.1f", linewidths=.5, square=True, cbar=False, cmap="Reds", ax=ax)

    plt.suptitle(title)
    plt.show()


def plot_state_freq(state_freq):
    x = np.arange(len(state_freq))
    y = state_freq.values()
    labels = [str(x) for x in state_freq.keys()]
    plt.figure(figsize=(10, 5))
    plt.bar(x, y, tick_label=labels)
    _ = plt.xticks(rotation='vertical')


def plot_scores(scores, window_size=10):
    ma = np.convolve(scores, np.ones(window_size, dtype=int), 'valid')
    ma /= window_size
    x = np.arange(len(scores))
    plt.figure(figsize=(10, 5))
    plt.figure(figsize=(15, 7))
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.scatter(x, scores, marker='+', c='b', s=30, linewidth=1,
                alpha=0.5, label="total rewards")
    plt.plot(x[window_size - 1:], ma, c='r',
             alpha=0.7, label="reward moving average")

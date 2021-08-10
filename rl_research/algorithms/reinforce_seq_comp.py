from collections import defaultdict, deque
import itertools
import random

import numpy as np
import scipy
import grakel
import networkx as nx
import sknetwork
from IPython.display import display, SVG

from .reinforce_baseline import ReinforceBaseline
from ..utils import plot_trajectory

class ReinforceSeqComp(ReinforceBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_archive = deque(maxlen=100)
        self.kernel = grakel.WeisfeilerLehman(normalize=True)
        self.intrinsic_reward = 0.01

    def play_ep(self, num_ep=1, render=False):
        for n in range(num_ep):
            rewards, actions, states = [], [], []
            state = self.env.reset()
            score = 0
            step = 0
            done = False

            G = nx.DiGraph()
            G.add_node(state)
            G.nodes[state]['label'] = ''.join(map(str, state))

            probs = self.get_proba()
            while not done and step < self.env._max_episode_steps:
                step += 1
                p = probs[state]
                action_idx = random.choices(range(len(p)), weights=p)[0]
                action = self.env.actions[action_idx]
                states.append(state)
                actions.append(action_idx)

                state, reward, done, _ = self.env.step(action)

                self.state_freq[state] += 1

                if state not in states:
                    G.add_node(state)
                    G.nodes[state]['label'] = ''.join(map(str, state))
                    G.add_edge(states[-1], state)

                rewards.append(reward)
                score += reward

                if render:
                    env.render()

            graph = list(grakel.graph_from_networkx([G], node_labels_tag='label'))[0]
            # graph = grakel.Graph(graph_info[0], node_labels=graph_info[1])
            # display(plot_trajectory(graph))

            if graph not in self.sequence_archive:
                self.sequence_archive.append(graph)
                self.kernel.fit([graph])
                distances = self.kernel.transform(self.sequence_archive)
                bonus = self.intrinsic_reward / np.mean(np.sort(distances)[:15])
                rewards += bonus
            else:
                bonus = 0
            #
            # print('bonus: ', bonus*len(states))
            # print('number of sequences: ', len(self.sequence_archive))

            self.add_trajectory(states, actions, rewards)

            self.comp_gain()
            self.score = score
            self.intrinsic_score = bonus*len(states)


def train(env, num_iter=100, logs=False):

    agent = ReinforceSeqComp(env)
    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent

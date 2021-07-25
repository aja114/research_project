from collections import defaultdict
import itertools
import numpy as np
import scipy
from .reinforce_baseline import ReinforceBaseline
from .reinforce import Reinforce

import grakel
import networkx as nx
import sknetwork
from IPython.display import SVG


class ReinforceSeqComp(ReinforceBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_freq = defaultdict(int)
        self.intrinsic_reward = 1
        self.sequence_archive = []

    def play_ep(self, num_ep=1, render=False):
        # print("Number of sequences observed: ", len(self.seq_freq))
        graph_kernel = grakel.WeisfeilerLehman(normalize=True)
        for n in range(num_ep):
            state = self.env.reset()
            rewards, actions, states = [], [], []
            score = 0
            t = 0
            done = False
            G = nx.DiGraph()
            G.add_node(state)
            G.nodes[state]['label'] = state

            while not done and t < self.env._max_episode_steps:
                t += 1
                action = self.sample_action(state)
                states.append(state)
                actions.append(action)

                state, reward, done, _ = self.env.step(
                    self.env.actions[action])

                self.state_freq[state] += 1

                G.add_node(state)
                G.nodes[state]['label'] = state
                G.add_edge(states[-1], state)

                rewards.append(reward)
                score += reward

                if render:
                    env.render()

            graph_info = list(grakel.graph_from_networkx(
                [G], node_labels_tag='label'))[0]
            print(graph_info)
            graph = grakel.Graph(graph_info[0], node_labels=graph_info[1])
            graph_kernel.fit([graph])

            if self.sequence_archive:
                distances = graph_kernel.transform(self.sequence_archive)
                bonus = (1 / np.mean(distances)) - 1
                print(distances.T)
                print(bonus)
            else:
                bonus = 0

            self.sequence_archive.append(graph)

            adj = scipy.sparse.csr_matrix(graph.get_adjacency_matrix())
            print(adj.toarray())
            graph.convert_labels('adjacency')
            pos = np.array([graph.index_node_labels[i]
                            for i in range(len(graph.index_node_labels))])
            im = sknetwork.visualization.svg_graph(adj, pos, directed=True)
            display(SVG(im))

            self.add_trajectory(states, actions, rewards)

            self.comp_gain()
            self.score = score
            self.intrinsic_score = bonus


def train(env, num_iter=100, logs=False):

    agent = ReinforceSeqComp(env)
    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent

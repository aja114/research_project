from collections import defaultdict
import itertools
import numpy as np
from .reinforce_baseline import ReinforceBaseline
from .reinforce import Reinforce


class ReinforceCountSeq(ReinforceBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_freq = defaultdict(int)
        self.intrinsic_reward = 1

    def play_ep(self, num_ep=1, render=False):
        # print("Number of sequences observed: ", len(self.seq_freq))
        for n in range(num_ep):
            state = self.env.reset()
            rewards, actions, states = [], [], []
            score = 0
            intrinsic_score = 0
            t = 0
            done = False
            seq = []
            while not done and t < self.env._max_episode_steps:
                t += 1
                action_idx, action = self.sample_action(state)
                states.append(state)
                actions.append(action_idx)

                state, extrinsic_reward, done, _ = self.env.step(action)

                self.state_freq[state] += 1

                if state not in states:
                    seq += state
                    self.seq_freq[tuple(seq)] += 1
                    intrinsic_reward = self.reward_calc(
                        self.intrinsic_reward, self.seq_freq[tuple(seq)],
                        t, alg='MBIE-EB')
                else:
                    intrinsic_reward = 0

                reward = extrinsic_reward + intrinsic_reward

                rewards.append(reward)
                score += extrinsic_reward
                intrinsic_score += intrinsic_reward

                if render:
                    env.render()

            trajectory = {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards)
            }
            self.trajectories.append(trajectory)
            self.score = score
            self.intrinsic_score = intrinsic_score


def train(env, num_iter=100, logs=False):

    agent = ReinforceCountSeq(env)
    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent

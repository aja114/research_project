from collections import defaultdict
import itertools
import numpy as np
from reinforce import Reinforce


class ReinforceCountSeq(Reinforce):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intrinsic_reward = 1
        self.seq_freq = defaultdict(int)

    def play_ep(self, num_ep=1, render=False):
        for n in range(num_ep):
            state = self.env.reset()
            rewards, actions, states = [], [], []
            score = 0
            t = 0
            done = False

            while not done and t < self.env._max_episode_steps:
                t += 1
                action = self.sample_action(state)
                states.append(state)
                actions.append(action)

                state, extrinsic_reward, done, _ = self.env.step(
                    self.env.actions[action])
                seq = tuple(itertools.chain.from_iterable(states))
                self.seq_freq[seq] += 1
                self.state_freq[state] += 1

                intrinsic_reward = self.reward_calc(
                    self.seq_freq[seq], t, alg='MBIE-EB')

                rewards.append(extrinsic_reward)

                score += extrinsic_reward

                reward = extrinsic_reward + intrinsic_reward

                if render:
                    env.render()

            trajectory = {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards)
            }
            self.trajectories.append(trajectory)
            self.comp_gain()
            self.score = score

    def reward_calc(self, freq, t, alg='UCB'):
        if alg == 'UCB':
            return np.sqrt(2*np.log(t)/freq)
        if alg == 'MBIE-EB':
            return np.sqrt(1/freq)
        if alg == 'BEB':
            return 1 / freq


def train(env, num_iter=100):
    agent = ReinforceCountSeq(env)
    agent.train(num_iter)
    return agent

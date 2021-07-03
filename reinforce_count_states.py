import numpy as np
from reinforce import Reinforce


class ReinforceCountState(Reinforce):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

                self.state_freq[state] += 1

                intrinsic_reward = self.reward_calc(
                    self.state_freq[state], t, alg='UCB')

                reward = intrinsic_reward + extrinsic_reward

                rewards.append(reward)
                score += reward

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
            return self.intrinsic_reward * np.sqrt(2*np.log(t)/freq)
        if alg == 'MBIE-EB':
            return self.intrinsic_reward * np.sqrt(1/freq)
        if alg == 'BEB':
            return self.intrinsic_reward / freq
        if alg == 'BEB-SQ':
            return self.intrinsic_reward / freq**2


def train(env, num_iter=100, logs=False):

    agent = ReinforceCountState(env)
    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent

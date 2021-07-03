import tensorflow as tf
import numpy as np
from NeuralNets import NNtf as NN


class Reinforce:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        i = len(env.states_n)
        out = env.actions_n
        self.network = NN(*args, inp=i, out=out, **kwargs)
        self.gamma = 0.95
        self.intrinsic_reward = 1
        self.state_freq = {x: 0 for x in self.env.states}
        self.trajectories = []

    def sample_action(self, inp):
        scaled_inp = self.scale_state(inp)
        dist = self.network.forward(scaled_inp)
        return int(dist.sample().numpy()[0])

    def get_action(self, inp):
        scaled_inp = self.scale_state(inp)
        action_idx = self.network.predict(scaled_inp)
        return self.env.actions[action_idx]

    def play_ep(self, num_ep=1, render=False):
        for n in range(num_ep):
            state = self.env.reset()
            rewards, actions, states = [], [], []
            score = 0
            t = 0
            done = False

            while not done and t < self.env._max_episode_steps:
                action = self.sample_action(state)
                states.append(state)
                actions.append(action)

                state, reward, done, _ = self.env.step(
                    self.env.actions[action])
                self.state_freq[state] += 1

                rewards.append(reward)
                score += reward

                if render:
                    env.render()

                t += 1

            trajectory = {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards)
            }
            self.trajectories.append(trajectory)
            self.comp_gain()
            self.score = score

    def comp_gain(self):
        for t in self.trajectories:
            if 'gains' not in t:
                r = t['rewards']
                g = np.zeros(r.shape)
                gammas = np.array([self.gamma**i for i in range(len(r))])

                for j in range(0, len(r)):
                    g[j] = np.sum(r[j:] * gammas[:len(r)-j])
                t['gains'] = g

    def update_network(self):
        states = np.concatenate([t["states"] for t in self.trajectories])
        actions = np.concatenate([t["actions"] for t in self.trajectories])
        gains = np.concatenate([t["gains"] for t in self.trajectories])

        # Normalise state in 0-1 range
        states = self.scale_state(states)

        states = tf.cast(tf.convert_to_tensor(states), dtype=tf.float32)
        actions = tf.cast(tf.convert_to_tensor(actions), dtype=tf.float32)
        gains = tf.cast(tf.convert_to_tensor(gains), dtype=tf.float32)

        # print(states)
        # print(actions)
        # print(self.network.forward(states).prob(actions))

        with tf.GradientTape() as tape:
            log_prob = self.network.forward(states).log_prob(actions)
            loss = -tf.math.reduce_mean(log_prob * gains)

        grads = tape.gradient(loss, self.network.weights)

        self.network.update(grads)

        self.trajectories = []

    def train(self, num_iter=100):
        print("\n"+"*"*100)
        print("TRAINING START\n")
        self.scores = np.zeros(num_iter)

        for i in range(num_iter):
            print()
            print("Iteration:", i)
            if i % 50 == 0:
                self.get_proba()

            self.play_ep()

            print("Score:", self.score)
            self.get_state_freq()

            self.update_network()
            self.scores[i] = self.score


        print("\n"+"*"*100)
        print("TRAINING ENDED\n")

    def train_without_logs(self, num_iter=100):
        self.scores = np.zeros(num_iter)
        for i in range(num_iter):
            self.play_ep()
            self.update_network()
            self.scores[i] = self.score

    def get_policy(self):
        self.policy = {x: self.get_action(x) for x in self.env.states}

    def get_proba(self):
        for s in self.env.states:
            scaled_s = self.scale_state(s)
            print(f"state: {s}, proba: {np.round(self.network.model(scaled_s.reshape(1, -1))[0], 2)}")

    def get_state_freq(self):
            f_s = 100*sum([x > 1 for x in self.state_freq.values()]
                          )/len(self.state_freq)
            print(f"% of visited states: {np.round(f_s, 1)}%", )

    def scale_state(self, s):
        return (np.array(s) - np.array(self.env.state_low)) / (np.array(self.env.state_high) - np.array(self.env.state_low))


def train(env, num_iter=100, logs=False):

    agent = Reinforce(env)
    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent

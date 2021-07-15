import tensorflow as tf
import numpy as np
from NeuralNets import NNtf as NN
import tensorflow as tf
from tensorflow import keras

import time

class Baseline():
    def __init__(self, *args, **kwargs):
        self.build_model(*args, **kwargs)

    def build_model(self, inp=4, h1=256, h2=256, out=1, init_weights=None):
        inp = keras.Input(shape=(inp, ))
        x = keras.layers.Dense(
            h1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(inp)
        x = keras.layers.Dense(
            h2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform')(x)
        outp = keras.layers.Dense(
            out, activation='linear', use_bias=True, kernel_initializer='glorot_uniform')(x)

        self.lr = 1e-3
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.model = keras.Model(inputs=inp, outputs=outp)

        self.weights = self.model.trainable_weights

    def predict(self, inp):
        if not isinstance(inp, tf.Tensor):
            inp = tf.convert_to_tensor(np.array(inp).reshape(1, -1))
        return self.model(inp).numpy()

    def update(self, grads):
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))


class ReinforceBaseline:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        i = len(env.states_n)
        out = env.actions_n
        self.network = NN(*args, inp=i, out=out, **kwargs)
        self.baseline = Baseline(*args, inp=i, out=1, **kwargs)

        self.gamma = 0.99
        self.intrinsic_reward = 1
        self.state_freq = {x: 0 for x in self.env.states}
        self.trajectories = []

    def sample_action(self, inp):
        scaled_inp = self.scale_state(inp)
        p = self.network.forward(scaled_inp)
        return int(np.random.choice(len(p), p = p))

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
            probs = self.get_proba()
            while not done and t < self.env._max_episode_steps:
                # action = self.sample_action(state)
                p = probs[state]
                action = int(np.random.choice(len(p), p = p))

                states.append(state)
                actions.append(action)

                state, reward, done, _ = self.env.step(
                    self.env.actions[action])

                # state, reward, done, _ = self.env.step(action)

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
            # print(trajectory['actions'])
            self.comp_gain()
            self.score = score

    def comp_gain(self):
        for t in self.trajectories:
            if 'gains' not in t:
                r = t['rewards']
                g = np.zeros(r.shape)
                g[-1] = r[-1]
                for j in range(len(r)-2, -1, -1):
                    g[j] = r[j] + self.gamma * g[j+1]

                t['gains'] = g

            states = states = tf.cast(
                tf.convert_to_tensor(t['states']), dtype=tf.float32)
            g = g - self.baseline.predict(states)
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

        gammas = [self.gamma**t for t in range(states.shape[0])]
        gammas = tf.cast(tf.convert_to_tensor(gammas), dtype=tf.float32)

        # Update baseline network
        with tf.GradientTape() as tape:
            v_preds = self.baseline.model(states)
            loss = -tf.math.reduce_mean(v_preds * gains * gammas)
        grads = tape.gradient(loss, self.baseline.weights)
        self.baseline.update(grads)

        # Update policy network
        with tf.GradientTape() as tape:
            log_prob = self.network.distributions(states).log_prob(actions)
            loss = -tf.math.reduce_mean(log_prob * gains)
        grads = tape.gradient(loss, self.network.weights)
        self.network.update(grads)

        self.trajectories = []

    def train(self, num_iter=100):

        # self.time_measurement = np.zeros()

        # ss = time.time()

        print("\n"+"*"*100)
        print("TRAINING START\n")
        self.scores = np.zeros(num_iter)

        for i in range(num_iter):
            print()
            print("Iteration:", i)
            if i % 50 == 0:
                p = self.get_proba()
                for s in self.env.states:
                    print(s, "probas: ", np.round(p[s], 2))

            self.play_ep()
            print("Score:", self.score)

            self.get_state_freq()

            self.update_network()
            self.scores[i] = self.score

        print("\n"+"*"*100)
        print("TRAINING ENDED\n")

        # self.time_measurement[0] += time.time() - ss
        #
        # print("TIME MEASURED: ", self.time_measurement)
        # print("TIME MEASURED in %: ", np.round(self.time_measurement * 100 / self.time_measurement[0], 0))

    def train_without_logs(self, num_iter=100):
        self.scores = np.zeros(num_iter)
        for i in range(num_iter):
            self.play_ep()
            self.update_network()
            self.scores[i] = self.score

    def get_policy(self):
        self.policy = {x: self.get_action(x) for x in self.env.states}

    def get_proba(self):
        scaled_s = self.scale_state(np.array(self.env.states))
        probas = self.network.model(scaled_s).numpy()
        p = {s: probas[i] for i, s in enumerate(self.env.states)}
        return p

    def get_state_freq(self):
        f_s = 100*sum([x > 1 for x in self.state_freq.values()]
                      )/len(self.state_freq)
        print(f"% of visited states: {np.round(f_s, 1)}%", )

    def scale_state(self, s):
        return (np.array(s) - np.array(self.env.state_low)) / (np.array(self.env.state_high) - np.array(self.env.state_low))
        return s


def train(env, num_iter=100, logs=False):
    agent = ReinforceBaseline(env)
    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent

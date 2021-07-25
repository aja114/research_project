import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

from .NeuralNets import NNtf as NN
from .reinforce import Reinforce

class Baseline():
    def __init__(self, *args, **kwargs):
        self.build_model(*args, **kwargs)

    def build_model(self, inp=4, h1=256, h2=256, out=1, init_weights=None):
        self.inp = inp
        self.out = out

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
            inp = tf.convert_to_tensor(np.array(inp).reshape(-1, self.inp))
        return self.model(inp).numpy()

    def update(self, grads):
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))


class ReinforceBaseline(Reinforce):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.baseline = Baseline(*args, inp=self.state_space, out=1, **kwargs)

    def comp_gain(self):
        for t in self.trajectories:
            if 'gains' not in t:
                r = t['rewards']
                g = np.zeros(r.shape)
                g[-1] = r[-1]
                for j in range(len(r)-2, -1, -1):
                    g[j] = r[j] + self.gamma * g[j+1]

                t['gains'] = g

                states = tf.cast(
                    tf.convert_to_tensor(t['states']), dtype=tf.float32)
                g = g - self.baseline.predict(states)
                t['gains'] = g

    def update_agent(self):
        self.comp_gain()

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
        self.update_baseline_net(states, gains, gammas)

        # Update the policy network
        self.update_policy_net(states, actions, gains)

        self.trajectories = []

    @tf.function(experimental_relax_shapes=True)
    def update_baseline_net(self, states, gains, gammas):
        with tf.GradientTape() as tape:
            v_preds = self.baseline.model(states)
            loss = -tf.math.reduce_mean(v_preds * gains * gammas)
        grads = tape.gradient(loss, self.baseline.weights)
        self.baseline.optimizer.apply_gradients(
            zip(grads, self.baseline.model.trainable_weights))

    @tf.function(experimental_relax_shapes=True)
    def update_policy_net(self, states, actions, gains):
        with tf.GradientTape() as tape:
            log_prob = self.policy.distributions(states).log_prob(actions)
            loss = -tf.math.reduce_mean(log_prob * gains)
        grads = tape.gradient(loss, self.policy.weights)
        self.policy.optimizer.apply_gradients(
            zip(grads, self.policy.model.trainable_weights))


def train(env, num_iter=100, logs=False):
    agent = ReinforceBaseline(env)
    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent

from RLAlgorithm import RLAlgorithm
from Utils.Noise import OUNoise
from Utils.Buffer import Buffer
from keras import layers
from keras.models import load_model as ldm
from keras.models import save_model as sdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


class DDPG(RLAlgorithm):
    def __init__(self, env, total_episodes=100, tau=5e-3, actor_lr=2e-3, critic_lr=1e-3, batch_size=32, max_len_buffer=1e4, gamma=0.99, epsilon=0.9, decay=0.99, min_epsilon=0.1, std_noise=0.1):
        super().__init__(env, total_episodes, actor_lr, gamma, epsilon, decay, min_epsilon)
        self.algorithm_name = "DDPG"
        self.tau = tau
        self.actor_lr = actor_lr
        self.critical_lr = critic_lr
        self.ou_noise = OUNoise(mean=np.zeros(
            1), std_deviation=float(std_noise) * np.ones(1))
        self.batch_size = batch_size
        self.memory = Buffer(
            max_len_buffer, self.env.num_states, self.env.num_actions, self.batch_size)
        self.__build_models()

    def __build_models(self):
        self.__actor_model = self.__get_actor()
        self.__critic_model = self.__get_critic()
        self.__target_actor = self.__get_actor()
        self.__target_critic = self.__get_critic()
        self.__target_actor.set_weights(self.__actor_model.get_weights())
        self.__target_critic.set_weights(self.__critic_model.get_weights())
        self.__critic_optimizer = tf.keras.optimizers.Adam(self.critical_lr)
        self.__actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def __get_actor(self):
        init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        inputs = layers.Input(shape=(self.num_states,))
        d = layers.Dense(128, activation="relu")(inputs)
        d = layers.Dense(256, activation="relu")(d)
        d = layers.Dense(384, activation="relu")(d)
        outputs = layers.Dense(1, activation="tanh",
                               kernel_initializer=init)(d)

        outputs = outputs * self.env.upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def __get_critic(self):
        state_input = layers.Input(shape=(self.env.num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        d = layers.Dense(128, activation="relu")(concat)
        d = layers.Dense(256, activation="relu")(d)
        d = layers.Dense(384, activation="relu")(d)
        outputs = layers.Dense(1)(d)

        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    @tf.function
    def replay(self):
        state_batch, action_batch, reward_batch, next_state_batch = self.memory.sample()
        with tf.GradientTape() as tape:
            target_actions = self.__target_actor(
                next_state_batch, training=True)
            y = reward_batch + self.gamma * self.__target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.__critic_model(
                [state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            critic_grad = tape.gradient(
                critic_loss, self.__critic_model.trainable_variables)
            self.__critic_optimizer.apply_gradients(
                zip(critic_grad, self.__critic_model.trainable_variables)
            )

        with tf.GradientTape() as tape:
            actions = self.__actor_model(state_batch, training=True)
            critic_value = self.__critic_model(
                [state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(
            actor_loss, self.__actor_model.trainable_variables)
        self.__actor_optimizer.apply_gradients(
            zip(actor_grad, self.__actor_model.trainable_variables)
        )

    def policy(self, state, *args, **kwargs):
        sampled_actions = tf.squeeze(self.__actor_model(state))
        noise = self.ou_noise()
        sampled_actions = sampled_actions.numpy() + noise
        legal_action = np.clip(
            sampled_actions, self.env.lower_bound, self.env.upper_bound)
        return [np.squeeze(legal_action)]

    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(self.tau * b + (1 - self.tau) * a)

    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    def save(self, path=".", file_name=None, *args):
        if os.path.isdir(path):
            if not path.endswith("rl_models"):
                models_path = os.path.join(path, "rl_models")
                if not os.path.isdir(models_path):
                    os.mkdir(models_path)
            else:
                models_path = path
            if file_name:
                self.model.save_weights(os.path.join(
                    models_path, file_name), *args)
            else:
                self.model.save_weights(
                    os.path.join(models_path, "DDPG"), *args)
        self.model.save_weights(path, *args)

    def save_model(self, path=".", file_name=None, *args):
        if os.path.isdir(path):
            if not path.endswith("rl_models"):
                models_path = os.path.join(path, "rl_models")
                if not os.path.isdir(models_path):
                    os.mkdir(models_path)
            else:
                models_path = path
            if file_name:
                sdm(self.model, os.path.join(models_path, file_name), *args)
            else:
                sdm(self.model, os.path.join(models_path, "DDPG.h5"), *args)
        sdm(self.model, path, *args)

    def load(self, path, *args):
        self.model.load_weights(path, *args)

    def load_model(self, path, *args):
        self.model = ldm(path, *args)

    def value_function(self, state, *args, **kwargs):
        pass

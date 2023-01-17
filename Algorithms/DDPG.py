from RLAlgorithm import RLAlgorithm
from Utils.Noise import OUNoise
from Utils.Buffer import Buffer
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt


class DDPG(RLAlgorithm):
    def __init__(self, env, total_episodes=100, tau=5e-3, actor_lr=2e-3, critic_lr=1e-3 , batch_size=32, max_len_buffer=1e4, gamma=0.99, epsilon=0.9, decay=0.99, min_epsilon=0.1, std_noise=0.1):
        super().__init__(env, total_episodes, actor_lr, gamma, epsilon, decay, min_epsilon)
        self.algorithm_name = "DDPG"
        self.tau = tau
        self.actor_lr = actor_lr
        self.critical_lr = critic_lr
        self.ou_noise = OUNoise(mean=np.zeros(
            1), std_deviation=float(std_noise) * np.ones(1))
        self.batch_size = batch_size
        self.memory = Buffer(max_len_buffer)
        self.__build_models()

    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    def save(self, path=".", file_name=None):
        return None

    def load(self, path):
        return None

    def value_function(self, state, *args, **kwargs):
        return None

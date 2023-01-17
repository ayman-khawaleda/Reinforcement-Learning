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

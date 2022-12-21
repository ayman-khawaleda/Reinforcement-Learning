from .RLAlgorithm import RLAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class DQN(RLAlgorithm):
    def __init__(self, env, total_episodes=10000, lr=0.5, gamma=0.9, epsilon=0.9, decay=0.99, min_epsilon=0.1):
        super().__init__(env, total_episodes, lr, gamma, epsilon, decay, min_epsilon)
        self.algorithm_name = "DQN"

    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    def policy(self, *args, **kwargs):
        return super().policy(*args, **kwargs)

    def value_function(self, state: int, *args, **kwargs):
        return super().value_function(state, *args, **kwargs)

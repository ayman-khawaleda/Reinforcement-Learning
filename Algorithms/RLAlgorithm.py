from abc import abstractmethod, ABC
import numpy as np
from Environments.Environment import Environment

class RLAlgorithm(ABC):
    def __init__(self, env: Environment, total_episodes=10000, lr=0.5, gamma=0.9, epsilon=0.9, decay=0.99, min_epsilon=0.1):
        self.env = env
        self.total_episodes = total_episodes
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.algorithm_name = ""

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def policy(self, *args, **kwargs):
        pass

    @abstractmethod
    def value_function(self, state: int, *args, **kwargs):
        pass



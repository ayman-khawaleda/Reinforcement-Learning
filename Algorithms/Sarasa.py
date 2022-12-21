from .Qlearning import Qlearning
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class SARSA(Qlearning):
    def __init__(self, env, total_episodes=500, lr=0.5, max_steps=25, gamma=0.9, epsilon=0.9, decay=0.99, min_epsilon=0):
        super().__init__(env, total_episodes, lr,
                         max_steps, gamma, epsilon, decay, min_epsilon)
        self.algorithm_name = "SARSA"

    def Q_next(self, state: int):
        action_idx = self.policy(state)
        return self.q_table[state, action_idx]

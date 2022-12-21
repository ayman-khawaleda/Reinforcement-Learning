from Agents.Agent import Agent
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


class Environment(ABC):
    def __init__(self, agent: Agent, reward_values: list) -> None:
        self.agent = agent
        self.reward_values = reward_values

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def next_state(self):
        pass



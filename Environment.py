from Agent import Agent,GridAgent
from abc import ABC,abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

class Environment:
    def __init__(self, agent: Agent, reward_values:list) -> None:
        self.agent = agent
        self.reward_values = reward_values
    
    @abstractmethod
    def reward(self):
        pass
    
    @abstractmethod
    def get_next_state(self):
        pass
    
    @abstractmethod
    def next_state(self):
        pass
    

class GridEnvironment(Environment):

    def __init__(self, agent: GridAgent, reward_values=[10, -10, -1], rows=5, cols=5, win_state=(3, 3), start_state=(0, 0), holes=[(1, 0), (1, 3), (3, 1), (3, 2)]) -> None:
        super().__init__(agent,reward_values)
        self.rows = rows
        self.cols = cols
        self.win_state = win_state
        self.start = start_state
        self.holes = holes
        self.size = rows * cols
        self.shape = (rows, cols)

    def reward(self):
        if self.agent.pos == self.win_state:
            return self.reward_values[0]
        elif self.agent.pos in self.holes:
            return self.reward_values[1]
        return self.reward_values[2]

    def is_win(self):
        return self.agent.pos == self.win_state

    def is_lose(self):
        return self.agent.pos in self.holes

    def get_state_index(self):
        return self.cols * self.agent.pos[0] + self.agent.pos[1]

    def next_state(self, action):
        if action == "up":
            nxtState = (self.agent.pos[0] - 1, self.agent.pos[1])
        elif action == "down":
            nxtState = (self.agent.pos[0] + 1, self.agent.pos[1])
        elif action == "left":
            nxtState = (self.agent.pos[0], self.agent.pos[1] - 1)
        else:
            nxtState = (self.agent.pos[0], self.agent.pos[1] + 1)
        if nxtState[0] >= 0 and nxtState[0] <= self.rows-1 and nxtState[1] >= 0 and nxtState[1] <= self.cols-1:
            return nxtState
        return self.agent.pos


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

    def __init__(self, agent: GridAgent, reward_values=[10, -10, -1], rows=4, cols=4, win_state=(3, 3), start_state=(0, 0), holes=[(1, 0), (1, 3), (3, 1), (3, 2)]) -> None:
        super().__init__(agent,reward_values)
        self.rows = rows
        self.cols = cols
        self.win_state = win_state
        self.start = start_state
        self.holes = holes
        self.size = rows * cols
        self.shape = (rows, cols)
        self.grid = np.zeros(self.shape,dtype='float32')

    def reward(self):
        if self.agent.pos == self.win_state:
            return self.reward_values[0]
        elif self.agent.pos in self.holes:
            return self.reward_values[1]
        return self.reward_values[2]

    def is_agent_win(self):
        return self.agent.pos[0] == self.win_state[0] and self.agent.pos[1] == self.win_state[1]

    def is_agent_lose(self):
        for hole in self.holes:
            if self.agent.pos[0] == hole[0] and self.agent.pos[1] == hole[1]:
                return True
        return False

    
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
    
    def visit(self, cell):
        self.grid[cell[0], cell[1]] += 1

    def print_path_as_heatmap(self, value_function, iters):
        data = np.ones(self.shape) * 150
        for hole in self.holes:
            data[hole[0], hole[1]] = 255
        self.agent.pos = self.start
        i=0
        while i<self.size:
            i+=1
            agent_pos = self.agent.pos
            data[agent_pos[0], agent_pos[1]] = 50
            if self.is_agent_win():
                break
            old_state = self.get_state_index()
            action_idx = value_function(old_state)
            self.agent.pos = self.next_state(self.agent.actions[action_idx])
        # The Environment
        # hm = sn.heatmap(data=data, linewidths=1,
        #                 linecolor="black", cmap='Blues', cbar=False)
        
        # What The Most Cells The Agent Visited
        self.grid[self.start[0], self.start[1]] = iters
        
        hm = sn.heatmap(data=self.grid/iters, linewidths=0,
                        linecolor="black", cmap='Blues', cbar=False, annot=True)
        plt.show()

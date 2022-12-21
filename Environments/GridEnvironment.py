from .Environment import Environment
from Agents.GridAgent import GridAgent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from config import set_numpy_seed
class GridEnvironment(Environment):

    def __init__(self, agent: GridAgent, reward_values=[10, -10, -1], rows=4, cols=4, win_state=(3, 3), start_state=(0, 0), holes=[(1, 0), (1, 3), (3, 1), (3, 2)]) -> None:
        super().__init__(agent, reward_values)
        self.rows = rows
        self.cols = cols
        self.win_state = win_state
        self.start = start_state
        self.holes = holes
        self.size = rows * cols
        self.shape = (rows, cols)
        self.grid = np.zeros(self.shape, dtype='float32')

    def random_holes(self, number=5):
        self.holes = []
        set_numpy_seed(seed=None)
        obstacles_rows = np.random.randint(0, self.rows, number)
        obstacles_cols = np.random.randint(0, self.cols, number)
        for obstacle in zip(obstacles_rows, obstacles_cols):
            if obstacle in [self.win_state, self.start]:
                continue
            self.holes.append(obstacle)
        set_numpy_seed(seed=15)

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

    def plot_env(self):
        plt.figure(figsize=(6, 4))
        data = np.ones(self.shape) * 150
        for hole in self.holes:
            data[hole[0], hole[1]] = 255
        data[self.start[0], self.start[1]] = 0
        data[self.win_state[0], self.win_state[1]] = 0
        # Plot The Environment
        hm = sn.heatmap(data=data, linewidths=2,
                        linecolor="black", cmap='Blues', cbar=False)

    def plot_path_as_heatmap(self, value_function, iters, title, show_values=True):
        plt.figure(figsize=(6, 4))
        self.agent.pos = self.start
        while range(self.size):
            agent_pos = self.agent.pos
            if self.is_agent_win():
                break
            old_state = self.get_state_index()
            action_idx = value_function(old_state)
            self.agent.pos = self.next_state(self.agent.actions[action_idx])

        # What The Most Cells The Agent Visited
        self.grid[self.start[0], self.start[1]] = iters/16

        hm = sn.heatmap(data=self.grid/iters, linewidths=0,
                        linecolor="black", cmap='Blues', cbar=False, annot=show_values)
        plt.title(title)

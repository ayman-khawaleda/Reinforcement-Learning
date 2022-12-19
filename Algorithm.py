from abc import abstractmethod, ABC
import numpy as np
from Environment import GridEnvironment,Environment
from tqdm import tqdm
import matplotlib.pyplot as plt


class RLAlgorithm(ABC):
    def __init__(self, env: Environment, total_episodes=10000, lr=0.5, gamma=0.9, epsilon=0.9, decay=0.99, min_epsilon=0.1):
        self.env = env
        self.total_episodes = total_episodes
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def policy(self, *args, **kwargs):
        pass

    @abstractmethod
    def value_function(self,state):
        pass


class Qlearning(RLAlgorithm):
    def __init__(self, env, total_episodes=100, lr=0.5, max_steps=25, gamma=0.9, epsilon=0.9, decay=0.99, min_epsilon=0):
        super().__init__(env, total_episodes, lr, gamma, epsilon, decay, min_epsilon)
        self.q_table = np.zeros((self.env.size, self.env.agent.n_actions))
        self.rewards = []
        self.max_steps = max_steps

    def print_q_table(self):
        print("------------------- Q LEARNING TABLE ------------------\n",
              self.q_table, "\n-------------------------------------------------------")

    def policy(self, state_idx, *args, **kwargs):
        rand = np.random.uniform(0, 1)
        if rand > self.epsilon:
            return self.value_function(state_idx)
        else:
            return np.random.randint(0, self.env.agent.n_actions)

    def fit(self):
        for episodes in tqdm(range(self.total_episodes)):
            self.env.agent.pos = self.env.start
            total_reward = 0
            for _ in range(self.max_steps):
                old_state_idx = self.env.get_state_index()
                action_idx = self.policy(old_state_idx)

                # Update The Agent Position And Get The Current State
                self.env.agent.pos = self.env.next_state(
                    self.env.agent.actions[action_idx])
                new_state_idx = self.env.get_state_index()

                reward = self.env.reward()
                total_reward += reward

                # Qt = Qt + alpha * ( (reward + gamma * max(newQ)) - Qt )
                self.q_table[old_state_idx, action_idx] += self.lr * (reward + self.gamma * np.max(
                    self.q_table[new_state_idx, :]) - self.q_table[old_state_idx, action_idx])

                # End The Episode If You Win Or Lose
                if self.env.is_win() and self.env.is_lose():
                    break

            # Trade off Exploration & Exploitation
            self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
            self.rewards.append(total_reward)

    def value_function(self, state:int):
        return np.argmax(self.q_table[state, :])

    def plot_reward(self):
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(self.total_episodes), self.rewards, color='blue')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward per Epidode')
        plt.show()
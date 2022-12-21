from .RLAlgorithm import RLAlgorithm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Qlearning(RLAlgorithm):
    def __init__(self, env, total_episodes=500, lr=0.5, max_steps=25, gamma=0.9, epsilon=0.9, decay=0.99, min_epsilon=0):
        super().__init__(env, total_episodes, lr, gamma, epsilon, decay, min_epsilon)
        self.q_table = np.zeros((self.env.size, self.env.agent.n_actions))
        self.rewards = []
        self.max_steps = max_steps
        self.algorithm_name = "Q-learning"

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
        self.total_iters = 0
        for episodes in tqdm(range(self.total_episodes)):
            self.env.agent.pos = self.env.start
            total_reward = 0
            for _ in range(self.max_steps) or self.env.is_agent_done():
                old_state_idx = self.env.get_state_index()
                action_idx = self.policy(old_state_idx)

                # Update The Agent Position And Get The Current State
                self.env.agent.pos = self.env.next_state(
                    self.env.agent.actions[action_idx])
                new_state_idx = self.env.get_state_index()

                reward = self.env.reward()
                total_reward += reward

                # Qt = Qt + alpha * ( (reward + gamma * max(newQ)) - Qt )
                self.q_table[old_state_idx, action_idx] += self.lr * (reward + self.gamma * self.Q_next(
                    new_state_idx) - self.q_table[old_state_idx, action_idx])

                self.env.visit(self.env.agent.pos)
                self.total_iters += 1
            # Trade off Exploration & Exploitation
            self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
            self.rewards.append(total_reward)

    def Q_next(self, state: int):
        return np.max(self.q_table[state, :])

    def value_function(self, state: int):
        return np.argmax(self.q_table[state, :])

    def plot_reward(self, step=5, color="blue"):
        plt.figure(figsize=(6, 4))
        plt.title(self.algorithm_name)
        # Plot The Pair Of (Reward, Episodes) Each Step
        plt.plot(np.arange(0, self.total_episodes, step), [
                 x for i, x in enumerate(self.rewards) if i % step == 0], color=color)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward per Epidode')

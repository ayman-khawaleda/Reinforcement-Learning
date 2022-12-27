from cgitb import reset
from .RLAlgorithm import RLAlgorithm
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam


class DQN(RLAlgorithm):
    def __init__(self, env, epochs=100, max_steps=99, lr=0.5, gamma=0.9, epsilon=0.9, decay=0.99, min_epsilon=0.1, batch_size=32):
        super().__init__(env, epochs, lr, gamma, epsilon, decay, min_epsilon)
        self.algorithm_name = "DQN"
        self.max_steps = max_steps
        self.rewards = []
        self.EPOCHS = epochs
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)
        self.model = Sequential()
        self.model.add(Input(self.env.size))
        self.model.add(Dense(24, activation="tanh", name="States"))
        self.model.add(Dense(48, activation="tanh"))
        self.model.add(Dense(self.env.agent.n_actions,
                       activation='linear', name="Actions"))
        self.model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.lr, decay=0.01))
        self.model.summary()

    def remember(self, state_idx, action_idx, reward, next_state_idx, done):
        self.memory.append(
            (state_idx, action_idx, reward, next_state_idx, done))

    def replay(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample(
            self.memory, min(len(self.memory), self.batch_size))
        for state_idx, action_idx, reward, next_state_idx, done in mini_batch:
            y_target = self.model.predict(state_idx)

            y_target[0][action_idx] = reward if done else reward + \
                self.gamma * self.Q_next(next_state_idx)
            x_batch.append(state_idx)
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch, y_batch),
                       batch_size=len(x_batch), verbose=0)
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

    def Q_next(self, state):
        return np.max(self.model.predict(state)[0])

    def fit(self, *args, **kwargs):
        self.total_iters = 0
        total_reward = 0
        for _ in tqdm(range(self.EPOCHS)):
            state_arr = np.zeros((1, self.env.size))
            state_idx = self.env.reset()
            state_arr[0, state_idx] = 1
            done = self.env.is_agent_done()
            for _ in range(self.max_steps):
                action_idx = self.policy(state_idx)
                self.env.agent.pos = self.env.next_state(
                    self.env.agent.actions[action_idx])
                new_state_idx = self.env.get_state_index()
                reward = self.env.reward()
                done = self.env.is_agent_done()

                if done:
                    break

                new_state_arr = np.zeros((1, self.env.size))
                new_state_arr[0, new_state_idx] = 1

                self.remember(state_idx, action_idx,
                              reward, new_state_idx, done)
                state_idx = new_state_idx
                total_reward += reward
                self.total_iters += 1

        self.rewards.append(total_reward)

        if len(self.memory) > self.batch_size:
            self.replay()

    def policy(self, state_idx, *args, **kwargs):
        rand = np.random.uniform(0, 1)
        if rand > self.epsilon:
            return self.value_function(state_idx)
        else:
            return np.random.randint(0, self.env.agent.n_actions)

    def value_function(self, state: int, *args, **kwargs):
        return np.argmax(self.model.predict(state))

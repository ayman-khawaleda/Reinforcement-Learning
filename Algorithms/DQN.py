from cgitb import reset

from pandas import array
from .RLAlgorithm import RLAlgorithm
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from keras.models import load_model as __ldm
from keras.models import save_model as __sdm
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam


class DQN(RLAlgorithm):
    def __init__(self, env, epochs=100, max_steps=99,neurons_num:list = [12,24], lr=0.5, gamma=0.99, epsilon=0.99, decay=0.999, min_epsilon=0.01, batch_size=32,max_len_queue=1000):
        super().__init__(env, epochs, lr, gamma, epsilon, decay, min_epsilon)
        self.algorithm_name = "DQN"
        self.max_steps = max_steps
        self.rewards = []
        self.EPOCHS = epochs
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_len_queue)
        self.neurons_num_list = neurons_num
        self.model = self.create_model()
        self.model.summary()

    def create_model(self):
        model = Sequential()
        model.add(Input(self.env.size))
        for n in self.neurons_num_list:
            model.add(Dense(n, activation="relu"))

        model.add(Dense(self.env.agent.n_actions,
                       activation='linear', name="Actions"))
        model.compile(loss='mse', optimizer=Adam(
            learning_rate=self.lr, decay=0.01))
        return model
    
    def remember(self, state_idx, action_idx, reward, next_state_idx, done):
        self.memory.append(
            (state_idx, action_idx, reward, next_state_idx, done))

    def replay(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample(
            self.memory, min(len(self.memory), self.batch_size))
        for state_idx, action_idx, reward, next_state_idx, done in mini_batch:
            y_target = self.pred(state_idx)

            y_target[0][action_idx] = reward if done else reward + \
                self.gamma * self.Q_next(next_state_idx)
            x_batch.append(self.state_as_one_hot_encoding(state_idx).squeeze())
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch),np.array(y_batch),
                    batch_size=len(x_batch), verbose=0)
        
        self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)

    def Q_next(self, state):
        return np.max(self.pred(state)[0])

    def fit(self, *args, **kwargs):
        self.total_iters = 0
        total_reward = 0
        for e in tqdm(range(self.EPOCHS)):
            state_idx = self.env.reset()
            done = self.env.is_agent_done()
            for step in range(self.max_steps):
                action_idx = self.policy(state_idx)
                self.env.agent.pos = self.env.next_state(
                    self.env.agent.actions[action_idx])
                new_state_idx = self.env.get_state_index()
                reward = self.env.reward()
                done = self.env.is_agent_done()

                self.remember(state_idx, action_idx,
                              reward, new_state_idx, done)
                state_idx = new_state_idx
                
                total_reward += reward
                self.total_iters += 1
                if done:
                    break

            self.rewards.append(total_reward)
            if len(self.memory) > self.batch_size:
                self.replay()

    def policy(self, state_idx, *args, **kwargs):
        rand = np.random.uniform(0, 1)
        if rand > self.epsilon:
            return self.value_function(state_idx)
        else:
            return np.random.randint(0, self.env.agent.n_actions)

    def state_as_one_hot_encoding(self, idx):
        arr = np.zeros((1, self.env.size))
        arr[0, idx] = 1
        return arr

    def pred(self, state):
        return self.model.predict(self.state_as_one_hot_encoding(state), verbose=0)

    def value_function(self, state: int, *args, **kwargs):
        return np.argmax(self.pred(state))

    def save(self, path=".",file_name=None, *args):
        if os.path.isdir(path):
            if not path.endswith("rl_models"):
                models_path = os.path.join(path, "rl_models")
                if not os.path.isdir(models_path):
                    os.mkdir(models_path)
            else:
                models_path = path
            if file_name:
                self.model.save_weights(os.path.join(models_path,file_name),*args)
            else:
                self.model.save_weights(os.path.join(models_path,"DQN"),*args)
        self.model.save_weights(path,*args)
    
    
    def save_model(self,path=".", file_name=None, *args):
        if os.path.isdir(path):
            if not path.endswith("rl_models"):
                models_path = os.path.join(path,"rl_models")
                if not os.path.isdir(models_path):
                    os.mkdir(models_path)
            else:
                models_path = path
            if file_name:
                __sdm(self.model, os.path.join(models_path,file_name),*args)
            else:
                __sdm(self.model, os.path.join(models_path,"DQN.h5"),*args)
        __sdm(self.model, path, *args)
    
    def load(self, path, *args):
        self.model.load_weights(path,*args)
    
    def load_model(self, path, *args):
        self.model = __ldm(path, *args)
from Environment import Environment
import gym 

class PendulumEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__(None, None)
        self.problem = "Pendulum-v1"
        self.env = gym.make(self.problem)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.upper_bound = self.env.action_space.high[0]
        self.lower_bound = self.env.action_space.low[0]
        self.done=False
    
    def reward(self):
        return self.reward
    
    def next_state(self,action):
        self.state, self.reward, self.done, self.info = self.env.step(action)
        return self.state
    
    def is_done(self):
        self.done

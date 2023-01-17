from .Environment import Environment
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
        return self.reward_
    
    def next_state(self,action):
        self.state, self.reward_, self.done, self.info = self.env.step(action)
        return self.state, self.reward_
    
    def is_done(self):
        return self.done
    
    def reset(self):
        return self.env.reset()

    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.render()
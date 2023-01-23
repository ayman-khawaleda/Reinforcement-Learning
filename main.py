from Algorithms import Sarasa, Qlearning, DQN, DDPG
from Environments.Continuous import PendulumEnvironment
from Agents.GridAgent import GridAgent
from config import set_numpy_seed
import matplotlib.pyplot as plt

if __name__ == "__main__":

    env = PendulumEnvironment()
    ddpg = DDPG.DDPG(env)
    ddpg.fit(render=False)
    ddpg.plot_reward()
    plt.show()
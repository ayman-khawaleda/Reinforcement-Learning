from Algorithms import Sarasa, Qlearning, DQN
from Environments.GridEnvironment import GridEnvironment
from Agents.GridAgent import GridAgent
from config import set_numpy_seed
import matplotlib.pyplot as plt

if __name__ == "__main__":
    set_numpy_seed(seed=15)
    agent = GridAgent()
    env = GridEnvironment(agent,reward_values=[200, -10, -1], rows=4, cols=4, win_state=(3, 3))
    env.random_holes(3)
    env.plot_env()

    dqn = DQN.DQN(env,epochs=50,neurons_num=[12,36],decay=0.97, batch_size=8,lr=0.1)
    dqn.fit()
    dqn.plot_reward(step=1)
    env.plot_path_as_heatmap(dqn.value_function,dqn.total_iters,"DQN")
    plt.show()

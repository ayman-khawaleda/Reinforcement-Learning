from Algorithm import SARSA, Qlearning
from Environment import GridEnvironment
from Agent import GridAgent
import matplotlib.pyplot as plt
if __name__ == "__main__":
    agent = GridAgent()
    env = GridEnvironment(agent)
    q_learning = Qlearning(env, total_episodes=500)
    q_learning.fit()
    env.print_path_as_heatmap(q_learning.value_function,q_learning.total_iters)
    # q_learning.print_q_table()
    sarsa = SARSA(env,total_episodes=500)
    sarsa.fit()
    # sarsa.print_q_table()
    # q_learning.plot_reward()
    # sarsa.plot_reward()
    plt.show()
    

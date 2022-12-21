from Algorithms import Sarasa, Qlearning
from Environments.GridEnvironment import GridEnvironment
from Agents.GridAgent import GridAgent

import matplotlib.pyplot as plt
if __name__ == "__main__":
    agent = GridAgent()
    env = GridEnvironment(agent,rows=6,cols=6,win_state=(5,5))
    env.random_holes(10)
    env.plot_env()
    
    q_learning = Qlearning.Qlearning(env, total_episodes=750,max_steps=99)
    q_learning.fit()
    sarsa = Sarasa.SARSA(env,total_episodes=750,max_steps=99)
    sarsa.fit()
    
    # q_learning.print_q_table()
    # sarsa.print_q_table()
    q_learning.plot_reward()
    sarsa.plot_reward()
    
    env.plot_path_as_heatmap(q_learning.value_function,
                             q_learning.total_iters, "Q-learning", show_values=False)
    env.plot_path_as_heatmap(sarsa.value_function,
                             sarsa.total_iters, "Sarsa", show_values=False)
    
    plt.show()

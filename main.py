from Algorithm import Qlearning
from Environment import GridEnvironment
from Agent import GridAgent

if __name__ == "__main__":
    agent = GridAgent()
    env = GridEnvironment(agent)
    q_learning = Qlearning(env, total_episodes=500, max_steps=25)
    print("Start Learning")
    q_learning.fit()
    q_learning.print_q_table()
    q_learning.plot_reward()
    env.print_path_as_heatmap(q_learning.value_function )

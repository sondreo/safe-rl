# import gym
import numpy as np
from time import sleep

from agent import Agent, QLearningAgent
from environment import (
    GYM_ENVS, create_environment,
    print_environment_info, print_all_available_environments
)
from utils import discretize_states


np.random.seed(1)  # Set random seed for experimental reproducibility


EPISODES = 500


def run():
    """Runs RL algorithm on selected agent in selected environment."""
    print('Running...\n')

    # print_all_available_environments()

    # Create and render gym environment
    env = create_environment(GYM_ENVS['classical_control'][1])
    env.render(mode='human')

    print_environment_info(env)

    # Create agent
    # agent = Agent(env.action_space)
    agent = QLearningAgent(env.action_space, env.observation_space)

    print(f'Q-table size: {agent.Q.shape}')

    for episode in range(EPISODES):
        env.render(mode='human')
        print(f'Episode {episode + 1}')

        # Initialize observed state and reward signal
        observation = env.reset()
        reward = 0

        done = False

        while not done:
            action = agent.act(observation)
            new_observation, reward, done, _ = env.step(action)
            agent.update_Q_table(observation, new_observation, action, reward)

            observation = new_observation

        # action = agent.act(observation, reward, done)
        # observation, reward, done, _ = env.step(action)

        # Adjust render rate
        sleep(1. / 20.)

    env.close()


if __name__ == "__main__":
    run()

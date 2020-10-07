# import gym
from time import sleep

from agent import Agent, QLearningAgent
from environment import GYM_ENVS, create_environment, print_environment_info


EPISODES = 500


def run():
    """Runs RL algorithm on selected agent in selected environment."""
    print('Running...\n')

    # Create and render gym environment
    env = create_environment(GYM_ENVS['classical_control'][1])
    env.render(mode='human')

    print_environment_info(env)

    # Create agent
    # agent = Agent(env.action_space)
    agent = QLearningAgent(env.action_space, 0.1)

    # Initialize observed state and reward signal
    observation = env.reset()
    reward = 0

    done = False

    for _ in range(EPISODES):
        env.render(mode='human')

        action = agent.act(observation, reward, done)
        observation, reward, done, _ = env.step(action)

        # Adjust render rate
        sleep(1. / 60.)

    env.close()


if __name__ == "__main__":
    run()

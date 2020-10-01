import gym
import gym_panda    # Import registers environment in Gym
from time import sleep

from agent import Agent

GYM_ENVS = {
    'classical_control': [
        'CartPole-v1',
        'MountainCar-v0',
        'Pendulum-v0'
    ],
    'panda': [
        'panda-v0'
    ]
}


EPISODES = 500


def create_environment(gym_env):
    env = gym.make(gym_env)

    # Fix for certain OpenAI Gym environments,
    # requiring to be reset prior to initial rendering
    if gym_env in GYM_ENVS['classical_control']:
        env.reset()

    return env


def run():
    # Create and render gym environment
    env = create_environment(GYM_ENVS['classical_control'][0])
    env.render()

    # Create agent
    agent = Agent(env.action_space)

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
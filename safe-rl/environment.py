import gym
import gym_panda  # Import registers environment in Gym


GYM_ENVS = {
    'classical_control': [
        'CartPole-v1',
        'MountainCar-v0',
        'MountainCarContinuous-v0',
        'Pendulum-v0'
    ],
    'panda': [
        'panda-v0'
    ]
}


def create_environment(env_name):
    env = gym.make(env_name)

    # Fix for certain OpenAI Gym environments,
    # requiring to be reset prior to initial rendering
    if env_name in GYM_ENVS['classical_control']:
        env.reset()

    return env


def print_environment_info(env):
    """Shows environment information"""

    print(f'Running environment: {env.unwrapped.spec.id}')
    print(f'State space format: {env.observation_space}')
    print(f'Action space format: {env.action_space}')

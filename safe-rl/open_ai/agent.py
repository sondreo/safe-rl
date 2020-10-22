import numpy as np

from utils import discretize_states


class Agent():
    """Agent base class."""

    def __init__(self, action_space, observation_space):
        """Initializes agent class."""

        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation, reward, done):
        """Perform an action according to a random policy."""

        return self.action_space.sample()


class QLearningAgent(Agent):
    """Q-learning agent class."""

    def __init__(self, action_space, observation_space, epsilon=0.8, alpha=0.2, gamma=0.9):
        """Initializes Q-learning agent class."""

        super().__init__(action_space, observation_space)
        self.observation_space = observation_space

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.scales = np.array([1, 10])

        print(f'Obs low: {self.observation_space.low}')

        # Discretize state space
        # state_space = discretize_states(
        #     self.observation_space.high, self.observation_space,
        #     np.array([1, 10]))
        state_space = (self.observation_space.high - self.observation_space.low) * self.scales
        num_states = np.round(state_space, 0).astype(int) + 1

        print(f'Num states: {num_states}')

        self.Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], self.action_space.n))

    def _discretize(self, observation):
        """Discretize states."""
        observation = (observation - self.observation_space.low) * self.scales
        return np.round(observation, 0).astype(int)

    def act(self, observation, reward=0, done=False):
        """Performs an action according to an epsilon-greedy policy."""

        observation = self._discretize(observation)

        if np.random.random() < 1 - self.epsilon:
            return np.argmax(self.Q[observation[0], observation[1]])
        else:
            return np.random.randint(0, self.action_space.n)

    def update_Q_table(self, observation, new_observation, action, reward):
        """Updates element in Q-table."""

        observation = self._discretize(observation)
        new_observation = self._discretize(new_observation)

        self.Q[observation[0], observation[1], action] += (
            self.alpha * (
                reward + self.gamma * np.max(self.Q[new_observation[0], new_observation[1]]) -
                self.Q[observation[0], observation[1], action]
                )
        )


if __name__ == "__main__":
    pass

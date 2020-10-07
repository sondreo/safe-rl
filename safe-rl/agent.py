class Agent():
    """Agent parent class."""

    def __init__(self, action_space):
        """Initializes agent class."""

        self.action_space = action_space

    def act(self, observation, reward, done):
        """Perform an action according to a random policy."""

        return self.action_space.sample()


class QLearningAgent(Agent):
    """Q-learning agent class."""

    def __init__(self, action_space, delta):
        """Initializes Q-learning agent class."""

        super().__init__(action_space)
        self.delta = delta

    def act(self, observation, reward, done):
        """Performs an action according to a set policy."""

        return 1


if __name__ == "__main__":
    pass

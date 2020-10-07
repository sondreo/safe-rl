class Agent():
    """Simple agent"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        """Perform an action according to a policy"""

        return self.action_space.sample()


if __name__ == "__main__":
    pass

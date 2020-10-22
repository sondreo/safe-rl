import numpy as np


def discretize_states(states, observation_space, scales):
    """Discretizes input states."""

    states = (states - observation_space.low) * scales

    return np.round(states, 0).astype(int)

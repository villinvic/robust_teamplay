import gymnasium
import numpy as np


class Policy:

    def __init__(self,environment: gymnasium.Env):

        self.n_actions = environment.action_space.n
        self.n_states = environment.observation_space.n

        self.action_probs: np.ndarray = None

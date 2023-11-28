import gymnasium
import numpy as np


class Policy:

    def __init__(self,environment: gymnasium.Env):

        self.n_actions = environment.action_space.n
        self.n_states = environment.observation_space.n

        self.action_probs: np.ndarray = None


    def sample_action(self, state):

        return np.random.choice(self.n_actions, p=self.action_probs[state])

    def sample_deterministic_action(self, state):
        return np.argmax(self.action_probs[state])


import gymnasium
import numpy as np


class BackgroundPopulation:

    def __init__(self, environment: gymnasium.Env):
        self.n_actions = environment.action_space[0].n
        self.n_states = environment.observation_space[0].n
        self.policies: np.ndarray = None

    def get_expected_policy(self, trained_policy, prior):

        policy = np.empty_like(self.policies[0])

        policy[:] = np.sum(prior()[:-1] * self.policies, axis=0) + prior[-1] * trained_policy



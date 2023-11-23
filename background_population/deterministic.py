import gymnasium
import numpy as np

class DeterministicPoliciesPopulation:

    def __init__(self, environment: gymnasium.Env):

        self.n_actions = environment.action_space.n
        self.n_states = environment.observation_space.n

        # A policy is an array pi of shape [STATES, ACTIONS] where pi[s, a] = pi(a|s)

    def build_population(self):

        all_action_sequences = itertools.

        all_deterministic = (
            []
        )

        self.population = ...

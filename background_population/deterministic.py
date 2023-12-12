import gymnasium
import numpy as np
from itertools import product

from background_population import bg_population


def build_deterministic_policies(n_actions, n_states, size=None, seed=None):

    sequences = np.array(list(product(range(n_actions), repeat=n_states)))
    if size is not None:
        np.random.seed(seed)
        sequences = sequences[np.random.choice(len(sequences), size)]

    policies = np.zeros((len(sequences), n_states, n_actions), dtype=np.float32)

    seq_indices, state_indices = np.indices(policies.shape[:2])

    policies[seq_indices, state_indices, sequences] = 1

    return policies


class DeterministicPoliciesPopulation(bg_population.BackgroundPopulation):

    def __init__(self, environment: gymnasium.Env):
        super().__init__(environment)
        self.num_policies = self.n_actions**self.n_states
        self.max_size = 1_000_000
        self.build_population()

    def build_population(self, size=None, seed=None):

        self.policies = build_deterministic_policies(self.n_actions, self.n_states, size, seed)




if __name__ == "__main__":

    env = RepeatedPrisonersDilemmaEnv(episode_length=2)
    bg = DeterministicPoliciesPopulation(env)

    bg.build_population()

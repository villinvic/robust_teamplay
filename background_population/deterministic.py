import gymnasium
import numpy as np
import itertools

from environments.repeated_prisoners import RepeatedPrisonersDilemmaEnv


class DeterministicPoliciesPopulation:

    def __init__(self, environment: gymnasium.Env):

        self.n_actions = environment.action_space[0].n
        self.n_states = environment.observation_space[0].n
        self.seq_len = environment.episode_length
        self.num_policies = self.n_actions**self.n_states
        print(self.num_policies)
        self.max_size = 2 #1_000_000

        # A policy is an array pi of shape [STATES, ACTIONS] where pi[s, a] = pi(a|s)

    def build_population(self):

        # should be a^s
        policies = []
        all_policies = [
            format(i, f'0{int(self.num_policies / 2)}b') for i in np.random.choice(self.num_policies, self.max_size, replace=False)
        ]

        print(all_policies)

        self.population = ...


if __name__ == "__main__":

    env = RepeatedPrisonersDilemmaEnv(episode_length=3)
    bg = DeterministicPoliciesPopulation(env)

    bg.build_population()

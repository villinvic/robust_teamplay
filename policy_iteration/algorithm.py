import numpy as np
from gym.spaces import Discrete

from background_population.bg_population import BackgroundPopulation
from beliefs.prior import Prior
from environments.mdp import compute_multiagent_mdp
from policies.policy import Policy


class PolicyIteration:

    def __init__(self, initial_policy, environment, epsilon=1e-3):

        self.policy = initial_policy
        self.n_states = self.policy.shape[0]
        self.n_actions = self.policy.shape[1]
        self.environment = environment
        self.epsilon = epsilon
        self.n_iter = 100


    def policy_evaluation_for_prior(
            self,
            bg_population: BackgroundPopulation,
            prior: Prior,
    ):
        # samples [batch_size, state, next_state, reward]

        e = np.inf
        values = np.zeros((len(bg_population.policies), self.n_states))

        old_value = np.empty_like(values[0])
        for i in range(len(bg_population.policies)):
            transition_function = compute_multiagent_mdp(self.environment.transition_function, bg_population.policies[i])
            while e < self.epsilon:
                old_value[:] = values[i]
                values[i, :] = np.sum(transition_function * self.policy[:, :, np.newaxis]
                                * (self.environment.reward_function[:, np.newaxis, np.newaxis] + values[i][np.newaxis, np.newaxis])
                                , axis=-1)
                e = np.max(np.abs(old_value-values[i]))

        return np.sum(values * prior(), axis=0), values


    def policy_evaluation_for_prior2(
            self,
            bg_population: BackgroundPopulation,
            prior: Prior,
    ):
        # samples [batch_size, state, next_state, reward]

        e = np.inf
        values = np.zeros((len(bg_population.policies), self.n_states))

        old_value = np.empty_like(values[0])
        for i in range(len(bg_population.policies)):
            transition_function = compute_multiagent_mdp(self.environment.transition_function, bg_population.policies[i])
            while e < self.epsilon:
                old_value[:] = values[i]

                for state in range(self.n_states):
                    for action in range(self.n_actions):
                        for next_state in range(self.n_states):
                            values[i, state] = (transition_function[state, action, next_state]
                                                * self.policy[state, action]
                                                * (self.environment.reward_function[next_state] + old_value[next_state])
                                                )
                e = np.max(np.abs(old_value-values[i]))

        return np.sum(values * prior(), axis=0), values


if __name__ == '__main__':


    class Env:
        def __init__(self, players=2, states=3, actions=2):
            transition_function = np.random.random((states,) + (actions,) * players + (states,))
            normalization = np.sum(transition_function, axis=0, keepdims=True)
            for i in range(players):
                normalization = np.sum(normalization, axis=i + 1, keepdims=True)
            transition_function /= normalization

            self.transition_function = transition_function
            self.reward_function = np.random.random(states)

    env = Env()

    policy = np.random.random((env.transition_function.shape[0], env.transition_function[1]))
    policy /= np.sum(policy, axis=1, keepdims=True)

    alg = PolicyIteration(initial_policy=policy, environment=env)

    alg.policy_evaluation_for_prior2()

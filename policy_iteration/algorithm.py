import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.spaces import Dict

from background_population.bg_population import BackgroundPopulation
from background_population.deterministic import DeterministicPoliciesPopulation
from beliefs.prior import Prior
from environments.mdp import compute_multiagent_mdp
from policies.policy import Policy


class PolicyIteration:

    def __init__(self, initial_policy : Policy, environment, epsilon=1e-3, learning_rate=1e-3):

        self.policy = initial_policy
        self.n_states = self.policy.action_logits.shape[0]
        self.n_actions = self.policy.action_logits.shape[1]
        self.environment = environment
        self.epsilon = epsilon
        self.n_iter = 100
        self.lr = learning_rate


    def policy_evaluation_for_prior(
            self,
            bg_population: BackgroundPopulation,
            prior: Prior,
    ):
        # samples [batch_size, state, next_state, reward]

        values = np.zeros((len(prior.beta_logits), self.n_states))
        old_value = np.empty_like(values[0])

        action_probs = self.policy.get_probs()

        for i in range(len(bg_population.policies)):
            transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, bg_population.policies[i])
            e = np.inf
            while e > self.epsilon:
                old_value[:] = values[i]
                values[i, :] = np.sum(reward_function * action_probs, axis=-1) + np.sum(np.sum(transition_function * action_probs[:, :, np.newaxis]
                                * (self.environment.gamma * values[np.newaxis, np.newaxis, i])
                                , axis=-1), axis=-1)

                e = np.max(np.abs(old_value-values[i]))

        # Selfplay (we want to maximize the avg joint rewards)
        # therefore, value function is the average function of the copies
        transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, action_probs, joint_rewards=True)
        e = np.inf
        while e > self.epsilon:
            old_value[:] = values[-1]

            values[-1, :] = np.sum(reward_function * action_probs, axis=-1) + np.sum(np.sum(transition_function * action_probs[:, :, np.newaxis]
                                  * (self.environment.gamma * values[np.newaxis, np.newaxis, -1])
                                  , axis=-1), axis=-1)

            e = np.max(np.abs(old_value - values[-1]))

        return np.sum(values * prior()[:, np.newaxis], axis=0), values


    def policy_evaluation_for_prior2(
            self,
            bg_population: BackgroundPopulation,
            prior: Prior,
    ):
        # samples [batch_size, state, next_state, reward]

        values = np.zeros((len(prior.beta_logits), self.n_states))

        action_probs = self.policy.get_probs()

        old_value = np.empty_like(values[0])
        for i in range(len(bg_population.policies)):

            transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, bg_population.policies[i])
            e = np.inf

            while e > self.epsilon:
                old_value[:] = values[i]
                values[i, :] = 0.

                for state in range(self.n_states):
                    for action in range(self.n_actions):
                        values[i, state] += action_probs[state, action] * reward_function[state, action]
                        for next_state in range(self.n_states):

                            values[i, state] += (transition_function[state, action, next_state]
                                                * action_probs[state, action]
                                                * (self.environment.gamma * old_value[next_state])
                                                )
                e = np.max(np.abs(old_value-values[i]))

        # Selfplay (we want to maximize the avg joint rewards)
        # therefore, value function is the average function of the copies
        transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, action_probs, joint_rewards=True)
        e = np.inf
        while e > self.epsilon:
            old_value[:] = values[-1]
            values[-1, :] = 0.
            for state in range(self.n_states):
                for action in range(self.n_actions):
                    values[-1, state] += action_probs[state, action] * reward_function[state, action]
                    for next_state in range(self.n_states):
                        values[-1, state] += (transition_function[state, action, next_state]
                                            * action_probs[state, action]
                                            * (self.environment.gamma * old_value[next_state])
                                            )
            e = np.max(np.abs(old_value - values[-1]))

        return np.sum(values * prior()[:, np.newaxis], axis=0), values

    def policy_improvement(self, bg_population, prior: Prior, vf):
        params = prior()
        action_probs = self.policy.get_probs()

        mdps = [
            compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, bg_population.policies[i])
            for i in range(len(bg_population.policies))
        ] + [compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, action_probs, joint_rewards=True)]
        transition_functions = [
            params[i] * mdp[0]
            for i, mdp  in enumerate(mdps)
        ]
        reward_functions = [
            params[i] * mdp[1]
            for i, mdp in enumerate(mdps)
        ]

        expected_transition_function = sum(transition_functions) / len(transition_functions)
        expected_reward_function = sum(reward_functions) / len(reward_functions)

        new_policy = np.zeros_like(action_probs)

        action_values = expected_reward_function + np.sum(expected_transition_function * (
            self.environment.gamma * vf
        ), axis=-1)

        new_policy[np.arange(len(new_policy)), np.argmax(action_values, axis=-1)] = 1.

        self.policy.action_logits[:] = self.policy.action_logits * (1-self.lr) + self.lr * new_policy

    def exact_pg(self, bg_population, prior: Prior, vf):

        action_probs = self.policy.get_probs()
        params = prior()

        mdps = [
            compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, bg_population.policies[i])
            for i in range(len(bg_population.policies))
        ] + [compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, action_probs, joint_rewards=True)]
        transition_functions = [
            params[i] * mdp[0]
            for i, mdp  in enumerate(mdps)
        ]
        reward_functions = [
            params[i] * mdp[1]
            for i, mdp in enumerate(mdps)
        ]

        expected_transition_function = sum(transition_functions) / len(transition_functions)
        expected_reward_function = sum(reward_functions) / len(reward_functions)

        q = expected_reward_function + self.environment.gamma * np.sum(expected_transition_function * vf[np.newaxis, np.newaxis], axis=-1)

        self.policy.apply_loss(q, lr=self.lr)

    def policy_improvement2(self, bg_population, prior: Prior, vf):
        params = prior()
        mdps = [
            compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, bg_population.policies[i])
            for i in range(len(bg_population.policies))
        ] + [compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, self.policy, joint_rewards=True)]
        transition_functions = [
            params[i] * mdp[0]
            for i, mdp  in enumerate(mdps)
        ]
        reward_functions = [
            params[i] * mdp[1]
            for i, mdp in enumerate(mdps)
        ]

        expected_transition_function = sum(transition_functions) / len(transition_functions)
        expected_reward_function = sum(reward_functions) / len(reward_functions)


        new_policy = np.zeros_like(self.policy)
        for state in range(self.n_states):
            for action in range(self.n_actions):
                new_policy[state, action] += expected_reward_function[state, action]
                for next_state in range(self.n_states):
                    new_policy[state, action] += (expected_transition_function[state, action, next_state]
                        * (self.environment.gamma * vf[next_state])
                    )


        exp_values = np.exp(new_policy)
        new_policy[:] = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

        self.policy[:] = self.policy * (1-self.lr) + self.lr * new_policy






if __name__ == '__main__':

    np.random.seed(0)

    class RandomDiscountedMDP:
        def __init__(self, players=2, states=3, actions=2):
            transition_function = np.random.random((states,) + (actions,) * players + (states,))

            self.transition_function = transition_function / transition_function.sum(axis=-1, keepdims=True)
            self.reward_function = np.random.random((states, ) + (actions,) * players)

            self.action_space = Dict({i: Discrete(actions) for i in range(players)})
            self.observation_space = Dict({i: Discrete(states) for i in range(players)})
            self.gamma = 0.9

            self.s0 = 0

    env = RandomDiscountedMDP()

    policy = np.random.random((env.transition_function.shape[0], env.transition_function.shape[1]))
    policy /= np.sum(policy, axis=1, keepdims=True)

    bg_population = DeterministicPoliciesPopulation(env)
    bg_population.build_population()

    alg = PolicyIteration(initial_policy=policy, environment=env)

    prior = Prior(len(bg_population.policies)+1)
    prior.initialize_uniformly()

    expected_vf, vf = alg.policy_evaluation_for_prior(bg_population, prior)
    expected_vf2, vf2 = alg.policy_evaluation_for_prior2(bg_population, prior)

    print(vf, vf2)

    policy = np.empty_like(alg.policy)
    policy[:] = alg.policy
    alg.policy_improvement(bg_population, prior, expected_vf)

    #print(alg.policy)
    alg.policy[:] = policy
    #print(alg.policy)

    alg.policy_improvement2(bg_population, prior, expected_vf)
    #print(alg.policy)

    input()

    assert(np.allclose(vf, vf2)), (vf, vf2)

    for i in range(5000):
        expected_vf, vf = alg.policy_evaluation_for_prior(bg_population, prior)

        vf_s0 = vf[:, env.s0]

        alg.policy_improvement2(bg_population, prior, expected_vf)
        prior.update_prior(vf_s0, regret=False)
        prior.project()

        print(i, prior(), expected_vf[env.s0])

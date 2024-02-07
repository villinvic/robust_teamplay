from time import time

import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.spaces import Dict

from background_population.bg_population import TabularBackgroundPopulation
from background_population.deterministic import DeterministicPoliciesPopulation
from beliefs.prior import Prior
from environments.mdp import compute_multiagent_mdp
from policies.policy import Policy
from policies.tabular_policy import TabularPolicy


class PolicyIteration:

    def __init__(self, initial_policy : Policy, environment, epsilon=1e-3, learning_rate=1e-3, lambda_=1e-3):

        self.policy = initial_policy
        self.n_states = self.policy.action_logits.shape[0]
        self.n_actions = self.policy.action_logits.shape[1]
        self.environment = environment
        self.epsilon = epsilon
        self.n_iter = 100
        self.lr = learning_rate
        self.lambda_ = lambda_


    def policy_evaluation_for_prior(
            self,
            bg_population: TabularBackgroundPopulation,
            prior: Prior,
    ):
        # samples [batch_size, state, next_state, reward]

        values = np.zeros((len(prior.beta_logits), self.n_states))
        action_probs = self.policy.get_probs()

        transition_functions = []
        reward_functions = []
        for i in range(len(bg_population.policies)):
            transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function,
                                                                          self.environment.reward_function,
                                                                          bg_population.policies[i],
                                                                          self.environment.curr_state_to_opp_state)
            transition_functions.append(transition_function)
            reward_functions.append(reward_function)

        sp_transition_function, sp_reward_function = compute_multiagent_mdp(self.environment.transition_function,
                                                                      self.environment.reward_function,
                                                                      action_probs,
                                                                      self.environment.curr_state_to_opp_state,
                                                                      joint_rewards=(0.5,0.5))
        transition_functions.append(sp_transition_function)
        reward_functions.append(sp_reward_function)

        transition_functions = np.stack(transition_functions, axis=0)
        reward_functions = np.stack(reward_functions, axis=0)

        action_probs_p1 = action_probs[np.newaxis]
        action_probs_p1_p4 = action_probs_p1[:, :, :, np.newaxis]

        #old_values = np.zeros_like(values)
        for t in range(self.epsilon):
            # old_values[:] = values
            # values[:] = 0.
            #
            # for state in range(self.n_states):
            #     s = 0
            #     for next_state in range(self.n_states):
            #         for action in range(self.n_actions):
            #             s += transition_functions[:, state, action, next_state]
            #             values[:, state] += transition_functions[:, state, action, next_state] * action_probs_p1[:, state, action] * (
            #                 reward_functions[:, state, action] + self.environment.gamma * old_values[:, next_state]
            #             )
            # print(reward_functions[:, :, :])
            # print( (transition_functions * action_probs_p1_p4 * reward_functions[:, :, :, np.newaxis]).sum(axis=-1).sum(axis=-1))
            #

            values[:, :] = np.sum(np.sum(

                transition_functions * action_probs_p1_p4 * (reward_functions[:, :, :, np.newaxis]
                            + self.environment.gamma * values[:, np.newaxis, np.newaxis])

                            , axis=-1), axis=-1)


        # for i in range(len(bg_population.policies)):
        #     transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, bg_population.policies[i])
        #     e = np.inf
        #     while e > self.epsilon:
        #         old_value[:] = values[i]
        #         values[i, :] = np.sum(reward_function * action_probs, axis=-1) + np.sum(np.sum(transition_function * action_probs[:, :, np.newaxis]
        #                         * (self.environment.gamma * values[np.newaxis, np.newaxis, i])
        #                         , axis=-1), axis=-1)
        #
        #         e = np.max(np.abs(old_value-values[i]))
        #
        # # Selfplay (we want to maximize the avg joint rewards)
        # # therefore, value function is the average function of the copies
        # transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, action_probs, joint_rewards=True)
        # e = np.inf
        # while e > self.epsilon:
        #     old_value[:] = values[-1]
        #
        #     values[-1, :] = np.sum(reward_function * action_probs, axis=-1) + np.sum(np.sum(transition_function * action_probs[:, :, np.newaxis]
        #                           * (self.environment.gamma * values[np.newaxis, np.newaxis, -1])
        #                           , axis=-1), axis=-1)
        #
        #     e = np.max(np.abs(old_value - values[-1]))

        return np.sum(values * prior()[:, np.newaxis], axis=0), values


    def policy_evaluation_for_scenario(
            self,
            scenario
    ):
        teammate, rewards = scenario

        # samples [batch_size, state, next_state, reward]

        value = np.zeros((self.n_states))

        action_probs = self.policy.get_probs()

        transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function,
                                                                      self.environment.reward_function,
                                                                      teammate,
                                                                      self.environment.curr_state_to_opp_state,
                                                                      rewards)

        action_probs_p3 = action_probs[:, :, np.newaxis]

        for t in range(self.epsilon):
            # old_values[:] = values
            value[:] = np.sum(
                np.sum(

                    transition_function * action_probs_p3 * (reward_function[:, :, np.newaxis]
                       + self.environment.gamma * value[np.newaxis, np.newaxis]
                                                             )

                       , axis=-1), axis=-1
            )
            #print(value)

        return value

    def policy_improvement_for_scenario(self, scenario, vf):

        teammate, rewards = scenario
        action_probs = self.policy.get_probs()

        transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function,
                                                                      self.environment.reward_function,
                                                                      teammate,
                                                                      self.environment.curr_state_to_opp_state,
                                                                      rewards)

        new_policy = np.zeros_like(action_probs)

        action_values = np.sum(transition_function * (
            self.environment.gamma * vf[np.newaxis, np.newaxis] + reward_function[:, :, np.newaxis]
        ), axis=-1)

        new_policy[np.arange(len(new_policy)), np.argmax(action_values, axis=-1)] = 1.

        self.policy.action_logits[:] = self.policy.action_logits * (1-self.lr) + self.lr * new_policy

    def exact_pg(self, bg_population, prior: Prior, vf, previous_copy: TabularPolicy = None):

        action_probs = self.policy.get_probs()
        all_policies = [bg_population.policies]
        if previous_copy is not None:
            all_policies.append(previous_copy.get_probs()[np.newaxis])
        else:
            all_policies.append(action_probs[np.newaxis])

        all_policies = np.concatenate(all_policies, axis=0)
        all_rewards = np.concatenate([np.tile([1., 0.], (len(bg_population.policies), 1)), [[0.5, 0.5]]], axis=0)
        #
        # expected_teammate = (all_policies * prior()[:, np.newaxis, np.newaxis]).sum(axis=0)
        # expected_rewards = (all_rewards * prior()[:, np.newaxis]).sum(axis=0)
        # expected_transition_function, expected_reward_function = compute_multiagent_mdp(
        #     self.environment.transition_function, self.environment.reward_function,
        #     expected_teammate, joint_rewards=expected_rewards
        # )


        gradients = []
        for teammate, reward_weights, V, scenario_prob \
                in zip(all_policies, all_rewards, vf, prior()):

            induced_transition_function, induced_reward_function = compute_multiagent_mdp(
                self.environment.transition_function, self.environment.reward_function,
                teammate, self.environment.curr_state_to_opp_state, joint_rewards=reward_weights
            )

            Q = induced_reward_function + self.environment.gamma * np.sum(induced_transition_function * V[np.newaxis, np.newaxis], axis=-1)



            gradients.append(scenario_prob * self.policy.compute_pg(
                Q, V, transition_function=induced_transition_function, lambda_=self.lambda_
            ))

        print(gradients)
        np.random.shuffle(gradients)
        for g in gradients:
         self.policy.apply_gradient(g, lr=self.lr)


    def exact_pg_mixture(self, bg_population, prior: Prior, vf, previous_copy: TabularPolicy = None):

        action_probs = self.policy.get_probs()
        all_policies = list(bg_population.policies)
        if previous_copy is not None:
            all_policies.append(previous_copy.get_probs())
        else:
            all_policies.append(action_probs)

        all_rewards = [np.array([1, 0], dtype=np.float32)] * len(all_policies) + [np.array([0.5, 0.5], dtype=np.float32)]
        expected_policy = np.zeros_like(action_probs)
        expected_rewards = np.zeros(2, dtype=np.float32)
        V = np.zeros_like(vf[0])

        for bg_policy, reward_weights, Vi, scenario_prob in zip(all_policies, all_rewards, vf, prior()):
            expected_policy += scenario_prob * bg_policy
            expected_rewards += scenario_prob * reward_weights
            V += scenario_prob * Vi


        induced_transition_function, induced_reward_function = compute_multiagent_mdp(
            self.environment.transition_function, self.environment.reward_function,
            expected_policy, self.environment.curr_state_to_opp_state, joint_rewards=expected_rewards
        )

        Q = induced_reward_function + self.environment.gamma * np.sum(induced_transition_function * V[np.newaxis, np.newaxis], axis=-1)

        gradient = self.policy.compute_pg(
            Q, V, transition_function=induced_transition_function, lambda_=self.lambda_
        )

        self.policy.apply_gradient(gradient, lr=self.lr)


    def exact_regret_pg(self, bg_population, prior: Prior, vfs, best_response_vfs):

        action_probs = self.policy.get_probs()

        all_policies = np.concatenate([bg_population.policies, action_probs[np.newaxis]], axis=0)
        all_rewards = np.concatenate([np.tile([1., 0.], (len(bg_population.policies), 1)), [[0.5, 0.5]]], axis=0)
        #
        # expected_teammate = (all_policies * prior()[:, np.newaxis, np.newaxis]).sum(axis=0)
        # expected_rewards = (all_rewards * prior()[:, np.newaxis]).sum(axis=0)
        # expected_transition_function, expected_reward_function = compute_multiagent_mdp(
        #     self.environment.transition_function, self.environment.reward_function,
        #     expected_teammate, joint_rewards=expected_rewards
        # )

        total_gradients = 0.
        for teammate, reward_weights, V, scenario_prob, V_star  \
                in zip(all_policies, all_rewards, vfs, prior(), best_response_vfs):

            induced_transition_function, induced_reward_function = compute_multiagent_mdp(
                self.environment.transition_function, self.environment.reward_function,
                teammate, self.environment.curr_state_to_opp_state, joint_rewards=reward_weights
            )

            Q_star = induced_reward_function + self.environment.gamma * np.sum(induced_transition_function * V_star[np.newaxis, np.newaxis], axis=-1)

            Q = induced_reward_function + self.environment.gamma * np.sum(induced_transition_function * V[np.newaxis, np.newaxis], axis=-1)

            #Q_regret = Q_star - Q
            total_gradients += - scenario_prob * self.policy.compute_pg(
                Q_star-Q, V_star - V, transition_function=induced_transition_function, lambda_=self.lambda_
            )
        self.policy.apply_gradient(total_gradients, lr=self.lr)

    def policy_improvement2(self, bg_population, prior: Prior, vf):
        action_probs = self.policy.get_probs()

        # mdps = [
        #     compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, bg_population.policies[i])
        #     for i in range(len(bg_population.policies))
        # ] + [compute_multiagent_mdp(self.environment.transition_function, self.environment.reward_function, self.policy, joint_rewards=(0.5,0.5))]
        # transition_functions = [
        #     params[i] * mdp[0]
        #     for i, mdp  in enumerate(mdps)
        # ]
        # reward_functions = [
        #     params[i] * mdp[1]
        #     for i, mdp in enumerate(mdps)
        # ]

        all_policies = np.concatenate([bg_population.policies, action_probs[np.newaxis]], axis=0)
        all_rewards = np.concatenate([np.stack([1., 0.], len(bg_population.policies), axis=0), [[0.5, 0.5]]], axis=0)

        expected_teammate =  (all_policies * prior()).sum(axis=0)
        expected_rewards = (all_rewards * prior()).sum(axis=0)
        expected_transition_function, expected_reward_function = compute_multiagent_mdp(
            self.environment.transition_function, self.environment.reward_function,
            expected_teammate, self.environment.curr_state_to_opp_state, joint_rewards=expected_rewards
        )

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
        def __init__(self, players=2, states=8, actions=2):
            transition_function = np.random.random((states,) + (actions,) * players + (states,))

            self.transition_function = transition_function / transition_function.sum(axis=-1, keepdims=True)
            self.reward_function = np.random.random((states, ) + (actions,) * players)

            self.action_space = Dict({i: Discrete(actions) for i in range(players)})
            self.observation_space = Dict({i: Discrete(states) for i in range(players)})
            self.gamma = 0.9

            self.s0 = 0

    env = RandomDiscountedMDP()


    policy = Policy(env)
    policy.action_logits[:] = np.random.random((env.transition_function.shape[0], env.transition_function.shape[1]))

    bg_population = DeterministicPoliciesPopulation(env)
    bg_population.build_population()

    alg = PolicyIteration(initial_policy=policy, environment=env, epsilon=10000)

    prior = Prior(len(bg_population.policies)+1)
    prior.initialize_uniformly()

    t = time()
    expected_vf, vf = alg.policy_evaluation_for_prior(bg_population, prior)
    t2 = time()
    alg.epsilon = 1e-3
    expected_vf2, vf2 = alg.policy_evaluation_for_prior2(bg_population, prior)

    print(vf, vf2, t2 - t, time() - t2)

    input()

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

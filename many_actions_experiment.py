import argparse

import joypy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from background_population.deterministic import DeterministicPoliciesPopulation
from beliefs.prior import Prior
from environments.many_actions import ManyActionEnv
import numpy as np

from environments.mdp import compute_multiagent_mdp
from policies.policy import Policy
from policies.tabular_policy import TabularPolicy
from policy_iteration.algorithm import PolicyIteration


def many_actions_experiment(n_actions, policy_lr, prior_lr, lambda_, use_regret=True, self_play=False):
    environment = ManyActionEnv(n_actions=n_actions)

    np.random.seed(0)
    robust_policy = Policy(environment)
    robust_policy.initialize_uniformly()
    # robust_policy.initialize_randomly()

    # do with policy types ?
    num_policies = None
    seed = 1
    bg_population = DeterministicPoliciesPopulation(environment)
    bg_population.build_population(size=num_policies, seed=seed)

    algo = PolicyIteration(robust_policy, environment, epsilon=5, learning_rate=policy_lr, lambda_=lambda_)

    # belief over worst teammate policy (all bg individuals and our self)
    belief = Prior(len(bg_population.policies) + 1, learning_rate=prior_lr)
    # belief.initialize_randomly()
    if self_play:
        belief.initialize_certain(belief.dim - 1)
    else:
        belief.initialize_uniformly()

    vfs = []
    regret_scores = []
    vf_scores = []

    # Compute best responses for regret
    best_response_vfs = np.empty((len(bg_population.policies) + 1, robust_policy.n_states), dtype=np.float32)
    priors = []
    priors.append(belief().copy())
    best_responses = {}
    for p_id in range(len(bg_population.policies) + 1):
        best_response = TabularPolicy(environment)
        best_response.initialize_uniformly()
        best_response.action_logits[0] = 0.
        best_response.action_logits[0, 2] = 1.
        best_response.action_logits[1] = 0.
        best_response.action_logits[1, 2] = 1.
        best_response.action_logits[2] = 0.
        best_response.action_logits[2, 2] = 1.
        p_algo = PolicyIteration(best_response, environment, learning_rate=1., epsilon=5)
        if p_id < len(bg_population.policies):
            scenario = bg_population.policies[p_id], (1, 0)
        else:
            scenario = best_response.get_probs(), (0.5, 0.5)
        for i in range(15):
            if p_id == len(bg_population.policies):
                scenario = best_response.get_probs(), (0.5, 0.5)
            vf = p_algo.policy_evaluation_for_scenario(scenario)
            if p_id == len(bg_population.policies):
                print(p_id, vf)
                input()

            p_algo.policy_improvement_for_scenario(scenario, vf)

        vf = p_algo.policy_evaluation_for_scenario(scenario)

        best_response_vfs[p_id] = vf
        best_responses[p_id] = best_response

    print(best_response_vfs[:, environment.s0])
    input()

    regrets = []
    for i in range(3000):

        expected_vf, vf = algo.policy_evaluation_for_prior(bg_population, belief)

        vf_s0 = vf[:, environment.s0]

        all_regrets = best_response_vfs - vf

        regret = all_regrets[:, environment.s0]

        if use_regret:
            # p_algo = PolicyIteration(best_responses[belief.dim-1], environment, learning_rate=1, epsilon=3)
            # scenario = robust_policy.get_probs(), (0.5, 0.5)
            # vf_sp = p_algo.policy_evaluation_for_scenario(scenario)
            #
            # current_best_response_vfs = best_response_vfs.copy()
            # current_best_response_vfs[-1] = vf_sp

            belief.update_prior(regret, regret=True)
            algo.exact_pg(bg_population, belief, vf)
            # algo.exact_regret_pg(bg_population, belief, vf, best_response_vfs)
        else:
            belief.update_prior(vf_s0, regret=False)
            algo.exact_pg(bg_population, belief, vf)

        vfs.append(expected_vf[environment.s0])
        regrets.append(np.sum(regret * belief()))

        vf_scores.append(np.mean(vf_s0))
        regret_scores.append(np.mean(regret))

        print(vf_scores[-1], regret_scores[-1])
        priors.append(belief().copy())
        a_probs = robust_policy.get_probs()
        entropy = np.mean(np.sum(-a_probs * np.log(a_probs + 1e-8), axis=-1))
        print(belief(), entropy)
        print(regret)
    print(robust_policy.get_probs())

    plt.plot(regrets, label="regret w.r.t prior")
    plt.plot(regret_scores, label="regret w.r.t uniform (test time) prior")
    plt.title("Regret over iterations")
    plt.ylabel("Regret")
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(vfs, label="policy utility w.r.t prior")
    plt.plot(vf_scores, label="policy utility w.r.t uniform (test time) prior")
    plt.title("Utility over iterations")
    plt.ylabel("Utility")
    plt.legend()

    plt.show()
    plt.clf()

    df = pd.DataFrame()
    num_plots = 15
    d = len(priors) // num_plots

    for i in range(0, len(priors), d):
        arr_size = 100 * len(priors[0])
        counts = np.int32(priors[i] * arr_size)
        k = 0
        while np.sum(counts) != arr_size:
            counts[k % len(counts)] += 1
            k += 1
        samples = []
        for idx in range(len(priors[0])):
            samples.extend([idx] * counts[idx])
        df[i] = samples

    joypy.joyplot(df, overlap=0, colormap=cm.OrRd_r, linecolor='b', linewidth=.5, hist=True, bins=len(priors[0]),
                  figsize=(4, 6), x_range=list(range(len(priors[0]))))
    plt.xticks(list(range(len(priors[0]))))

    plt.title("Scenario distribution over iterations")
    plt.xlabel("Scenarios")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='repeated_prisoners_experiment',
    )
    parser.add_argument("experiment", type=str, choices=["main", "eval", "find_best"], default="main")
    parser.add_argument("--n_actions", type=int, default=8)
    parser.add_argument("--policy_lr", type=float, default=1e-2)
    parser.add_argument("--prior_lr", type=float, default=1e-2)
    parser.add_argument("--use_regret", type=bool, default=False)
    parser.add_argument("--lambda_", type=float, default=1e-3)
    parser.add_argument("--sp", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pop_size", type=int, default=4)

    args = parser.parse_args()

    if args.experiment == "main":
        many_actions_experiment(n_actions=args.n_actions, policy_lr=args.policy_lr, prior_lr=args.prior_lr, use_regret=args.use_regret,
                                lambda_=args.lambda_)
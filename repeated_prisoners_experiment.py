import itertools
import os
import string
from typing import Union

import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joypy
from matplotlib import cm
from tqdm import tqdm

from background_population.deterministic import DeterministicPoliciesPopulation
from beliefs.prior import Prior
from environments.mdp import compute_multiagent_mdp
from environments.repeated_prisoners import RepeatedPrisonersDilemmaEnv
from policies.policy import Policy
from policies.tabular_policy import TabularPolicy
from policy_iteration.algorithm import PolicyIteration
import argparse
import subprocess
import multiprocessing as mp

def main(policy_lr, prior_lr, n_seeds=50, episode_length=2, pop_size=2):

    approaches = [dict(
            scenario_distribution_optimization="Regret maximizing",
            use_regret=True,
            policy_lr=policy_lr,
            prior_lr=prior_lr,
            true_solution=False
        ),
        dict(
            scenario_distribution_optimization="Utility minimizing",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=prior_lr,
            true_solution=False

        ),
        dict(
            scenario_distribution_optimization="Fixed uniform",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=0.,
            true_solution=False
        ),
        dict(
            true_solution=True
        ),
    ]
    all_jobs = []
    for seed in range(n_seeds):
        seeded_configs = [{
            "seed": seed,
            "episode_length": episode_length,
            "pop_size": pop_size,
            "job": repeated_prisoners_experiment_with_config if idx < len(approaches)-1
            else repeated_prisoners_best_solution_with_config
        } for idx in range(len(approaches))]

        for config, approach in zip(seeded_configs, approaches):
            config.update(**approach)
        all_jobs.extend(seeded_configs)

    results = []

    with mp.Pool(os.cpu_count(), maxtasksperchild=1) as p:
        for ret, config in tqdm(
                zip(p.imap(run_job, all_jobs), all_jobs), total=len(all_jobs)
        ):
            ret.update(**config)
            results.append(ret)

    data = pandas.DataFrame(results)
    approach_results = data # #data[data['true_solution'] == False]
    true_solution = data[data['true_solution']]

    utility_error = []
    regret_error = []
    for seed in range(n_seeds):
        utility_error.append(approach_results[approach_results["seed"]==seed]["utility"].values
                             - true_solution[true_solution["seed"]==seed]["utility"].values)
        regret_error.append(approach_results[approach_results["seed"]==seed]["regret"].values
                             - true_solution[true_solution["seed"]==seed]["regret"].values)

    # Remove rows with "Regret Maximizing" from the dataframe

    # Calculate utility error by subtracting utility of "Regret Maximizing" from other approaches
    approach_results['utility_error']  = [item for sublist in utility_error for item in sublist]
    approach_results['regret_error'] = [item for sublist in regret_error for item in sublist]

    grouped_data = approach_results.groupby('scenario_distribution_optimization').agg({
        'utility': 'mean',
        'utility_error': 'std',
        'regret': 'mean',
        'regret_error': 'std',
    })
    #grouped_data = approach_results.groupby('scenario_distribution_optimization').agg(['mean'])[['utility', 'regret']]
    # grouped_data.apply(lambda row: f"{row['utility']:.3f} $\\pm$ {row['utility_error']:.3f}", axis=1)
    grouped_data['utility'] = grouped_data.apply(lambda row:
                    f"${row['utility']:.3f} \\pm {row['utility_error']:.3f}$",
                                               axis=1)

    grouped_data['regret'] = grouped_data.apply(lambda row:
                    f"${row['regret']:.3f} \\pm {row['regret_error']:.3f}$",
                                               axis=1)


    # Output LaTeX table with mean utility plus/minus utility error std and std regret
    latex_table = string.capwords(
        grouped_data[['utility', 'regret']].to_latex(escape=False, float_format="%.3f").replace("_", " ").replace("nan", "0"))

    print(latex_table)

    #print(data)

def run_job(config):
    job = config.pop("job")
    return job(config)

def repeated_prisoners_experiment_with_config(config):
    return repeated_prisoners_experiment(**config)

def repeated_prisoners_best_solution_with_config(config):
    return find_best(**config)

def repeated_prisoners_experiment(
        policy_lr=1e-3,
        prior_lr=1e-3,
        use_regret=False,
        self_play=False,
        lambda_=0.,
        seed=0,
        episode_length=1,
        pop_size=None,
        **kwargs):

    environment = RepeatedPrisonersDilemmaEnv(episode_length=episode_length)

    robust_policy = Policy(environment)
    robust_policy.initialize_uniformly()
    #robust_policy.initialize_randomly()

    # do with policy types ?
    num_policies = pop_size
    seed = seed
    bg_population = DeterministicPoliciesPopulation(environment)
    bg_population.build_population(size=num_policies, seed=seed)

    algo = PolicyIteration(robust_policy, environment, epsilon=4, learning_rate=policy_lr, lambda_=lambda_)

    # belief over worst teammate policy (all bg individuals and our self)
    belief = Prior(len(bg_population.policies)+1, learning_rate=prior_lr)
    #belief.initialize_randomly()
    if self_play:
        belief.initialize_certain(belief.dim - 1)
    else:
        #belief.initialize_randomly()
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
        p_algo = PolicyIteration(best_response, environment, learning_rate=1, epsilon=4)
        if p_id < len(bg_population.policies):
            scenario = bg_population.policies[p_id], (1, 0)
        else:
            scenario = best_response.get_probs(), (0.5, 0.5)
        for i in range(episode_length * 5):
            if p_id == len(bg_population.policies):
                scenario = best_response.get_probs(), (0.5, 0.5)
            vf = p_algo.policy_evaluation_for_scenario(scenario)

            p_algo.policy_improvement_for_scenario(scenario, vf)

        vf = p_algo.policy_evaluation_for_scenario(scenario)

        best_response_vfs[p_id] = vf
        best_responses[p_id] = best_response
    #
    # print(best_response_vfs[:, environment.s0])
    # input()

    # d = {
    #     0: "collaborate",
    #     1: "defect"
    # }
    # for i, p in enumerate(bg_population.policies):
    #     string = f'policy [{i}] \n '
    #     for state in range(p.shape[0]):
    #         string += f"state {state} -> {d[np.argmax(p[state])]}\n"
    #     print(string)
    #     print()
    #
    # for i, p in best_responses.items():
    #     #t, r = compute_multiagent_mdp(environment.transition_function, environment.reward_function, p)
    #     print(i, p)
    #     print(p.get_probs())
    #
    # input()


    regrets = []
    for i in range(5000):

        expected_vf, vf = algo.policy_evaluation_for_prior(bg_population, belief)

        vf_s0 = vf[:, environment.s0]

        all_regrets = best_response_vfs - vf

        regret_s0 = all_regrets[:, environment.s0]

        if use_regret:
            # p_algo = PolicyIteration(best_responses[belief.dim-1], environment, learning_rate=1, epsilon=3)
            # scenario = robust_policy.get_probs(), (0.5, 0.5)
            # vf_sp = p_algo.policy_evaluation_for_scenario(scenario)
            #
            # current_best_response_vfs = best_response_vfs.copy()
            # current_best_response_vfs[-1] = vf_sp
            algo.exact_pg(bg_population, belief, vf)
            belief.update_prior(regret_s0, regret=True)
            #algo.exact_regret_pg(bg_population, belief, vf, best_response_vfs)
        else:
            algo.exact_pg(bg_population, belief, vf)
            belief.update_prior(vf_s0, regret=False)




        vfs.append(expected_vf[environment.s0])
        regrets.append(np.sum(regret_s0 * belief()))

        vf_scores.append(np.mean(vf_s0))
        regret_scores.append(np.mean(regret_s0))

        # subprocess.call("clear")
        # print()
        # print(f"--- Iteration {i} ---")
        # print()
        #
        # print("Test time score (utility, regret):", vf_scores[-1], regret_scores[-1])
        # print("w.r.t current prior (utility, regret):", vfs[-1], regrets[-1])
        # print("Utility per scenario:", vf_s0)
        # print("Regret per scenario", regret_s0)
        #
        #
        # priors.append(belief().copy())
        # a_probs = robust_policy.get_probs()
        # entropy = np.mean(np.sum(-a_probs * np.log(a_probs+1e-8), axis=-1))
        # print("Current distribution over scenarios :", belief())
        #
        # print("Policy:", robust_policy.get_probs())

    argmax_actions = np.zeros_like(robust_policy.action_logits)
    argmax_actions[np.arange(len(argmax_actions)), np.argmax(robust_policy.action_logits, axis=-1)] = 1.
    argmax_policy = TabularPolicy(
        environment,
        argmax_actions
    )
    algo_p = PolicyIteration(argmax_policy, environment, epsilon=5)
    expected_vf, vf = algo_p.policy_evaluation_for_prior(bg_population, belief)
    vf_s0 = vf[:, environment.s0]
    all_regrets = best_response_vfs - vf
    regret_s0 = all_regrets[:, environment.s0]
    # print()
    # print("Results:")
    #print("Test time score (utility, regret):", np.mean(vf_s0), np.mean(regret_s0))


    #print(regret)
    return {"utility": np.mean(vf_s0), "regret": np.mean(regret_s0)}


    data = {
        "priors": priors,
        "values": vfs,
        "value_scores": vfs,
        "regrets": regrets,
        "regret_score": regret_scores,
    }

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
        counts = np.int32(priors[i] * 10000)
        arr_size = 10000
        print(np.sum(counts), arr_size)
        k = 0
        while np.sum(counts) != arr_size:
            counts[k] += 1
            k += 1
        samples = []
        for idx in range(len(priors[0])):
            samples.extend([idx] * counts[idx])
        df[i] = samples

        print(samples)
    input()

    joypy.joyplot(df, overlap=0, colormap=cm.OrRd_r, linecolor='b', linewidth=.5, hist=True, bins=len(priors[0]),
                  figsize=(4,6), x_range=list(range(len(priors[0]))))
    plt.xticks(list(range(len(priors[0]))))
    plt.title("Scenario distribution over iterations")
    plt.xlabel("Scenarios")
    plt.show()

    # final_belief = belief()
    # plt.bar(range(len(final_belief)), final_belief)
    # plt.title("final scenario distribution")
    # plt.show()
    # plt.clf()

    print(robust_policy.get_probs())
    #print(priors[-1])



def eval_policies():

    best_vfs = [ 7.,  14.,  10. , 11.,   7.,  15.,   6.,  13., 19.5]

    environment = RepeatedPrisonersDilemmaEnv(episode_length=3)
    tit_for_tat = TabularPolicy(environment, environment.tit_for_tat)
    coop_then_defect = TabularPolicy(environment, environment.cooperate_then_defect)

    found = TabularPolicy(environment, np.array([[0,1],
 [0, 1],
 [0., 1],
 [1., 0],
 [1 , 0],
 [0., 1],
 [1., 0],
 [1., 0],
 [0., 1],
 [1., 0],
 [0., 1],
 [1., 0],
 [0., 1],
 [1., 0],
 [0., 1],
 [0., 1],
 [1., 0],
 [1., 0],
 [1., 0],
 [1., 0],
 [1, 0]], dtype=np.float32)
                          )
    found2 = TabularPolicy(environment, np.zeros_like(found.action_logits))
    found2.action_logits[:, 1] = 1.


    d = {
        0: "collaborate",
        1: "defect"
    }
    for p in [tit_for_tat.action_logits, coop_then_defect.action_logits]:
        string = f'\n'
        for state in range(p.shape[0]):
            string += f"state {state} -> {d[np.argmax(p[state])]}\n"
        print(string)
        print()

    # do with policy types ?
    num_policies = 8

    bg_population = DeterministicPoliciesPopulation(environment)
    bg_population.build_population()

    np.random.seed(0)
    bg_population.policies = bg_population.policies[np.random.choice(len(bg_population.policies), num_policies)]

    test_scenario_distrib = Prior(num_policies + 1)
    test_scenario_distrib.initialize_uniformly()

    algo = PolicyIteration(tit_for_tat, environment, epsilon=5)
    tit_for_tat_vf , tit_for_tat_vfs = algo.policy_evaluation_for_prior(bg_population, test_scenario_distrib)
    tit_for_tat_regrets = [
        bv-v[environment.s0] for v, bv in zip(tit_for_tat_vfs, best_vfs)
    ]
    algo = PolicyIteration(coop_then_defect, environment, epsilon=5)
    coop_then_defect_vf , coop_then_defect_vfs = algo.policy_evaluation_for_prior(bg_population, test_scenario_distrib)
    coop_then_defect_regrets = [
        bv-v[environment.s0] for v, bv in zip(coop_then_defect_vfs, best_vfs)
    ]

    algo = PolicyIteration(found, environment, epsilon=5)
    found_vf , found_vfs = algo.policy_evaluation_for_prior(bg_population, test_scenario_distrib)
    found_regrets = [
        bv-v[environment.s0] for v, bv in zip(found_vfs, best_vfs)
    ]

    algo = PolicyIteration(found2, environment, epsilon=5)
    found2_vf , found2_vfs = algo.policy_evaluation_for_prior(bg_population, test_scenario_distrib)

    found2_regrets = [
        bv-v[environment.s0] for v, bv in zip(found2_vfs, best_vfs)
    ]

    print(tit_for_tat_vf[environment.s0], coop_then_defect_vf[environment.s0], found_vf[environment.s0], found2_vf[environment.s0])

    print(tit_for_tat_regrets, coop_then_defect_regrets, found_regrets, found2_regrets)

    print(np.mean(tit_for_tat_regrets), np.mean(coop_then_defect_regrets), np.mean(found_regrets), np.mean(found2_regrets))



def find_best(seed, episode_length, pop_size, max_check=2048, **kwargs):

    environment = RepeatedPrisonersDilemmaEnv(episode_length=episode_length)


    # robust_policy.initialize_randomly()

    # do with policy types ?
    num_policies = pop_size
    bg_population = DeterministicPoliciesPopulation(environment)
    bg_population.build_population(size=num_policies, seed=seed)

    policy = TabularPolicy(environment)
    policy.initialize_uniformly()
    algo = PolicyIteration(policy, environment, epsilon=3, learning_rate=1., lambda_=0.)


    deterministic_set = DeterministicPoliciesPopulation(environment)
    deterministic_set.build_population(size=max_check)
    p = Prior(dim=len(bg_population.policies)+1)
    p.initialize_uniformly()

    best_response_vfs = np.empty((len(bg_population.policies) + 1, policy.n_states), dtype=np.float32)
    best_responses = {}
    for p_id in range(len(bg_population.policies) + 1):
        best_response = TabularPolicy(environment)
        best_response.initialize_uniformly()
        p_algo = PolicyIteration(best_response, environment, learning_rate=1, epsilon=4)
        if p_id < len(bg_population.policies):
            scenario = bg_population.policies[p_id], (1, 0)
        else:
            scenario = best_response.get_probs(), (0.5, 0.5)
        for i in range(episode_length * 5):
            if p_id == len(bg_population.policies):
                scenario = best_response.get_probs(), (0.5, 0.5)
            vf = p_algo.policy_evaluation_for_scenario(scenario)

            p_algo.policy_improvement_for_scenario(scenario, vf)

        vf = p_algo.policy_evaluation_for_scenario(scenario)

        best_response_vfs[p_id] = vf
        best_responses[p_id] = best_response

    values = np.zeros((len(deterministic_set.policies), len(bg_population.policies)+1))
    for i, deterministic_policy in enumerate(deterministic_set.policies):
        policy.action_logits[:] = deterministic_policy

        _, pi_values = algo.policy_evaluation_for_prior(bg_population, p)
        values[i] = pi_values[:, environment.s0]

    best_policy = np.argmax(np.mean(values, axis=-1))

    return {
        "utility": np.mean(values[best_policy]),
        "regret": np.mean(best_response_vfs[:, environment.s0] - values[best_policy])
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='repeated_prisoners_experiment',
    )
    parser.add_argument("experiment", type=str, choices=["main", "unit_run", "eval", "find_best"], default="main")
    parser.add_argument("--policy_lr", type=float, default=1e-2)
    parser.add_argument("--prior_lr", type=float, default=1e-2)
    parser.add_argument("--use_regret", type=bool, default=False)
    parser.add_argument("--lambda_", type=float, default=0.)
    parser.add_argument("--sp", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=2)
    parser.add_argument("--pop_size", type=int, default=4)

    parser.add_argument("--n_seeds", type=int, default=50)




    args = parser.parse_args()

    if args.experiment == "unit_run":
        repeated_prisoners_experiment(
            args.policy_lr,
            args.prior_lr,
            args.use_regret,
            args.sp,
            args.lambda_,
            args.seed,
            args.episode_length,
            args.pop_size
        )
    elif args.experiment == "main":
        main(
            args.policy_lr,
            args.prior_lr,
            args.n_seeds,
            args.episode_length,
            args.pop_size
        )
    elif args.experiment == "eval":
        eval_policies()
    elif args.experiment == "find_best":
        find_best(args.seed, args.episode_length, args.pop_size)
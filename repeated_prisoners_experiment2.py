import itertools
import os
import pickle
import string
import time
from collections import defaultdict
from typing import Union
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joypy
from matplotlib import cm
from tqdm import tqdm

from background_population.bg_population import TabularBackgroundPopulation
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

from pyplot_utils import make_grouped_boxplot, make_grouped_plot


def main(policy_lr, prior_lr, lambda_, n_seeds=1, episode_length=10, pop_size=2, n_steps=1000,
         env_seed=0,
         plot_regret=True):



    approaches = [
        dict(
            scenario_distribution_optimization="Regret maximizing",
            use_regret=True,
            policy_lr=policy_lr,
            prior_lr=prior_lr,
            n_steps=n_steps,
            main_approach=True,
        ),
        dict(
            scenario_distribution_optimization="Utility minimizing",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=prior_lr,
            n_steps=n_steps,
            main_approach=False

        ),
        dict(
            scenario_distribution_optimization="Fixed uniform",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=0.,
            n_steps=n_steps,
            main_approach=False,
        ),
        dict(
            scenario_distribution_optimization="Self play",
            use_regret=False,
            policy_lr=policy_lr,
            prior_lr=0.,
            n_steps=n_steps,
            self_play=True,
            main_approach=False,
        ),
        dict(
            scenario_distribution_optimization="Random policy",
            use_regret=False,
            policy_lr=0.,
            prior_lr=0.,
            n_steps=2,
            main_approach=False,
        ),
    ]

    # lr_samples = np.logspace(-6, -3, 10)
    #
    # approaches = [
    #     dict(
    #         scenario_distribution_optimization=f"Regret maximizing beta_lr={lr:.0E}",
    #         use_regret=True,
    #         policy_lr=policy_lr,
    #         prior_lr=lr,
    #         n_steps=n_steps,
    #         n_states=n_states,
    #         n_actions=n_actions,
    #         history_window=history_window,
    #         true_solution=False
    #     )
    #     for lr in lr_samples]

    if plot_regret:
        whiskers = (50, 100)
        plot_type = "regret"
    else:
        whiskers = (0, 50)
        plot_type = "utility"
    name = f"prisoners_n_steps={n_steps}_env_seed={env_seed}_lr={policy_lr:.0E}_beta_lr={prior_lr:.0E}"

    all_jobs = []
    for seed in range(n_seeds):
        seeded_configs = [{
            "seed": seed,
            "lambda_": lambda_,
            "episode_length": episode_length,
            "pop_size": pop_size,
            "run_name": name,
            "job": prisoners_experiment_with_config #if idx < len(approaches)-1
            #else repeated_prisoners_best_solution_with_config
        } for idx in range(len(approaches))]

        for config, approach in zip(seeded_configs, approaches):
            config.update(**approach)
        all_jobs.extend(seeded_configs)

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # results[approach][train/test][metric][data list]

    with mp.Pool(os.cpu_count(), maxtasksperchild=1) as p:
        for ret, config in tqdm(
                zip(p.imap(run_job, all_jobs), all_jobs), total=len(all_jobs)
        ):
            ret["config"] = config
            approach = config["scenario_distribution_optimization"]
            for train_or_test, data in ret.items():
                for k, v in data.items():
                    results[approach][train_or_test][k].append(v)

    train_grouped_data = {}
    test_grouped_data = {}

    for approach, metrics in results.items():
        config = metrics.pop("config")

        run_data = {}

        for metric, data in metrics["train"].items():
            stacked = np.stack(data)
            meaned = np.mean(stacked, axis=0)
            run_data[metric] = meaned

        train_grouped_data[approach] = run_data

        # Test
        run_data = {}

        for metric, data in metrics["test"].items():

            if plot_regret:
                data = [d["regret"] for d in data]
            else:
                data = [d["utility"] for d in data]

            stacked = np.stack(data)
            meaned = np.mean(stacked, axis=0)
            run_data[metric] = meaned

        test_grouped_data[approach] = run_data

    make_grouped_plot(train_grouped_data, name=f"train_{name}")
    make_grouped_boxplot(test_grouped_data, name=f"boxplot_{plot_type}_{name}", whiskers=whiskers, plot_type=plot_type)

def run_job(config):
    job = config.pop("job")
    return job(config)

def prisoners_experiment_with_config(config):
    return prisoners_experiment(**config)

def prisoners_experiment(
        policy_lr=1e-3,
        prior_lr=1e-3,
        use_regret=False,
        self_play=False,
        lambda_=0.5,
        seed=0,
        episode_length=1,
        pop_size=None,
        n_steps=1000,
        main_approach=False,
        run_name="",
        **kwargs):

    np.random.seed(seed)

    environment = RepeatedPrisonersDilemmaEnv(episode_length=episode_length)

    robust_policy = Policy(environment)
    robust_policy.initialize_uniformly()
    #robust_policy.initialize_randomly()

    # do with policy types ?
    num_policies = pop_size
    print(pop_size)
    bg_population = DeterministicPoliciesPopulation(environment)
    bg_population.build_population(size=num_policies)

    algo = PolicyIteration(robust_policy, environment, epsilon=episode_length, learning_rate=policy_lr, lambda_=lambda_)

    # belief over worst teammate policy (all bg individuals and our self)
    belief = Prior(len(bg_population.policies)+1, learning_rate=prior_lr)
    #belief.initialize_randomly()
    if self_play:
        belief.initialize_certain(belief.dim - 1)
    else:
        #belief.initialize_randomly()print
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
        print(p_id)
        best_response = TabularPolicy(environment)
        best_response.initialize_uniformly()

        policy_history = [
            best_response.get_probs(),
            best_response.get_probs()
        ]

        p_algo = PolicyIteration(best_response, environment, learning_rate=1, epsilon=episode_length)
        if p_id < len(bg_population.policies):
            scenario = bg_population.policies[p_id], (1, 0)
        else:
            scenario = best_response.get_probs(), (0.5, 0.5)
        for i in range(episode_length * 5):
            policy_history.append(best_response.get_probs())
            old_best_response = policy_history.pop(0)

            if p_id == len(bg_population.policies):
                scenario = old_best_response, (0.5, 0.5)
            vf = p_algo.policy_evaluation_for_scenario(scenario)
            p_algo.policy_improvement_for_scenario(scenario, vf)

            if np.allclose(old_best_response, best_response.get_probs()):
                break

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
    worst_case_regrets = []
    policy_history = [
        TabularPolicy(environment, robust_policy.get_probs()),
        TabularPolicy(environment, robust_policy.get_probs())
                      ]

    #belief.beta_logits[:] = 0.3333, 0.3333, 0.3333

    for i in range(n_steps):

        policy_history.append(TabularPolicy(environment, robust_policy.get_probs()))
        previous_robust_policy = policy_history.pop(0)

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
            algo.exact_pg(bg_population, belief, vf, previous_copy=previous_robust_policy)
            belief.update_prior(regret_s0, regret=True)
            #algo.exact_regret_pg(bg_population, belief, vf, best_response_vfs)
        else:
            algo.exact_pg(bg_population, belief, vf, previous_copy=previous_robust_policy)
            belief.update_prior(vf_s0, regret=False)

        vfs.append(expected_vf[environment.s0])
        regrets.append(np.sum(regret_s0 * belief()))

        vf_scores.append(np.mean(vf_s0))
        regret_scores.append(np.mean(regret_s0))
        worst_case_regrets.append(np.max(regret_s0))

        # subprocess.call("clear")
        # print()
        print(f"--- Iteration {i} ---")
        # print()
        #
        # print("Test time score (utility, regret):", vf_scores[-1], regret_scores[-1])
        # print("w.r.t current prior (utility, regret):", vfs[-1], regrets[-1])
        # print("Utility per scenario:", vf_s0)
        # print("Regret per scenario", regret_s0)


        priors.append(belief().copy())
        a_probs = robust_policy.get_probs()
        #entropy = np.mean(np.sum(-a_probs * np.log(a_probs+1e-8), axis=-1))
        #print("Current distribution over scenarios :", belief())
        #print("Policy:", robust_policy.get_probs())

    argmax_actions = np.zeros_like(robust_policy.action_logits)
    argmax_actions[np.arange(len(argmax_actions)), np.argmax(robust_policy.action_logits, axis=-1)] = 1.
    argmax_policy = TabularPolicy(
        environment,
        argmax_actions
    )
    #robust_policy = argmax_policy

    #algo_p = PolicyIteration(argmax_policy, environment, epsilon=5)
    algo_p = PolicyIteration(robust_policy, environment, epsilon=episode_length)
    expected_vf, vf = algo_p.policy_evaluation_for_prior(bg_population, belief)
    vf_s0 = vf[:, environment.s0]
    all_regrets = best_response_vfs - vf
    regret_s0 = all_regrets[:, environment.s0]

    # print("wrt to learn distribution",
    #       np.sum(vf_s0*priors[-1], axis=0),
    #       np.sum(regret_s0 * priors[-1], axis=0)
    #       )
    # print()
    # print("Results:")
    #print("Test time score (utility, regret):", np.mean(vf_s0), np.mean(regret_s0))

    minimax_worst_case_distribution_path = run_name + "worst_case_distribution.pkl"
    if main_approach:
        minimax_worst_case_distribution = priors[-1]
        with open(minimax_worst_case_distribution_path, "wb+") as f:
            pickle.dump(minimax_worst_case_distribution, f)
    else:
        while not os.path.exists(minimax_worst_case_distribution_path):
            time.sleep(1)
        time.sleep(1)

        with open(minimax_worst_case_distribution_path, "rb") as f:
            minimax_worst_case_distribution = pickle.load(f)



    # EVALUATION
    print("Running evaluation...")

    test_results = {
        "r$\Sigma(\Mathcal{B}^\mathrm{train})$" : {"utility": vf_s0, "regret": regret_s0},
        "r$\Sigma^\mathrm{self-play}$": {"utility": vf_s0[-1:], "regret": regret_s0[-1:]}
    }

    samples = np.random.choice(len(minimax_worst_case_distribution), 2048, p=minimax_worst_case_distribution)
    test_results["r$\beta^*$"] = {
        "utility": vf_s0[samples],
        "regret" : regret_s0[samples]
    }


    # We sample 3 random test sets with 9 environments
    if episode_length > 1:

        test_background_population = DeterministicPoliciesPopulation(environment)
        test_background_population.build_population(pop_size*4)

        _, main_policy_vf = algo_p.policy_evaluation_for_prior(test_background_population, Prior(len(test_background_population.policies)+1, learning_rate=0))
        best_response_vfs = np.empty((len(test_background_population.policies) + 1, robust_policy.n_states), dtype=np.float32)
        for p_id in range(len(test_background_population.policies) + 1):
            best_response = TabularPolicy(environment)
            best_response.initialize_uniformly()
            p_algo = PolicyIteration(best_response, environment, learning_rate=1, epsilon=episode_length)
            policy_history = [
                best_response.get_probs(),
                best_response.get_probs()
            ]
            if p_id < len(test_background_population.policies):
                scenario = test_background_population.policies[p_id], (1, 0)
            else:
                scenario = best_response.get_probs(), (0.5, 0.5)
            for i in range(episode_length * 5):
                policy_history.append(best_response.get_probs())
                old_best_response = policy_history.pop(0)
                if p_id == len(test_background_population.policies):
                    scenario = old_best_response, (0.5, 0.5)
                vf = p_algo.policy_evaluation_for_scenario(scenario)

                p_algo.policy_improvement_for_scenario(scenario, vf)

                if np.allclose(old_best_response, best_response.get_probs()):
                    break

            vf = p_algo.policy_evaluation_for_scenario(scenario)

            best_response_vfs[p_id] = vf

        vf_s0 = main_policy_vf[:, environment.s0]
        regret_s0 = best_response_vfs[:, environment.s0] - vf_s0

        np.random.seed(4)
        for random_set_idx in range(3):
            scenario_idxs = np.random.choice(len(test_background_population.policies) + 1, size=9, replace=False)

            test_results[fr"$\Sigma^{{ {random_set_idx+1} }}$"] = {
                "utility": vf_s0[scenario_idxs],
                "regret" : regret_s0[scenario_idxs]
            }


        # we also test against the minimax distribution (TODO)
        # samples = np.random.choice(len(minimax), 512, p=minimax)
        # results["minimax"] = {
        #     "utility": vf_s0[samples],
        #     "regret" : regret_s0[samples]
        # }


        # lastly, We handpick a set, with:
        # a fully cooperating, defective policy, selfplay
        # cooperative = np.zeros_like(robust_policy.action_logits)
        # cooperative[:, 0] = 1
        # defective = np.zeros_like(robust_policy.action_logits)
        # defective[:, 1] = 1
        #
        # # And well known interesting policies
        #
        #
        # test_background_population = TabularBackgroundPopulation(environment)
        #
        # test_background_population.policies = np.stack([environment.cooperate_then_defect, environment.tit_for_tat, cooperative, defective])
        #
        # _, main_policy_vf = algo_p.policy_evaluation_for_prior(test_background_population, Prior(len(test_background_population.policies)+1, learning_rate=0))
        # best_response_vfs = np.empty((len(test_background_population.policies) + 1, robust_policy.n_states), dtype=np.float32)
        # for p_id in range(len(test_background_population.policies) + 1):
        #     best_response = TabularPolicy(environment)
        #     best_response.initialize_uniformly()
        #     p_algo = PolicyIteration(best_response, environment, learning_rate=1, epsilon=episode_length)
        #     if p_id < len(test_background_population.policies):
        #         scenario = test_background_population.policies[p_id], (1, 0)
        #     else:
        #         scenario = best_response.get_probs(), (0.5, 0.5)
        #     for i in range(episode_length * 5):
        #         if p_id == len(test_background_population.policies):
        #             scenario = best_response.get_probs(), (0.5, 0.5)
        #         vf = p_algo.policy_evaluation_for_scenario(scenario)
        #
        #         p_algo.policy_improvement_for_scenario(scenario, vf)
        #
        #     vf = p_algo.policy_evaluation_for_scenario(scenario)
        #
        #     best_response_vfs[p_id] = vf
        #
        # vf_s0 = main_policy_vf[:, environment.s0]
        # regret_s0 = best_response_vfs[:, environment.s0] - vf_s0
        #
        # results["handpicked"] = {
        #     "utility": vf_s0,
        #     "regret": regret_s0
        # }

    return {
        "train": {
            "regret": regret_scores,
            "learned distribution regret": regrets,
            "worst-case regret":worst_case_regrets,
        },
        "test": test_results
    }


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='repeated_prisoners_experiment',
    )
    parser.add_argument("experiment", type=str, choices=["main", "unit_run", "eval", "find_best"], default="main")
    parser.add_argument("--policy_lr", type=float, default=1e-2)
    parser.add_argument("--prior_lr", type=float, default=1e-2)
    parser.add_argument("--use_regret", type=bool, default=False)
    parser.add_argument("--lambda_", type=float, default=0.)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--sp", type=bool, default=False)
    parser.add_argument("--history_window", type=int, default=2)
    parser.add_argument("--n_states", type=int, default=2)
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=2)
    parser.add_argument("--pop_size", type=int, default=4)
    parser.add_argument("--n_seeds", type=int, default=50)

    parser.add_argument("--env_seed", type=int, default=0)




    args = parser.parse_args()

    if args.experiment == "unit_run":
        prisoners_experiment(
            args.policy_lr,
            args.prior_lr,
            args.use_regret,
            args.sp,
            args.lambda_,
            args.seed,
            args.episode_length,
            args.pop_size,
            args.n_steps,
            args.n_states,
            args.n_actions,
            args.history_window,
            args.env_seed,
        )
    elif args.experiment == "main":
        main(
            args.policy_lr,
            args.prior_lr,
            args.lambda_,
            args.n_seeds,
            args.episode_length,
            args.pop_size,
            args.n_steps,

        )

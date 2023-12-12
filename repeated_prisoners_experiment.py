import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joypy
from matplotlib import cm

from background_population.deterministic import DeterministicPoliciesPopulation
from beliefs.prior import Prior
from environments.repeated_prisoners import RepeatedPrisonersDilemmaEnv
from policies.policy import Policy
from policies.tabular_policy import TabularPolicy
from policy_iteration.algorithm import PolicyIteration
import argparse


def main(policy_lr, prior_lr, use_regret, self_play, lambda_):

    environment = RepeatedPrisonersDilemmaEnv(episode_length=3)

    np.random.seed(0)
    robust_policy = Policy(environment)
    robust_policy.initialize_uniformly()
    #robust_policy.initialize_randomly()

    # do with policy types ?
    num_policies = 8
    seed = 0
    bg_population = DeterministicPoliciesPopulation(environment)
    bg_population.build_population(size=num_policies, seed=seed)

    algo = PolicyIteration(robust_policy, environment, epsilon=4, learning_rate=policy_lr, lambda_=lambda_)

    # belief over worst teammate policy (all bg individuals and our self)
    belief = Prior(len(bg_population.policies)+1, learning_rate=prior_lr)
    #belief.initialize_randomly()
    if self_play:
        belief.initialize_certain(belief.dim - 1)
    else:
        belief.initialize_uniformly()

    vfs = []
    regret_scores = []
    vf_scores = []

    # Compute best responses for regret
    best_response_vfs = np.empty((len(bg_population.policies) + 1, robust_policy.n_states), dtype=np.float32)
    best_response = TabularPolicy(environment)
    p_belief = Prior(len(bg_population.policies) + 1)

    priors = []
    priors.append(belief().copy())
    best_responses = np.zeros_like(bg_population.policies)
    for p_id in range(len(bg_population.policies) + 1):
        best_response.initialize_uniformly()
        p_algo = PolicyIteration(best_response, environment, learning_rate=1, epsilon=10)
        if p_id < len(bg_population.policies):
            scenario = bg_population.policies[p_id], (1, 0)
        else:
            scenario = best_response.get_probs(), (0.5, 0.5)
        for i in range(10):
            vf = p_algo.policy_evaluation_for_scenario(scenario)

            p_algo.policy_improvement_for_scenario(scenario, vf)

        vf = p_algo.policy_evaluation_for_scenario(scenario)

        best_response_vfs[p_id] = vf
        best_responses[p_id] = best_response.get_probs()

    regrets = []
    for i in range(2000):

        expected_vf, vf = algo.policy_evaluation_for_prior(bg_population, belief)
        vf_s0 = vf[:, environment.s0]

        all_regrets = best_response_vfs - vf

        regret = all_regrets[:, environment.s0]

        if use_regret:
            belief.update_prior(regret, regret=True)
            algo.exact_pg(bg_population, belief, all_regrets, regret=True, best_responses=best_responses)
        else:
            belief.update_prior(vf_s0, regret=False)
            algo.exact_pg(bg_population, belief, vf, regret=False)




        vfs.append(expected_vf[environment.s0])
        regrets.append(np.sum(regret * belief()))

        vf_scores.append(np.mean(vf_s0))
        regret_scores.append(np.mean(regret))

        print(vf_scores[-1], regret_scores[-1])
        priors.append(belief().copy())
        a_probs = robust_policy.get_probs()
        entropy = np.mean(np.sum(-a_probs * np.log(a_probs+1e-8), axis=-1))
        print(belief(), entropy)
        print(regret)
        #print(robust_policy.get_probs())
    return


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

    d = {
        0: "collaborate",
        1: "defect"
    }
    for i, p in enumerate(bg_population.policies):
        string = f'policy [{i}] \n '
        for state in range(p.shape[0]):
            string += f"state {state} -> {d[np.argmax(p[state])]}\n"
        print(string)
        print()




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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                    prog='repeated_prisoners_experiment',
    )
    parser.add_argument("experiment", type=str, choices=["main", "eval"], default="main")
    parser.add_argument("--policy_lr", type=float, default=1e-2)
    parser.add_argument("--prior_lr", type=float, default=1e-2)
    parser.add_argument("--use_regret", type=bool, default=False)
    parser.add_argument("--lambda_", type=float, default=1e-3)
    parser.add_argument("--sp", type=bool, default=False)




    args = parser.parse_args()

    if args.experiment == "main":
        main(args.policy_lr, args.prior_lr, args.use_regret, args.sp, args.lambda_)
    elif args.experiment == "eval":
        eval_policies()
import numpy as np
from matplotlib import pyplot as plt

from background_population.deterministic import DeterministicPoliciesPopulation
from beliefs.prior import Prior
from environments.repeated_prisoners import RepeatedPrisonersDilemmaEnv
from policies.policy import Policy
from policies.tabular_policy import TabularPolicy
from policy_iteration.algorithm import PolicyIteration


def main():

    environment = RepeatedPrisonersDilemmaEnv(episode_length=3)

    robust_policy = Policy(environment)
    robust_policy.initialize_uniformly()
    #robust_policy.initialize_randomly()

    # do with policy types ?
    num_policies = 8

    bg_population = DeterministicPoliciesPopulation(environment)
    bg_population.build_population()

    np.random.seed(0)
    bg_population.policies = bg_population.policies[np.random.choice(len(bg_population.policies), num_policies)]

    algo = PolicyIteration(robust_policy, environment, epsilon=1e-4, learning_rate=1e-2)

    # belief over worst teammate policy (all bg individuals and our self)
    belief = Prior(len(bg_population.policies)+1, learning_rate=1e-3)
    belief.initialize_uniformly()

    vfs = []
    scores = []


    use_regret = True

    if use_regret:
        # Compute best responses for regret
        best_response_vfs = np.empty((len(bg_population.policies) + 1,), dtype=np.float32)
        best_response = TabularPolicy(environment)
        p_belief = Prior(len(bg_population.policies) + 1)

        for p_id in range(len(bg_population.policies) + 1):
            best_response.initialize_uniformly()
            p_algo = PolicyIteration(best_response, environment, learning_rate=1)
            p_belief.initialize_certain(idx=p_id)

            for i in range(10):
                vf, tmp = p_algo.policy_evaluation_for_prior(bg_population, p_belief)

                p_algo.policy_improvement(bg_population, p_belief, vf)

            vf, tmp = p_algo.policy_evaluation_for_prior(bg_population, p_belief)

            print(best_response.get_params())
            input()

            best_response_vfs[p_id] = vf[environment.s0]

        regrets = []
        for i in range(5000):
            print(i)

            expected_vf, vf = algo.policy_evaluation_for_prior(bg_population, belief)
            vf_s0 = vf[:, environment.s0]
            algo.exact_pg(bg_population, belief, expected_vf)

            regret = best_response_vfs - vf_s0
            belief.update_prior(regret, regret=use_regret)

            vfs.append(expected_vf[environment.s0])
            regrets.append(np.sum(regret * belief()))
            scores.append(np.mean(vf_s0))

            print(robust_policy.get_probs())
            print(belief())
            print(regret)

        plt.plot(regrets)
        plt.show()
        plt.clf()

    else:
        for i in range(1000):
            print(i)
            print(robust_policy.get_probs())

            expected_vf, vf = algo.policy_evaluation_for_prior(bg_population, belief)
            vf_s0 = vf[:, environment.s0]
            algo.exact_pg(bg_population, belief, expected_vf)
            belief.update_prior(vf_s0, regret=use_regret)

            vfs.append(expected_vf[environment.s0])
            scores.append(np.mean(vf_s0))


    plt.plot(vfs)
    plt.show()
    plt.clf()
    final_belief = belief()
    plt.bar(range(len(final_belief)), final_belief)
    plt.show()
    plt.clf()
    plt.plot(scores)
    plt.show()

    print(robust_policy.get_probs())
    print(final_belief)

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

    print(scores[-1])


def eval_policies():
    environment = RepeatedPrisonersDilemmaEnv(episode_length=3)
    tit_for_tat = environment.tit_for_tat
    coop_then_defect = environment.cooperate_then_defect

    found = np.array([[0.55 ,0.45],
 [0, 1],
 [0., 1],
 [1., 0],
 [1 ,0],
 [1., 0],
 [1. ,0],
 [1. ,0],
 [1. ,0],
 [1. ,0],
 [1. ,0],
 [1. ,0],
 [1., 0],
 [1., 0],
 [1., 0],
 [1., 0],
 [1., 0],
 [1., 0],
 [1., 0],
 [1., 0],
 [1., 0]], dtype=np.float32)

    found2 = np.zeros_like(found)
    found2[:, 1] = 1.

    d = {
        0: "collaborate",
        1: "defect"
    }
    for p in [tit_for_tat, coop_then_defect]:
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

    algo = PolicyIteration(tit_for_tat, environment, epsilon=1e-4)
    tit_for_tat_vf , _ = algo.policy_evaluation_for_prior(bg_population, test_scenario_distrib)

    algo = PolicyIteration(coop_then_defect, environment, epsilon=1e-4)
    coop_then_defect_vf , _ = algo.policy_evaluation_for_prior(bg_population, test_scenario_distrib)


    algo = PolicyIteration(found, environment, epsilon=1e-4)
    found_vf , _ = algo.policy_evaluation_for_prior(bg_population, test_scenario_distrib)

    algo = PolicyIteration(found2, environment, epsilon=1e-4)
    found2_vf , _ = algo.policy_evaluation_for_prior(bg_population, test_scenario_distrib)

    print(tit_for_tat_vf[environment.s0], coop_then_defect_vf[environment.s0], found_vf[environment.s0], found2_vf[environment.s0])


if __name__ == '__main__':

    main()
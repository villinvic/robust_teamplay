import multiprocessing as mp

from beliefs.rllib_scenario_distribution import ScenarioSet
from rllib_experiments.learn_best_responses import main
import fire
import os
import numpy as np

num_proc = os.cpu_count() - 2


def get_policy_id(p):
    if isinstance(int, np.int64):
        p = f"background_deterministic_{p}"
    else:
        p = f"background_{p}"
    return p
def run(
        bg_policies=[0],
        env_seed=0,
        n_states=5,
        n_actions=3,
        num_players=2,
        episode_length=64,
        history_length=2,
        full_one_hot=True
):

    named_bg_policies = [
        get_policy_id(p) for p in bg_policies
    ]

    run_dicts = [
        dict(
            bg_policies=bg_policies,
            env_seed=env_seed,
            n_states=n_states,
            n_actions=n_actions,
            num_players=num_players,
            episode_length=episode_length,
            history_length=history_length,
            full_one_hot=full_one_hot,
            relevant_scenarios=[relevant_scenario]
        )
        for relevant_scenario in ScenarioSet(num_players, named_bg_policies).scenario_list
    ]

    with mp.Pool(num_proc, maxtasksperchild=1) as p:
        p.map(main, run_dicts)

if __name__ == '__main__':

    fire.Fire(run)


from typing import List, Dict

import fire
from ray.rllib import Policy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from rich.progress import Progress

from beliefs.rllib_scenario_distribution import Scenario, ScenarioSet
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy

import multiprocessing as mp
import os
import numpy as np

from rllib_experiments.configs import get_env_config

NUM_CPU = os.cpu_count() - 1

def shuffle_dict(input_dict):
    # Convert the dictionary into a list of key-value pairs
    items = list(input_dict.items())
    # Shuffle the list
    np.random.shuffle(items)
    # Convert the shuffled list back into a dictionary
    shuffled_dict = dict(items)
    return shuffled_dict


def run_episode(policies: Dict[str, Policy], env_maker, n_episodes = 10):
    env = env_maker()
    n_focal = len([
        1 for policy_id in policies if "background" not in policy_id
    ])

    per_capita_focal_mean = 0

    for i in range(n_episodes):
        policies = shuffle_dict(policies)
        agent_to_policy = {
            agent_id: policy_id
            for agent_id, policy_id in zip(env._agent_ids, policies.keys())
        }
        done = False
        obs, _ = env.reset()
        while not done:
            actions = {agent_id: policies[agent_to_policy[agent_id]].compute_single_action(obs[agent_id], state=None)
                       for agent_id in policies.keys()
                       }
            obs, rewards, dones, truncs, _ = env.step(actions)

            done = dones["__all__"] or truncs["__all__"]

            timestep_focal_rewards = [
                r for agent_id, r in rewards if not "background" in agent_to_policy[agent_id]
            ]
            per_capita_focal_mean += sum(timestep_focal_rewards) / n_focal


    return {
        "focal_per_capita_mean" : per_capita_focal_mean / n_episodes
    }


class PolicyCkpt:
    """
    Can either be a deterministic policy,
    or a named policy
    """

    NAMED_POLICY_PATH = "data/policies/{name}"
    def __init__(self, name):

        self.name = name

        if "deterministic" in name:
            _, policy_seed = name.split("_")

            def make(environment):

                return RLlibDeterministicPolicy(
                    environment.action_space[0],
                    environment.action_space[0],
                    {"_disable_preprocessor_api": True},
                    seed = int(policy_seed)
                )
        elif name == "Random":

            def make(environment):
                return RandomPolicy(
                    environment.action_space[0],
                    environment.action_space[0],
                    {},
                )

        else:
            def make(environment):
                return Policy.from_checkpoint(self.NAMED_POLICY_PATH.format(name=name))

        self.make = make


class Benchmarking:
    """
    We want to benchmark policies in various test sets
    """
    def __init__(self, policies: List[str], test_sets: List[Scenario]):
        pass


class Evaluation:
    """
    feed env and name of the test set
    loads required policies
    """

    TEST_SET_PATH = "data/test_sets/{env}/name"

    def __init__(self, test_set: str, env: str, env_config: Dict,
                 eval_config: Dict):


        self.test_set_name = test_set
        self.test_set = ScenarioSet.from_YAML(Evaluation.TEST_SET_PATH.format())
        self.env_config = get_env_config(env)(**env_config)
        self.env_maker = self.env_config.get_maker()
        self.environment = self.env_maker()

        self.eval_config = eval_config

    def eval_policy_on_scenario(self, policy_name, scenario_name):

        scenario = self.test_set[scenario_name]

        focal_policy = PolicyCkpt(policy_name).make(self.environment)
        focal_policies = {
            f"name_{i}": focal_policy
            for i in scenario.num_copies
        }
        background_policies = {
            f"background_{bg_policy_name}": PolicyCkpt(bg_policy_name).make(self.environment)
            for bg_policy_name in scenario.background_policies
        }

        policies = {
            **focal_policies,
            **background_policies
        }
        return {scenario_name: run_episode(policies, self.env_maker, n_episodes=self.eval_config["n_episodes"])}


    def evaluate_policy(self, policy_name):

        jobs = [
            (policy_name, scenario_name)
            for scenario_name in self.test_set.scenario_list
        ]

        res = {}
        with Progress() as progress:
            task = progress.add_task(f"[green]Evaluating {policy_name} on test set {self.test_set_name}", total=len(jobs))
            with mp.Pool(np.minimum([NUM_CPU, len(self.test_set)]), maxtasksperchild=1) as p:
                for out in p.imap_unordered(self.eval_policy_on_scenario, jobs):
                    res.update(**out)
                    progress.update(task, advance=1)

        return {
            policy_name: res
        }


def run(
        policies=["deterministic_0"],
        sets=["deterministic_set_0"],
        env="RandomPOMDP",
):
    # TODO : for each set, save the stats in data/



if __name__ == '__main__':

    fire.Fire(run)

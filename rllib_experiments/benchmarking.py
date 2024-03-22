import pathlib
from typing import List, Dict

import fire
from ray.rllib import Policy, SampleBatch
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.models.preprocessors import get_preprocessor
from rich.progress import Progress
import yaml

from beliefs.rllib_scenario_distribution import Scenario, ScenarioSet
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy

import multiprocessing as mp
import os
import numpy as np

from rllib_experiments.configs import get_env_config
import ray
import logging

ray.logger.setLevel(logging.ERROR)

NUM_CPU = os.cpu_count() - 2

def shuffle_dict(input_dict):
    # Convert the dictionary into a list of key-value pairs
    items = list(input_dict.items())
    # Shuffle the list
    np.random.shuffle(items)
    # Convert the shuffled list back into a dictionary
    shuffled_dict = dict(items)
    return shuffled_dict


def run_episode(policies: Dict[str, Policy], env, n_episodes = 10):

    n_focal = len([
        1 for policy_id in policies if "background" not in policy_id
    ])

    preprocessor = get_preprocessor(env.observation_space[0])(
        env.observation_space[0]
    )

    episodic_focal_per_capita = []

    for i in range(n_episodes):
        policies = shuffle_dict(policies)
        agent_to_policy = {
            agent_id: policy_id
            for agent_id, policy_id in zip(env._agent_ids, policies.keys())
        }
        states = {
            agent_id: policies[agent_to_policy[agent_id]].get_initial_state() for agent_id in env._agent_ids
        }

        done = False
        obs, _ = env.reset()
        focal_per_capita = 0
        while not done:
            actions = {}
            for agent_id in env._agent_ids:
                policy_id= agent_to_policy[agent_id]

                input_dict = {
                    SampleBatch.OBS: preprocessor.transform(obs[agent_id])[np.newaxis],
                }
                for i, s in enumerate(states[agent_id]):
                    input_dict[f"state_in_{i}"] = s

                action, next_state, _ = policies[policy_id].compute_actions_from_input_dict(
                input_dict,
            )
                states[agent_id] = next_state
                actions[agent_id] = action[0]

            obs, rewards, dones, truncs, _ = env.step(actions)

            done = dones["__all__"] or truncs["__all__"]

            timestep_focal_rewards = [
                r for agent_id, r in rewards.items() if not "background" in agent_to_policy[agent_id]
            ]
            focal_per_capita += sum(timestep_focal_rewards) / n_focal
        episodic_focal_per_capita.append(focal_per_capita)


    return {
        "focal_per_capita_mean": float(np.mean(episodic_focal_per_capita)),
        "focal_per_capita_ste": float(np.std(episodic_focal_per_capita) / np.sqrt(n_episodes))
    }


class PolicyCkpt:
    """
    Can either be a deterministic policy,
    or a named policy
    """
    NAMED_POLICY_PATH = str(pathlib.Path(__file__).parent.resolve()) + "/../data/policies/{env}/{name}"

    def __init__(self, name, env_name=""):

        self.name = name
        self.env_name = env_name

        if "deterministic" in name:
            try:
                _, policy_seed = name.split("_")
            except ValueError as e:
                raise ValueError(f"Malformed policy name: {name}.") from e

            def make(environment):

                return RLlibDeterministicPolicy(
                    environment.observation_space[0],
                    environment.action_space[0],
                    {"_disable_preprocessor_api": True},
                    seed = int(policy_seed)
                )
        elif name == "random":

            def make(environment):
                return RandomPolicy(
                    environment.observation_space[0],
                    environment.action_space[0],
                    {},
                )

        else:
            def make(environment):
                return Policy.from_checkpoint(PolicyCkpt.NAMED_POLICY_PATH.format(name=name, env=env_name))

        self.make = make


def eval_policy_on_scenario(
        packed_args
):

    (scenario_name,
    policy_name,
    test_set,
    env_config) = packed_args

    scenario = test_set[scenario_name]
    environment = env_config.get_maker()()
    env_id = env_config.get_env_id()
    focal_policy = PolicyCkpt(policy_name, env_id).make(environment)
    focal_policies = {
        f"{policy_name}_{i}": focal_policy
        for i in range(scenario.num_copies)
    }
    background_policies = {
        f"background_{bg_policy_name}": PolicyCkpt(bg_policy_name, env_id).make(environment)
        for bg_policy_name in scenario.background_policies
    }

    policies = {
        **focal_policies,
        **background_policies
    }
    return {scenario_name: run_episode(policies, environment, n_episodes=test_set.eval_config["num_episodes"])}



class Benchmarking:
    """
    We want to benchmark policies in various test sets
    # TODO : we need this if benchmarking is slow
    """
    def __init__(self,
                 policies: List[str],
                 test_sets: List[str],
                 env: str,
                 env_config: Dict,
                 eval_config: Dict
                 ):

        set_evaluations = [
            Evaluation(test_set, env, env_config=env_config, eval_config=eval_config)
            for test_set in test_sets
        ]

        tasks = []

        for evaluation in set_evaluations:
            for policy in policies:
                tasks.extend(evaluation.get_tasks(policy))



class Evaluation:
    """
    feed env and name of the test set
    loads required policies
    """
    EVAL_PATH = str(pathlib.Path(__file__).parent.resolve()) + "/../data/evaluation/{env}/{set_name}.YAML"

    def __init__(self, test_set: str, env: str, env_config: Dict):

        self.test_set_name = test_set
        self.env_name = env,
        self.env_config = env_config
        self.env_config = get_env_config(env)(**env_config)
        self.environment = self.env_config.get_maker()()
        self.test_set = ScenarioSet.from_YAML(
            ScenarioSet.TEST_SET_PATH.format(env=self.env_config.get_env_id(), set_name=test_set)
        )

    def eval_policy_on_scenario(self, policy_name, scenario_name):

        scenario = self.test_set[scenario_name]
        env_maker = self.env_config.get_maker()
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
        return {scenario_name: run_episode(policies, env_maker, n_episodes=self.eval_config["n_episodes"])}


    def evaluate_policy(self, policy_name):

        jobs = [
            (scenario_name, policy_name, self.test_set, self.env_config)
            for scenario_name in self.test_set.scenario_list
        ]

        res = {}
        with Progress() as progress:
            task = progress.add_task(f'[green]Evaluating Policy "{policy_name}" on test set "{self.test_set_name}"', total=len(jobs))

            with mp.Pool(np.minimum(NUM_CPU, len(self.test_set)), maxtasksperchild=1) as p:
                for out in p.imap_unordered(eval_policy_on_scenario, jobs):
                    res.update(**out)
                    progress.update(task, advance=1)

        scores = np.array([result["focal_per_capita_mean"] for result in res.values()])
        score_stes = np.array([result["focal_per_capita_ste"] for result in res.values()])

        if "distribution" in self.test_set.eval_config:
            expected_utility = float(np.sum(np.array(self.test_set.eval_config["distribution"]) * scores))
            overall_ste = float(np.sum(np.array(self.test_set.eval_config["distribution"]) * score_stes))
        else:
            expected_utility = float(np.mean(scores))
            overall_ste = float(np.mean(score_stes))

        out_dict = {
            "overall_score": expected_utility,
            "overall_standard_error": overall_ste,
            "per_scenario": res
        }
        return {
            policy_name: out_dict
        }

    @staticmethod
    def run_jobs(jobs):


        res = {}
        with mp.Pool(np.minimum(NUM_CPU, len(jobs)), maxtasksperchild=1) as p:
            for out in p.imap_unordered(eval_policy_on_scenario, jobs):
                res.update(**out)

        # TODO if needed...


    def load(self):
        path = Evaluation.EVAL_PATH.format(env=self.env_config.get_env_id(),
                          set_name=self.test_set_name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                evaluation = yaml.safe_load(f)
            if evaluation is None:
                evaluation = {}
        else:
            evaluation = {}
            parent_path = os.sep.join(path.split(os.sep)[:-1])
            os.makedirs(parent_path, exist_ok=True)
            print("Created directory:", parent_path)
            with open(path, 'w') as f:
                yaml.safe_dump(evaluation, f)

        return evaluation

    def save(self, evaluation):
        path = Evaluation.EVAL_PATH.format(env=self.env_config.get_env_id(),
                          set_name=self.test_set_name)

        with open(path, 'w') as f:
            yaml.safe_dump(
                evaluation
                , f)


def run(
        policies=["deterministic_0", "deterministic_1", "random"],
        sets=["deterministic_set_0"],
        env="RandomPOMDP",
        **env_config
):

    # TODO : for each set, save the stats in data/
    for test_set in sets:
        evaluation = Evaluation(test_set, env, env_config=env_config)
        set_eval = evaluation.load()

        past_evaluations = list(set_eval.keys())
        
        
        for policy in policies:
            # We do not want to remove existing scores, probably

            if policy not in past_evaluations:
                set_eval.update(
                    **evaluation.evaluate_policy(policy)
                )
                evaluation.save(set_eval)






if __name__ == '__main__':

    fire.Fire(run)
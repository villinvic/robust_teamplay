import itertools
import os.path
import pathlib
import pickle
import queue
from collections import deque, defaultdict, ChainMap
from copy import copy, deepcopy
from functools import partial
from typing import Dict, Tuple, Union, Optional, List

import numpy as np
import ray
import yaml
from ray.rllib import Policy, SampleBatch, BaseEnv
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.typing import ResultDict, AgentID, PolicyID
from ray.rllib.algorithms.ppo.ppo import PPOConfig

from beliefs.prior import project_onto_simplex
from utils import SmoothMetric


def distribution_to_hist(distrib, precision=10000):
    return np.random.choice(len(distrib), size=(precision,), p=distrib)


class BackgroundFocalSGDA(DefaultCallbacks):

    def __init__(self, scenarios):
        super().__init__()

        self.beta: ScenarioDistribution = None
        self.scenarios = scenarios

    def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:

        self.beta = ScenarioDistribution(algorithm)
        del self.scenarios

        if not self.beta.config.learn_best_responses_only:

            setattr(algorithm, "base_cleanup", algorithm.cleanup)

            def on_algorithm_save(algo):

                # Dump learned distribution as test_set
                test_set_name = ""
                if not self.beta.config.self_play:
                    if self.beta.config.beta_lr == 0.:
                        test_set_name = "train_set_uniform"
                    elif self.beta.config.use_utility:
                        test_set_name = "train_set_maximin_utility_distribution"
                    else:
                        test_set_name = "train_set_minimax_regret_distribution"

                    dump_path = ScenarioSet.TEST_SET_PATH.format(
                        env=self.beta.config.env,
                        set_name=test_set_name,
                    )

                    self.beta.scenarios.to_YAML(
                        dump_path,
                        eval_config={
                            "distribution": [
                                float(np.mean(p)) for p in np.stack(list(self.beta.past_betas), axis=1)
                            ],
                            "num_episodes": 1000
                        }
                    )
                algo.base_cleanup()

            algorithm.cleanup = on_algorithm_save.__get__(algorithm, type(algorithm))

    def on_postprocess_trajectory(
            self,
            *,
            worker: "RolloutWorker",
            episode: Episode,
            agent_id: AgentID,
            policy_id: PolicyID,
            policies: Dict[PolicyID, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
            **kwargs,
    ) -> None:
        """
        Swap rewards to mean focal per capita return
        """

        # content of original_batches changes when connectors are disbaled
        # focal_rewards = [
        #     batch[SampleBatch.REWARDS] for agent_id, (_, batch) in original_batches.items()
        #     if "background" not in episode._agent_to_policy[agent_id]
        # ]

        focal_rewards = [
            batch[SampleBatch.REWARDS] for agent_id, (policy_id, policy_cls, batch) in original_batches.items()
            if "background" not in policy_id
        ]

        mean_focal_per_capita = sum(focal_rewards) / len(focal_rewards)
        postprocessed_batch[SampleBatch.REWARDS][:] = mean_focal_per_capita


    def on_episode_created(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        env_index: int,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:

        # Force policies to be selected through the mapping function now
        for agent_id in base_env.get_agent_ids():
            episode.policy_for(agent_id)

        sub_env = base_env.get_sub_environments(as_dict=True)[env_index]

        policies = list(episode._agent_to_policy.values())
        scenario_name = Scenario.get_scenario_name(policies)
        print(self.scenarios)
        scenario_id = self.scenarios.scenario_to_id[scenario_name]
        setattr(episode, "policies", policies)
        setattr(episode, "scenario", scenario_name)
        setattr(sub_env, "current_scenario_id", scenario_id)

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        focal_rewards = [
            episode.agent_rewards[agent_id, policy_id] for agent_id, policy_id in episode.agent_rewards
            if "background" not in policy_id
        ]
        episodic_mean_focal_per_capita = sum(focal_rewards) / len(focal_rewards)

        episode.custom_metrics[f"{episode.scenario}_utility"] = episodic_mean_focal_per_capita

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        self.beta.update(result)


class ScenarioSet:
    TEST_SET_PATH = str(pathlib.Path(__file__).parent.resolve()) + "/../data/test_sets/{env}/{set_name}.YAML"

    def __init__(self, scenarios: Dict[str, "Scenario"] = None, eval_config=None):
        self.scenarios = {}
        self.scenario_list = []
        self.eval_config = {}
        self.scenario_to_id = {}

        if scenarios is not None:
            self.scenarios = scenarios
            self.scenario_list = list(scenarios.keys())
            self.scenario_to_id = {
                scenario_name: i for i, scenario_name in enumerate(self.scenario_list)
            }

        if eval_config is not None:
            self.eval_config.update(**eval_config)

    def build_from_population(self, num_players, background_population):
        del self.scenario_list
        self.scenarios = {}

        for num_copies in range(1, num_players + 1):

            for background_policies in itertools.combinations_with_replacement(background_population,
                                                                               num_players - num_copies):
                policies = (Scenario.MAIN_POLICY_ID,) + (Scenario.MAIN_POLICY_COPY_ID,) * (
                        num_copies - 1) + background_policies

                scenario_name = Scenario.get_scenario_name(policies)

                self.scenarios[scenario_name] = Scenario(num_copies, list(background_policies))

        self.scenario_list = np.array(list(self.scenarios.keys()))
        self.scenario_to_id = {
            scenario_name: i for i, scenario_name in enumerate(self.scenario_list)
        }

    def sample_scenario(self, distribution):
        return np.random.choice(self.scenario_list, p=distribution)

    def __getitem__(self, item):
        return self.scenarios[item]

    def __len__(self):
        return len(self.scenario_list)

    def split(self, n=None):
        if n is None:
            n = len(self.scenario_list)

        subsets = []
        for sublist in np.split(self.scenario_list, n):
            subset = copy(self)
            subset.scenario_list = list(sublist)
            subsets.append(subset)

        return subsets

    @classmethod
    def from_YAML(cls, path):
        """
        :param path: path of the yaml
        :return: the list of scenarios contained in there
        """

        with open(path, 'r') as f:
            d = yaml.safe_load(f)

        eval_config = d.get("eval_config", {"num_episodes": 1000})

        scenarios = {
            scenario_name: Scenario(scenario_config["focal"], scenario_config["background"])
            for scenario_name, scenario_config in d["scenarios"].items()
        }
        return cls(scenarios=scenarios, eval_config=eval_config)

    def to_YAML(self, path: str, eval_config: Dict = None):

        if eval_config is not None:
            eval_config.update(**self.eval_config)
        else:
            eval_config = self.eval_config

        scenario_set = {
            "scenarios": {

                scenario_name: {
                    "focal": scenario.num_copies,
                    "background": [policy_id.removeprefix("background_") for policy_id in scenario.background_policies]
                }
                for scenario_name, scenario in self.scenarios.items()
            },

            "eval_config": eval_config
        }

        parent_path = os.sep.join(path.split(os.sep)[:-1])
        if not os.path.exists(parent_path):
            print("Created directory:", parent_path)
            os.makedirs(parent_path, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(scenario_set, f,
                           default_flow_style=None,
                           width=50, indent=4
                           )


#
# class PPOBFSGDAConfig(PPOConfig):
#     def __init__(self, algo_class=None):
#         super().__init__(algo_class=algo_class)
#
#         self.beta_lr = 5e-2
#         self.beta_eps = 1e-2
#
#         self.beta_smoothing = 1000
#         self.copy_weights_freq = 5
#         self.best_response_timesteps_max = 1_000_000
#
#         self.use_utility = False
#         self.self_play = False
#         self.learn_best_responses_only = True
#
#         self.best_response_utilities_path = os.getcwd() +  "/data/best_response_utilities/{env_name}.pkl"
#
#         # TODO if we have deep learning bg policies:
#         # Use our PolicyCkpt class
#         self.background_population_path = None
#
#         self.callbacks_class = BackgroundFocalSGDA
#         self.scenarios = None
#
#     def training(
#             self,
#             *,
#             beta_lr: Optional[float] = NotProvided,
#             beta_smoothing: Optional[float] = NotProvided,
#             use_utility: Optional[bool] = NotProvided,
#             self_play: Optional[bool] = NotProvided,
#             copy_weights_freq: Optional[int] = NotProvided,
#             best_response_timesteps_max: Optional[int] = NotProvided,
#             best_response_utilities_path: Optional[str] = NotProvided,
#             learn_best_responses_only: Optional[bool] = NotProvided,
#             beta_eps: Optional[float] = NotProvided,
#             scenarios: ScenarioSet = NotProvided,
#             **kwargs,
#     ) -> "PPOConfig":
#
#         super().training(**kwargs)
#         if beta_lr is not NotProvided:
#             self.beta_lr = beta_lr
#         if beta_smoothing is not NotProvided:
#             self.beta_smoothing = beta_smoothing
#         if use_utility is not NotProvided:
#             self.use_utility = use_utility
#         if self_play is not NotProvided:
#             self.self_play = self_play
#         if copy_weights_freq is not NotProvided:
#             self.copy_weights_freq = copy_weights_freq
#         if learn_best_responses_only is not NotProvided:
#             self.learn_best_responses_only = learn_best_responses_only
#         if best_response_utilities_path is not NotProvided:
#             self.best_response_utilities_path = best_response_utilities_path
#         if beta_eps is not NotProvided:
#             self.beta_eps = beta_eps
#
#         if best_response_timesteps_max is not NotProvided:
#             self.best_response_timesteps_max = best_response_timesteps_max
#
#         assert scenarios is not NotProvided, "You must provide an initial scenario set."
#         self.scenarios = scenarios
#
#         return self


def make_bf_sgda_config(cls) -> "BFSGDAConfig":
    class BFSGDAConfig(cls):
        def __init__(self):
            super().__init__()

            self.beta_lr = 5e-2
            self.beta_eps = 1e-2

            self.beta_smoothing = 1000
            self.copy_weights_freq = 5
            self.copy_history_len = 30
            self.best_response_timesteps_max = 1_000_000

            self.use_utility = False
            self.self_play = False
            self.learn_best_responses_only = True

            self.best_response_utilities_path = os.getcwd() + "/data/best_response_utilities/{env_name}.YAML"

            # TODO if we have deep learning bg policies:
            self.background_population_path = None


            # Must be specified in the training config.
            self.scenarios = None

        def training(
                self,
                *,
                beta_lr: Optional[float] = NotProvided,
                beta_smoothing: Optional[float] = NotProvided,
                use_utility: Optional[bool] = NotProvided,
                self_play: Optional[bool] = NotProvided,
                copy_weights_freq: Optional[int] = NotProvided,
                best_response_timesteps_max: Optional[int] = NotProvided,
                best_response_utilities_path: Optional[str] = NotProvided,
                learn_best_responses_only: Optional[bool] = NotProvided,
                copy_history_len: Optional[int] = NotProvided,

                beta_eps: Optional[float] = NotProvided,
                scenarios: ScenarioSet = NotProvided,
                **kwargs,
        ) -> "PPOConfig":

            super().training(**kwargs)
            if beta_lr is not NotProvided:
                self.beta_lr = beta_lr
            if beta_smoothing is not NotProvided:
                self.beta_smoothing = beta_smoothing
            if use_utility is not NotProvided:
                self.use_utility = use_utility
            if self_play is not NotProvided:
                self.self_play = self_play
            if copy_weights_freq is not NotProvided:
                self.copy_weights_freq = copy_weights_freq
            if copy_history_len is not NotProvided:
                self.copy_history_len = copy_history_len
            if learn_best_responses_only is not NotProvided:
                self.learn_best_responses_only = learn_best_responses_only
            if best_response_utilities_path is not NotProvided:
                self.best_response_utilities_path = best_response_utilities_path
            if beta_eps is not NotProvided:
                self.beta_eps = beta_eps

            if best_response_timesteps_max is not NotProvided:
                self.best_response_timesteps_max = best_response_timesteps_max

            assert scenarios is not NotProvided, "You must provide an initial scenario set."
            self.scenarios = scenarios
            self.callbacks_class = partial(BackgroundFocalSGDA, scenarios=scenarios)

            return self

    return BFSGDAConfig()


class Scenario:
    MAIN_POLICY_ID = "MAIN_POLICY"
    MAIN_POLICY_COPY_ID = "MAIN_POLICY_COPY"  # An older version of the main policy

    def __init__(self, num_copies, background_policies):
        self.num_copies = num_copies
        self.background_policies = background_policies

    def get_policies(self):
        policies = [Scenario.MAIN_POLICY_ID] + [Scenario.MAIN_POLICY_COPY_ID] * (
                    self.num_copies - 1) + self.background_policies

        # We suppose the order of players does not matter, but we shuffle it in cases s0 is different for each player.
        np.random.shuffle(policies)
        return policies

    @staticmethod
    def get_scenario_name(policies):
        np_policies = np.array(policies)
        main_policy_mask = ["background" not in p for p in np_policies]
        num_copies = len(np.argwhere(main_policy_mask))

        return f"<c={num_copies}, b={tuple(sorted(np_policies[np.logical_not(main_policy_mask)]))}>"


class ScenarioMapper:
    def __init__(self, scenarios=None, distribution=None):
        if distribution is None:
            distribution = np.ones(len(scenarios), dtype=np.float32) / len(scenarios)

        self.mappings = {}
        self.distribution = distribution
        self.scenarios = scenarios

    def __call__(self, agent_id, episode, worker, **kwargs):
        if episode.episode_id not in self.mappings:
            scenario_name = worker.config.scenarios.sample_scenario(self.distribution)

            self.mappings[episode.episode_id] = worker.config.scenarios[scenario_name].get_policies()

        mapping = self.mappings[episode.episode_id]
        policy_id = mapping.pop()
        if len(mapping) == 0:
            del self.mappings[episode.episode_id]

        return policy_id


class ScenarioDistribution:

    def __init__(self, algo: Algorithm, learn_best_responses=False):

        self.algo: Algorithm = algo
        self.config: "PPOBFSGDAConfig" = algo.config
        self.learn_best_responses = learn_best_responses

        self.best_response_utilities = {}

        self.scenarios: ScenarioSet = self.config.scenarios
        self.beta_logits = np.ones(len(self.scenarios), dtype=np.float32) / len(self.scenarios)
        if self.config.self_play:
            self.beta_logits[:] = [
                float(len(self.scenarios[scenario].background_policies) == 0)
                for scenario in self.scenarios.scenario_list
            ]

        self.past_betas = deque([], maxlen=self.config.beta_smoothing)
        self.weights_history = None
        self.copy_iter = 0
        self.prev_timesteps = 0
        self.weights_0 = ray.put(self.algo.get_weights([Scenario.MAIN_POLICY_ID])[Scenario.MAIN_POLICY_ID])

        # Init copy weights and best response utilities if needed:
        self.copy_weights(reset=True)
        self.missing_best_responses: list = []
        self.current_best_response_scenario = None
        self.current_best_response_utility = SmoothMetric(lr=0.98)
        self.scenario_utilities = defaultdict(lambda: SmoothMetric(lr=0.9))

        if not self.config.use_utility:

            self.best_response_timesteps = defaultdict(int)
            self.missing_best_responses: deque = deque(list(self.scenarios.scenario_list))
            self.load_best_response_utilities()
            if len(self.missing_best_responses) > 0:
                self.current_best_response_scenario = self.missing_best_responses.popleft()

            self.set_matchmaking()

        _base_compile_iteration_results = algo._compile_iteration_results

        def _compile_iteration_results_with_scenario_counts(
                _, *, episodes_this_iter, step_ctx, iteration_results=None
        ):
            r = _base_compile_iteration_results(episodes_this_iter=episodes_this_iter, step_ctx=step_ctx,
                                                iteration_results=iteration_results)

            scenario_counts = defaultdict(int)
            for episode in episodes_this_iter:
                for k in episode.custom_metrics:
                    if "utility" in k:
                        scenario_counts[k.rstrip("_utility")] += 1

            r["custom_metrics"]["scenario_counts"] = scenario_counts

            return r

        self.algo._compile_iteration_results = _compile_iteration_results_with_scenario_counts.__get__(self.algo, type(self.algo))

    def set_matchmaking(self):

        if self.current_best_response_scenario is None:
            distrib = self.beta_logits.copy()

        else:
            # We are learning best responses
            distrib = np.array([
                float(scenario == self.current_best_response_scenario) for scenario in self.scenarios.scenario_list
            ])

        self.algo.workers.foreach_worker(
            lambda w: w.set_policy_mapping_fn(
                ScenarioMapper(distribution=distrib)
            ),
        )

    def copy_weights(self, reset=False):
        if reset:
            self.weights_history = [self.weights_0 for _ in range(3)]
            self.copy_iter = 0
        else:
            last_weights = self.algo.get_weights([Scenario.MAIN_POLICY_ID])[Scenario.MAIN_POLICY_ID]
            self.weights_history.append(last_weights)
            if len(self.weights_history) > 30:
                self.weights_history.pop(0)

        weights = np.random.choice(self.weights_history[:-2])

        d = {Scenario.MAIN_POLICY_COPY_ID: weights}
        if reset:
            d[Scenario.MAIN_POLICY_ID] = weights

        self.algo.workers.foreach_worker(
            lambda w: w.set_weights(d)
        )

    def load_best_response_utilities(self):

        path = self.config.best_response_utilities_path.format(env_name=self.config.env)

        if os.path.exists(path):
            with open(path, "r") as f:
                best_response_utilities = yaml.safe_load(f)

            for scenario_name in self.scenarios.scenario_list:
                if scenario_name in best_response_utilities:
                    self.best_response_utilities[scenario_name] = best_response_utilities[scenario_name]
                    self.missing_best_responses.remove(scenario_name)

    def save_best_response_utilities(self):

        path = self.config.best_response_utilities_path.format(env_name=self.config.env)

        to_save = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                best_response_utilities = yaml.safe_load(f)
                to_save.update(best_response_utilities)

        # Update the best responses with the better ones found here.
        for new_value, scenario in self.best_response_utilities.items():
            if to_save.get(scenario, -np.inf) < new_value:
                to_save[scenario] = new_value

        with open(path, "w") as f:
            yaml.safe_dump(to_save, f)

    def beta_gradients(self, loss):
        if self.config.self_play or self.config.beta_lr == 0.:
            self.past_betas.append(self.beta_logits.copy())
            return

        self.beta_logits[:] = project_onto_simplex(self.beta_logits + loss * self.config.beta_lr)

        self.past_betas.append(self.beta_logits.copy())

        # Allow any scenario to be sampled with beta_eps prob
        self.beta_logits[:] = self.beta_logits * (1 - self.config.beta_eps) + self.config.beta_eps / len(
            self.beta_logits)

    def update(self, result: ResultDict):
        """
        We should get the regret/utility and update the distribution
        Update the policy mapping function after
        """

        time_steps = result["timesteps_total"]
        iter_data = result["custom_metrics"]

        # We are learning the best responses here.
        # Test if we are done learning some best responses
        if (not self.config.use_utility) and (self.current_best_response_scenario is not None):

            expected_utility = iter_data.get(f"{self.current_best_response_scenario}_utility_mean", 0.)
            if self.current_best_response_scenario not in self.best_response_utilities:
                self.current_best_response_utility.set(expected_utility)
                self.best_response_utilities[
                    self.current_best_response_scenario] = self.current_best_response_utility.get()
            else:
                self.current_best_response_utility.update(expected_utility)

                self.best_response_utilities[self.current_best_response_scenario] = float(np.maximum(
                    self.current_best_response_utility.get(),
                    self.best_response_utilities[self.current_best_response_scenario]
                ))

            self.best_response_timesteps[self.current_best_response_scenario] += time_steps - self.prev_timesteps
            if self.best_response_timesteps[
                self.current_best_response_scenario] >= self.config.best_response_timesteps_max:

                # expected_utility = iter_data[f"{self.current_best_response_scenario}_utility_mean"]
                # self.best_response_utilities[self.current_best_response_scenario] = expected_utility

                self.save_best_response_utilities()
                if len(self.missing_best_responses) > 0:
                    self.current_best_response_scenario = self.missing_best_responses.popleft()
                    self.set_matchmaking()
                else:
                    # Will move on to learning the minimax regret solution
                    self.current_best_response_scenario = None

                # Reset main policy to 0, along with its history of weights
                self.copy_weights(reset=True)

        self.copy_iter += 1

        if self.copy_iter % self.config.copy_weights_freq == 0:
            self.copy_weights()

        if self.config.use_utility or (self.current_best_response_scenario is None):
            if self.config.learn_best_responses_only:
                # We are done computing best responses, stop
                self.algo.stop()
            # Compute lossweight=iter_data["scenario_counts"][scenario]

            utilities = np.array([
                self.scenario_utilities[scenario].update(iter_data.get(f"{scenario}_utility_mean", np.nan),
                                                         weight=iter_data["scenario_counts"][scenario])
                for scenario in self.scenarios.scenario_list
            ])

            if self.config.use_utility:
                beta_losses = utilities
                beta_losses = np.max(beta_losses) - beta_losses + np.min(beta_losses)
            else:
                regrets = np.array([
                    self.best_response_utilities[scenario] - self.scenario_utilities[scenario].get()
                    for scenario in self.scenarios.scenario_list
                ])
                beta_losses = regrets
                for scenario in self.scenarios.scenario_list:
                    iter_data[f"{scenario}_regret_mean"] = (self.best_response_utilities[scenario]
                                                            - self.scenario_utilities[scenario].get())
                iter_data[f"worst_case_regret"] = np.max(regrets)
                iter_data[f"uniform_regret"] = np.mean(regrets)
                iter_data[f"curr_distrib_regret"] = np.sum(regrets * self.beta_logits)

            iter_data[f"worst_case_utility"] = np.min(utilities)
            iter_data[f"uniform_utility"] = np.mean(utilities)
            iter_data[f"curr_distrib_utility"] = np.sum(utilities * self.beta_logits)

            # # Todo : is this fine ? We shouldn't make this  happen with large batch sizes
            # beta_losses[np.isnan(beta_losses)] = np.nanmean(beta_losses)
            self.beta_gradients(beta_losses)

            # Update the matchmaking scheme
            self.set_matchmaking()

            # TODO : plot evolution of distribution

        self.prev_timesteps = time_steps

        if not self.config.use_utility:
            result["custom_metrics"]["missing_best_response_utilities"] = len(self.missing_best_responses)
            if self.current_best_response_scenario is None:
                result["custom_metrics"]["best_response_timesteps"] = 0

            else:
                result["custom_metrics"]["best_response_timesteps"] = self.best_response_timesteps[
                    self.current_best_response_scenario]

        result["hist_stats"]["scenario_distribution"] = distribution_to_hist(self.beta_logits)
        for scenario, prob in zip(self.scenarios.scenario_list, self.beta_logits):
            result["custom_metrics"][f"{scenario}_probability"] = prob

        # Get rid of duplicates
        result.update(**result["custom_metrics"])
        del result["custom_metrics"]
        del result["sampler_results"]["custom_metrics"]


        return result


if __name__ == '__main__':
    p = "data/test_sets/RandomPOMDP_seed_0_n_states_5_n_actions_3_num_players_2_episode_length_100_history_length_2_full_one_hot_True/deterministic_set_tmp.YAML"
    test_set = ScenarioSet.from_YAML(
        "data/test_sets/RandomPOMDP_seed_0_n_states_5_n_actions_3_num_players_2_episode_length_100_history_length_2_full_one_hot_True/deterministic_set_0.YAML")
    test_set.to_YAML(p, eval_config={"test_OK": [0, 1, 2, 3]})

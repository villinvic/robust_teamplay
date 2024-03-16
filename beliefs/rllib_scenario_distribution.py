import itertools
import os.path
import pickle
import queue
from collections import deque, defaultdict
from copy import copy, deepcopy
from typing import Dict, Tuple, Union, Optional, List

import numpy as np
import ray
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


class BackgroundFocalSGDA(DefaultCallbacks):

    def __init__(self):
        super().__init__()

        self.beta: ScenarioDistribution = None

    def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:

        self.beta = ScenarioDistribution(algorithm)

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

        focal_rewards = [
            batch[SampleBatch.REWARDS] for agent_id, (_, batch) in original_batches.items()
            if "background" not in episode._agent_to_policy[agent_id]
        ]

        mean_focal_per_capita = sum(focal_rewards) / len(focal_rewards)
        postprocessed_batch[SampleBatch.REWARDS][:] = mean_focal_per_capita

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

        policies = episode._agent_to_policy.values()
        scenario_name = Scenario.get_scenario_name(list(policies))

        episode.custom_metrics[f"{scenario_name}_utility"] = episodic_mean_focal_per_capita


    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        self.beta.update(result)




class ScenarioSet:
    def __init__(self, num_players, background_population):
        self.scenarios = {}

        for num_copies in range(1, num_players + 1):

            for background_policies in itertools.combinations_with_replacement(background_population,
                                                                               num_players - num_copies):
                policies = (Scenario.MAIN_POLICY_ID,) + (Scenario.MAIN_POLICY_COPY_ID,) * (
                        num_copies - 1) + background_policies

                scenario_name = Scenario.get_scenario_name(policies)

                self.scenarios[scenario_name] = Scenario(num_copies, list(background_policies))

        self.scenario_list = np.array(list(self.scenarios.keys()))

    def sample_scenario(self, distribution):
        return np.random.choice(self.scenario_list, p=distribution)

    def __getitem__(self, item):
        return self.scenarios[item]

    def __len__(self):
        return len(self.scenario_list)

    def split(self, n=None):
        if n in None:
            n = len(self.scenario_list)

        subsets = []
        for sublist in np.split(self.scenario_list, n):
            subset = copy(self)
            subset.scenario_list = list(sublist)
            subsets.append(subset)
        return subsets



class PPOBFSGDAConfig(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class)

        self.beta_lr = 5e-2
        self.beta_smoothing = 1000
        self.use_utility = False
        self.scenarios = None
        self.copy_weights_freq = 5
        self.beta_eps = 1e-2

        self.learn_best_responses_only = True
        self.best_response_timesteps_max = 1_000_000
        self.best_response_utilities_path = os.getcwd() +  "/data/best_response_utilities/{env_name}.pkl"

        # TODO if we have deep learning bg policies:
        self.background_population_path = None

        self.callbacks_class = BackgroundFocalSGDA

    def training(
            self,
            *,
            beta_lr: Optional[float] = NotProvided,
            beta_smoothing: Optional[float] = NotProvided,
            use_utility: Optional[bool] = NotProvided,
            copy_weights_freq: Optional[int] = NotProvided,
            best_response_timesteps_max: Optional[int] = NotProvided,
            best_response_utilities_path: Optional[str] = NotProvided,
            learn_best_responses_only: Optional[bool] = NotProvided,
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
        if copy_weights_freq is not NotProvided:
            self.copy_weights_freq = copy_weights_freq
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

        return self


def make_bf_sgda_config(cls):
    class BFSGDAConfig(cls):
        def __init__(self):
            super().__init__()

            self.beta_lr = 5e-2
            self.beta_smoothing = 1000
            self.use_utility = False
            self.scenarios = None
            self.copy_weights_freq = 5
            self.beta_eps = 1e-2

            self.learn_best_responses_only = True
            self.best_response_timesteps_max = 1_000_000
            self.best_response_utilities_path = os.getcwd() + "/data/best_response_utilities/{env_name}.pkl"

            # TODO if we have deep learning bg policies:
            self.background_population_path = None

            self.callbacks_class = BackgroundFocalSGDA

        def training(
                self,
                *,
                beta_lr: Optional[float] = NotProvided,
                beta_smoothing: Optional[float] = NotProvided,
                use_utility: Optional[bool] = NotProvided,
                copy_weights_freq: Optional[int] = NotProvided,
                best_response_timesteps_max: Optional[int] = NotProvided,
                best_response_utilities_path: Optional[str] = NotProvided,
                learn_best_responses_only: Optional[bool] = NotProvided,
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
            if copy_weights_freq is not NotProvided:
                self.copy_weights_freq = copy_weights_freq
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

            return self

    return BFSGDAConfig()



class Scenario:
    MAIN_POLICY_ID = "MAIN_POLICY"
    MAIN_POLICY_COPY_ID = "MAIN_POLICY_COPY"  # An older version of the main policy

    def __init__(self, num_copies, background_policies):
        self.num_copies = num_copies
        self.background_policies = background_policies

    def get_policies(self):
        policies = [Scenario.MAIN_POLICY_ID] + [Scenario.MAIN_POLICY_COPY_ID] * (self.num_copies-1) + self.background_policies

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
        # The background population should already be loaded in the multiagent config of the algorithm
        # TODO : this needs information on
        # Which scenario do we need to optimize on ? -> All that can be constructed from the bg population !
        # What if we want to learn against one scenario specifically ?
        # Save a file that stores best response utility depending on the scenario
        # -> a dict bru[f"{env_name}_{env_seed}_{scenario_n_copies}_{background_policy_1_path}_..._{background_policy_m_path}"] = bru
        # TODO : write a way to load a background population, and compute all bru needed -> dump into file.
        # -> Focus iteratively on each scenario with a new initialized policy, dump after t timesteps, repeat (skip if entry already in file)
        self.algo: Algorithm = algo
        self.config: PPOBFSGDAConfig = algo.config
        self.learn_best_responses = learn_best_responses

        self.best_response_utilities = {}

        self.scenarios = self.config.scenarios
        self.beta_logits = np.ones(len(self.scenarios), dtype=np.float32) / len(self.scenarios)
        self.past_priors = list()
        self.weights_history = None
        self.copy_iter = 0
        self.prev_timesteps = 0
        self.weights_0 = ray.put(self.algo.get_weights([Scenario.MAIN_POLICY_ID])[Scenario.MAIN_POLICY_ID])

        # Init copy weights and best response utilities if needed:
        self.copy_weights(reset=True)
        self.missing_best_responses: list = []
        self.current_best_response_scenario = None
        self.current_best_response_utility = SmoothMetric(lr=0.98)
        self.scenario_utilities = defaultdict(lambda : SmoothMetric(lr=0.99))

        if not self.config.use_utility:

            self.best_response_timesteps = defaultdict(int)
            self.missing_best_responses: deque = deque(list(self.scenarios.scenario_list))
            self.load_best_response_utilities()
            if len(self.missing_best_responses) > 0:
                self.current_best_response_scenario = self.missing_best_responses.popleft()

            self.set_matchmaking()

        _base_compile_iteration_results = algo._compile_iteration_results
        def _compile_iteration_results(
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


        #setattr(self.algo, "_base_compile_iteration_results", self.algo._compile_iteration_results)
        self.algo._compile_iteration_results = _compile_iteration_results.__get__(self.algo, type(self.algo))



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
            self.weights_history = deque([self.weights_0 for _ in range(19)], maxlen=20)
            self.copy_iter = 0
        else:
            last_weights = ray.put(self.algo.get_weights([Scenario.MAIN_POLICY_ID])[Scenario.MAIN_POLICY_ID])
            self.weights_history.append(last_weights)

        weights = self.weights_history.popleft()

        d = {Scenario.MAIN_POLICY_COPY_ID: weights}
        if reset:
            d[Scenario.MAIN_POLICY_ID] = weights

        self.algo.workers.foreach_worker(
            lambda w: w.set_weights(d)
        )

    def load_best_response_utilities(self):

        path = self.config.best_response_utilities_path.format(env_name=self.config.env)

        if os.path.exists(path):
            with open(path, "rb") as f:
                best_response_utilities = pickle.load(f)
            for scenario_name in self.scenarios.scenario_list:
                if scenario_name in best_response_utilities:
                    self.best_response_utilities[scenario_name] = best_response_utilities[scenario_name]
                    self.missing_best_responses.remove(scenario_name)

    def save_best_response_utilities(self):

        path = self.config.best_response_utilities_path.format(env_name=self.config.env)

        to_save = {}
        if os.path.exists(path):
            with open(path, "rb") as f:
                best_response_utilities = pickle.load(f)
                to_save.update(best_response_utilities)

        to_save.update(**self.best_response_utilities)

        with open(path, "wb+") as f:
            pickle.dump(to_save, f)

    def beta_gradients(self, loss):
        self.beta_logits[:] = project_onto_simplex(self.beta_logits + loss * self.config.beta_lr)

        # Allow any scenario to be sampled with beta_eps prob
        self.beta_logits[:] = self.beta_logits * (1-self.config.beta_eps) + self.config.beta_eps / len(self.beta_logits)

        self.past_priors.append(self.beta_logits.copy())

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
                self.best_response_utilities[self.current_best_response_scenario] = self.current_best_response_utility.get()
            else:
                self.current_best_response_utility.update(expected_utility)

                self.best_response_utilities[self.current_best_response_scenario] = np.maximum(
                    self.current_best_response_utility.get(),
                    self.best_response_utilities[self.current_best_response_scenario]
                )

            self.best_response_timesteps[self.current_best_response_scenario] += time_steps - self.prev_timesteps
            if self.best_response_timesteps[
                    self.current_best_response_scenario] >= self.config.best_response_timesteps_max:

                #expected_utility = iter_data[f"{self.current_best_response_scenario}_utility_mean"]
                #self.best_response_utilities[self.current_best_response_scenario] = expected_utility

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

            if self.config.use_utility:
                beta_losses = np.array([
                    self.scenario_utilities[scenario].update(iter_data.get(f"{scenario}_utility_mean", np.nan),
                                                             weight=iter_data["scenario_counts"][scenario])
                    for scenario in self.scenarios.scenario_list
                ])
                beta_losses = np.max(beta_losses) - beta_losses + np.min(beta_losses)
            else:
                beta_losses = np.array([
                    self.best_response_utilities[scenario] - self.scenario_utilities[scenario].update(iter_data.get(f"{scenario}_utility_mean", np.nan),
                                                                                                      weight=iter_data["scenario_counts"][scenario])
                    for scenario in self.scenarios.scenario_list
                ])
                for scenario in self.scenarios.scenario_list:
                    iter_data[f"{scenario}_regret_mean"] =  self.best_response_utilities[scenario] - self.scenario_utilities[scenario].get()
                iter_data[f"worst_case_regret"] = np.max(beta_losses)
                iter_data[f"uniform_regret"] = np.mean(beta_losses)
                iter_data[f"curr_distrib_regret"] = np.sum(beta_losses * self.beta_logits)

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
                result["custom_metrics"]["best_response_timesteps"] = self.best_response_timesteps[self.current_best_response_scenario]



        result["hist_stats"]["scenario_distribution"] = self.beta_logits.copy()
        for scenario, prob in zip(self.scenarios.scenario_list, self.beta_logits):
            result["custom_metrics"][f"{scenario}_probability"] = prob

        # Get rid of duplicates
        del result["sampler_results"]["custom_metrics"]

        return result


if __name__ == '__main__':

    scenario_set = ScenarioSet(2, ["background_bob", "background_luc"])
    print()
    l = scenario_set.split(len(scenario_set.scenario_list))
    for subset in l:
        print(subset.scenario_list)
import itertools
import queue
from collections import deque
from typing import Dict, Tuple, Union, Optional, List

import numpy as np
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


class PPOBFSGDAConfig(PPOConfig):
    def __init__(self, algo_class = None):
        super().__init__(algo_class=algo_class)

        self.beta_lr = 5e-2
        self.beta_smoothing = 1000
        self.use_utility = False

        self.callbacks_class = BackgroundFocalSGDA

    def training(
        self,
        *,
        beta_lr: Optional[float] = NotProvided,
        beta_smoothing: Optional[float] = NotProvided,
        use_utility: Optional[bool] = NotProvided,
        **kwargs,
    ) -> "PPOConfig":

        super().training(**kwargs)
        if beta_lr is not None:
            self.beta_lr = beta_lr
        if beta_smoothing is not None:
            self.beta_smoothing = beta_smoothing
        if use_utility is not None:
            self.use_utility = use_utility

        return self



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
            if policy_id == Scenario.MAIN_POLICY_ID
        ]
        episodic_mean_focal_per_capita = sum(focal_rewards) / len(focal_rewards)

        policies = episode._agent_to_policy.values()
        scenario_name = Scenario.get_scenario_name(list(policies))

        episode.custom_metrics[f"{scenario_name}_utility" ] = episodic_mean_focal_per_capita

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
                policies = (Scenario.MAIN_POLICY_ID,) * num_copies + background_policies
                scenario_name = Scenario.get_scenario_name(policies)
                self.scenarios[scenario_name] = Scenario(num_copies, list(background_policies))

        self.scenario_list = np.array(list(self.scenarios.keys()))

    def sample_scenario(self, distribution):
        return np.random.choice(self.scenario_list, p=distribution)

    def __len__(self):
        return len(self.scenario_list)


class Scenario:

    MAIN_POLICY_ID = "MAIN_POLICY"

    def __init__(self, num_copies, background_policies):
        self.num_copies = num_copies
        self.background_policies = background_policies

    def get_policies(self, main_policy_id):
        policies = [main_policy_id] * self.num_copies + self.background_policies
        np.random.shuffle(policies)
        return policies

    @staticmethod
    def get_scenario_name(policies):

        np_policies = np.array(policies)
        main_policy_mask = np_policies == Scenario.MAIN_POLICY_ID
        num_copies = len(np.argwhere(main_policy_mask))

        return f"<c={num_copies}, b={tuple(sorted(np_policies[not main_policy_mask]))}>"


class ScenarioMapper:
    def __init__(self, distribution, scenarios):

        self.mappings = {}
        self.distribution = distribution
        self.scenarios = scenarios

    def __call__(self, agent_id, episode, worker, **kwargs):
            if episode.episode_id not in self.mappings:
                self.mappings[episode.episode_id] = self.scenarios.sample_scenario(self.distribution).get_policies()

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

        self.best_responses_utility = {}

        self.scenarios = ScenarioSet(self.algo.config.multiagent["policies"])
        self.beta_logits = np.ones(len(self.scenarios), dtype=np.float32)

        self.past_priors = list()

    def beta_gradients(self, loss):

        if self.config.use_utility:
            loss = np.max(loss) - loss + np.min(loss)

        self.beta_logits[:] = project_onto_simplex(self.beta_logits + loss * self.config.beta_lr)

        self.past_priors.append(self.beta_logits.copy())


    def update(self, result: ResultDict):
        """
        We should get the regret/utility and update the distribution
        Update the policy mapping function after
        """

        print(result)


        # Compute loss
        # loss = ...
        # self.beta_gradients(loss)
        #
        # self.algo.workers.foreach_worker(
        #     lambda w: w.set_policy_mapping_fn(
        #     ScenarioMapper(self.beta_logits.copy(), self.scenarios)
        # ),
        # )


        # TODO : plot evolution of distribution






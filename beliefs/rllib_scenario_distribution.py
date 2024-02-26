from typing import Dict, Tuple, Union, Optional

from ray.rllib import Policy, SampleBatch, BaseEnv
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import ResultDict, AgentID, PolicyID


def decorate_with_learnable_matchmaking(algo: Algorithm):

    class BackgroundFocalSGDA(algo):

        def setup(self, config):
            # Create the LeagueBuilder object, allowing it to build the multiagent
            # config as well.
            self.scenario_distribution = ScenarioDistribution(self, config)

            super().setup(config)

        def step(self) -> ResultDict:
            # Perform a full step (including evaluation).
            result = super().step()

            # Based on the (train + evaluate) results, update the distribution of scenarios
            self.scenario_distribution.update(result=result)

            return result




    BackgroundFocalSGDA.__name__ = f"BackgroundFocalSGDA[{Algorithm.__name__}]"
    BackgroundFocalSGDA.__qualname__ = f"BackgroundFocalSGDA[{Algorithm.__name__}]"

    return BackgroundFocalSGDA


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

        mean_focal_per_capita = sum([focal_rewards]) / len(focal_rewards)
        postprocessed_batch[SampleBatch.REWARDS][:] = mean_focal_per_capita


    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:

        self.beta.update(result)


class ScenarioDistribution:

    def __init__(self, algo: Algorithm, lr=1e-2, learn_best_responses=False):
        # The background population should already be loaded in the multiagent config of the algorithm
        # TODO : this needs information on
        # Which scenario do we need to optimize on ? -> All that can be constructed from the bg population !
        # What if we want to learn against one scenario specifically ?
        # Save a file that stores best response utility depending on the scenario
        # -> a dict bru[f"{env_name}_{env_seed}_{scenario_n_copies}_{background_policy_1_path}_..._{background_policy_m_path}"] = bru
        # TODO : write a way to load a background population, and compute all bru needed -> dump into file.
        # -> Focus iteratively on each scenario with a new initialized policy, dump after t timesteps, repeat (skip if entry already in file)
        self.algo: Algorithm = algo
        self.config = algo.config["scenario_distribution_config"]
        self.lr = lr
        self.learn_best_responses = learn_best_responses


    def update(self, result: ResultDict):
        """
        We should get the regret/utility and update the distribution
        Update the policy mapping function after
        """
        pass


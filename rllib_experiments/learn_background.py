import os
from typing import Dict, Tuple, Union, Optional

import numpy as np
import fire
from ray.rllib import SampleBatch, BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.ppo import PPOConfig

from ray import tune
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.tune import register_env

from beliefs.rllib_scenario_distribution import Scenario, ScenarioMapper, ScenarioSet, \
    make_bf_sgda_config
from environments.rllib.random_mdp import RandomPOMDP
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy

from rllib_experiments.configs import get_env_config

num_workers = os.cpu_count() - 2


class SocialRewards(DefaultCallbacks):

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

        other_rewards = [
            batch[SampleBatch.REWARDS] for aid, (pid, policy_cls, batch) in original_batches.items()
            if aid != agent_id
        ]
        mean_other_rewards = sum(other_rewards) / max([len(other_rewards), 1.])

        social_weight = policies[policy_id].config["social_weight"]

        episode.custom_metrics[policy_id + "_trajectory_other_rewards"] = np.sum(mean_other_rewards)
        episode.custom_metrics[policy_id + "_trajectory_own_rewards"] = np.sum(postprocessed_batch[SampleBatch.REWARDS].copy())


        postprocessed_batch[SampleBatch.REWARDS][:] = (
                postprocessed_batch[SampleBatch.REWARDS] * (1. - social_weight)
                +
                mean_other_rewards * social_weight
        ) * 0.

        episode.custom_metrics[policy_id + "_trajectory_optimized_rewards"] = np.sum(postprocessed_batch[SampleBatch.REWARDS][:])




def main(
        *,
        num_background=8,
        version="0.7",
        env="RandomPOMDP",

        **kwargs
):
    env_config = get_env_config(
        environment_name=env
    )(**kwargs)

    env_config_dict = env_config.as_dict()
    env_id = env_config.get_env_id()

    register_env(env_id, env_config.get_maker(num_scenarios=1))

    rollout_fragment_length = env_config.episode_length // 10

    dummy_env = env_config.get_maker(num_scenarios=1)()

    policies = {
        f"IMPALA_policy_{social_weight:.2f}": (
            None,
            dummy_env.observation_space[0],
            dummy_env.action_space[0],
            dict(
                social_weight=social_weight,
            )
        ) for social_weight in np.linspace(-0.5, 2., num_background, endpoint=True)
    }
    pids = list(policies)

    def policy_mapping(*args, **kwargs):
        return np.random.choice(pids)

    batch_size = rollout_fragment_length * num_workers
    max_samples = 50_000_000
    num_iters = max_samples // batch_size
    config = ImpalaConfig().training(

        # IMPALA
        #opt_type="rmsprop",
        entropy_coeff=1e-4,
        train_batch_size=batch_size,
        momentum=0.,
        epsilon=1e-5,
        decay=0.99,
        lr=2e-3,
        grad_clip=50.,
        gamma=0.995,

        model={
            "custom_model": "models.softmax.multi_values.MultiValueSoftmax",
            "custom_model_config": {
                "n_scenarios": 1,
            }
        }
    ).rollouts(
        num_rollout_workers=num_workers,
        sample_async=False,
        create_env_on_local_worker=False,
        num_envs_per_worker=1,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode="truncate_episodes",
        enable_connectors=True,
    ).environment(
        env=env_id,
        env_config=env_config_dict
    ).resources(num_gpus=0
    ).framework(framework="tf"
    ).multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping,
    ).experimental(
        _disable_preprocessor_api=True
    ).callbacks(
        callbacks_class=SocialRewards
    )

    exp = tune.run(
        "IMPALA",
        name=f"learn_background_v{version}",
        config=config,
        checkpoint_at_end=True,
        checkpoint_freq=100,
        keep_checkpoints_num=3,
        stop={
            "timesteps_total": max_samples,
        },
    )

if __name__ == '__main__':
    fire.Fire(main)

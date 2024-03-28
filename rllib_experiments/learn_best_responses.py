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
from constants import PolicyIDs
from environments.rllib.random_mdp import RandomPOMDP
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy
from rllib_experiments.benchmarking import PolicyCkpt

from rllib_experiments.configs import get_env_config

# ma_cartpole_cls = make_multi_agent("Pendulum-v1")
#
#
# def env_maker_test(env_config):
#     return ma_cartpole_cls({"num_agents": 2})


def main(
        *,
        background=["random", "deterministic_0"],
        env="RandomPOMDP",
        **kwargs
):
    """
    TODO: raises tf graph error when an entry is already in the best response data.
    """
    env_config = get_env_config(
        environment_name=env
    )(**kwargs)

    env_config_dict = env_config.as_dict()
    env_id = env_config.get_env_id()

    background_population = [
        PolicyCkpt(p_name, env_id)
        for p_name in background
    ]

    scenarios = ScenarioSet()

    #TODO : automate background naming, this was annoying to debug
    scenarios.build_from_population(
        num_players=env_config.num_players,
        background_population=["background_" + p.name for p in background_population]
    )

    register_env(env_id, env_config.get_maker(num_scenarios=len(scenarios)))

    rollout_fragment_length = env_config.episode_length // 10
    dummy_env = env_config.get_maker(num_scenarios=len(scenarios))()

    policies = {
        "background_" + p.name : p.get_policy_specs(dummy_env)
        for p in background_population
    }

    num_workers = (os.cpu_count() - 1 - len(scenarios)) // len(scenarios)

    for policy_id in (PolicyIDs.MAIN_POLICY_ID, PolicyIDs.MAIN_POLICY_COPY_ID):
        policies[policy_id] = (
            None,
            dummy_env.observation_space[0],
            dummy_env.action_space[0],
            {}
        )

    # config = PPOConfig().training(

    # ImpalaConfig().training(
    #     opt_type="rmsprop",
    #     entropy_coeff=1e-4,
    #     train_batch_size=rollout_fragment_length * 16,
    #     momentum=0.,
    #     epsilon=1e-5,
    #     decay=0.99,
    #
    #     gamma=0.99,
    # )
    batch_size = rollout_fragment_length * num_workers
    max_samples = 5_000_000
    num_iters = max_samples // batch_size

    config = make_bf_sgda_config(ImpalaConfig).training(
        learn_best_responses_only=True,
        scenarios=tune.grid_search(scenarios.split()),

        copy_weights_freq=1,
        copy_history_len=10,
        best_response_timesteps_max=max_samples,

        # IMPALA
        # opt_type="rmsprop",
        entropy_coeff=1e-4,
        train_batch_size=batch_size,
        momentum=0.,
        epsilon=1e-5,
        decay=0.99,
        lr=2e-3,
        grad_clip=50.,
        gamma=0.995,

        # PPO
        # lambda_=0.95,
        # gamma=0.99,
        # entropy_coeff=1e-4,
        # lr=1e-4,
        # lambda_=0.95,
        # gamma=0.999,
        # entropy_coeff=1e-3,
        # lr=1e-2,
        # use_critic=False,
        # use_gae=False,
        # kl_coeff=0.,
        # #kl_target=1e-2, #1e-2
        # clip_param=1000.,
        # grad_clip=None,
        # train_batch_size=rollout_fragment_length * num_workers * 16,
        # sgd_minibatch_size=rollout_fragment_length * num_workers * 16,
        # num_sgd_iter=1,
        model={
            "custom_model": "models.softmax.multi_values.MultiValueSoftmax",
            "custom_model_config": {
                "n_scenarios": len(scenarios),
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
        policies_to_train={PolicyIDs.MAIN_POLICY_ID},
        policy_mapping_fn=ScenarioMapper(
            scenarios=scenarios
        ),
    ).experimental(
        _disable_preprocessor_api=True
    )

    exp = tune.run(
        "IMPALA",
        name="best_responses_learning",
        config=config,
        checkpoint_at_end=True,
        checkpoint_freq=30,
        keep_checkpoints_num=3,
    )


if __name__ == '__main__':
    fire.Fire(main)

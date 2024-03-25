import os

import fire
from ray.rllib import SampleBatch
from ray.rllib.algorithms.impala import ImpalaConfig, ImpalaTF1Policy
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig

from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from beliefs.rllib_scenario_distribution import Scenario, ScenarioMapper, ScenarioSet, \
    make_bf_sgda_config
from environments.rllib.random_mdp import RandomPOMDP
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy

from rllib_experiments.configs import get_env_config

num_workers = os.cpu_count() - 2


def main(
        *,
        background=(0,),
        version="0.6",
        env="RandomPOMDP",
        use_utility=False,
        beta_lr=2e-1,
        **kwargs
):
    env_config = get_env_config(
        environment_name=env
    )(**kwargs)

    env_config_dict = env_config.as_dict()
    env_id = env_config.get_env_id()

    register_env(env_id, env_config.get_maker())

    rollout_fragment_length = env_config.episode_length // 10
    dummy_env = env_config.get_maker()()

    policies = {
        f"background_deterministic_{bg_policy_seed}": (
            RLlibDeterministicPolicy,
            dummy_env.observation_space[0],
            dummy_env.action_space[0],
            dict(
                config=env_config_dict,
                seed=bg_policy_seed,
                #_disable_preprocessor_api=True,
            )


        ) for i, bg_policy_seed in enumerate(background)
    }
    background_population = list(policies.keys())
    scenarios = ScenarioSet()

    scenarios.build_from_population(
        num_players=env_config.num_players,
        background_population=background_population
    )

    class InfoPolicy(ImpalaTF1Policy):

        def __init__(self, *agrs, **kwargs):
            super().__init__(*agrs, **kwargs)

            self.model.view_requirements[SampleBatch.INFOS].used_for_training = True
            self.view_requirements[SampleBatch.INFOS].used_for_training = True

            print(self.model.view_requirements, self.view_requirements)


    for policy_id in (Scenario.MAIN_POLICY_ID, Scenario.MAIN_POLICY_COPY_ID):
        policies[policy_id] = (InfoPolicy, dummy_env.observation_space[0], dummy_env.action_space[0], {})

    config = make_bf_sgda_config(ImpalaConfig).training(
        beta_lr=beta_lr, #2e-1,
        beta_smoothing=2000,
        use_utility=use_utility,
        scenarios=scenarios,
        copy_weights_freq=1,
        copy_history_len=10,
        beta_eps=1e-2,
        learn_best_responses_only=False,

        # IMPALA
        # opt_type="rmsprop",
        entropy_coeff=4e-3,
        train_batch_size=env_config.episode_length * num_workers ,
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
        # lambda_=1.0,
        # gamma=1.,
        # entropy_coeff=0.,
        # lr=1e-2,
        # use_critic=False,
        # use_gae=False,
        # #kl_coeff=0.,
        # #kl_target=1e-2, #1e-2
        # #clip_param=10.,
        # # #clip_param=0.2,
        # grad_clip=100.,
        # train_batch_size=64*num_workers*16,
        # sgd_minibatch_size=64*num_workers*2,
        # num_sgd_iter=16,
        model={
            # "fcnet_hiddens": [], # We learn a parameter for each state, simple softmax parametrization
            # "vf_share_layers": False,
            # #"fcnet_activation": "linear",

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
        batch_mode="complete_episodes",
        enable_connectors=True,
    ).environment(
        env=env_id,
        env_config=env_config_dict
    ).resources(num_gpus=0
    ).framework(framework="tf"
    ).multi_agent(
        policies=policies,
        policies_to_train={Scenario.MAIN_POLICY_ID},
        policy_mapping_fn=ScenarioMapper(
            scenarios=scenarios
        ),
    ).experimental(
        _disable_preprocessor_api=False
    ).reporting(
        min_time_s_per_iteration=0,
        min_train_timesteps_per_iteration=1,
        min_sample_timesteps_per_iteration=0,
    )

    exp = tune.run(
        "IMPALA",
        name=f"BF_SGDA_v{version}",
        config=config,
        checkpoint_at_end=True,
        checkpoint_freq=300,
        keep_checkpoints_num=3,
        stop={
            "timesteps_total": 50_000_000,
        },
    )




if __name__ == '__main__':
    fire.Fire(main)

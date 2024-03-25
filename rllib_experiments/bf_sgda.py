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
        version="0.7",
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

    background_population = [
        f"background_deterministic_{bg_policy_seed}"
        for bg_policy_seed in background
    ]
    scenarios = ScenarioSet()

    scenarios.build_from_population(
        num_players=env_config.num_players,
        background_population=background_population
    )

    register_env(env_id, env_config.get_maker(len(scenarios)))

    rollout_fragment_length = env_config.episode_length
    dummy_env = env_config.get_maker(len(scenarios))()

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

    # class InfoPolicy(ImpalaTF1Policy):
    #
    #     def _init_view_requirements(self):
    #         # If ViewRequirements are explicitly specified.
    #         if getattr(self, "view_requirements", None):
    #             return
    #
    #         # Use default settings.
    #         # Add NEXT_OBS, STATE_IN_0.., and others.
    #         self.view_requirements = self._get_default_view_requirements()
    #         # Combine view_requirements for Model and Policy.
    #         # TODO(jungong) : models will not carry view_requirements once they
    #         # are migrated to be organic Keras models.
    #         self.view_requirements.update(**self.model.view_requirements)
    #     def _get_default_view_requirements(self):
    #         view_reqs = super()._get_default_view_requirements()
    #         view_reqs.update(**self.model.view_requirements)
    #         return view_reqs


    for policy_id in (Scenario.MAIN_POLICY_ID, Scenario.MAIN_POLICY_COPY_ID):
        policies[policy_id] = (None, dummy_env.observation_space[0], dummy_env.action_space[0], {})

    config = make_bf_sgda_config(ImpalaConfig).training(
        beta_lr=beta_lr, #2e-1,
        beta_smoothing=2000,
        use_utility=use_utility,
        scenarios=scenarios,
        copy_weights_freq=1,
        copy_history_len=10,
        beta_eps=4e-2,
        learn_best_responses_only=False,

        # IMPALA
        # opt_type="rmsprop",
        entropy_coeff=tune.grid_search([0., 1e-4]),
        train_batch_size=tune.grid_search([
            rollout_fragment_length * num_workers,
            env_config.episode_length * num_workers, env_config.episode_length * num_workers * 4
        ]) ,
        momentum=0.,
        epsilon=1e-5,
        decay=0.99,
        lr=tune.grid_search([1e-2, 2e-3, 1e-3]),
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
        _disable_preprocessor_api=True
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

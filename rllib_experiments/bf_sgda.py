import os

import fire
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig

from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from beliefs.rllib_scenario_distribution import Scenario, PPOBFSGDAConfig, ScenarioMapper, ScenarioSet, \
    make_bf_sgda_config
from environments.rllib.random_mdp import RandomPOMDP
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy


def env_maker(env_config):

    return RandomPOMDP(**env_config)

num_workers = os.cpu_count() - 2


def main(
        bg_policies=[0],
        env_seed=0,
        n_states=5,
        n_actions=3,
        num_players=2,
        episode_length=64,
        history_length=2,
        full_one_hot=True
):
    env_config = dict(
        n_states=n_states,
        n_actions=n_actions,
        episode_length=episode_length,
        seed=env_seed,
        num_players=num_players,
        history_length=history_length,
        full_one_hot=full_one_hot
    )


    config_name = str(env_config).replace("'", "").replace(" ", "").replace(":", "_").replace(",", "_")[1:-1]
    env_name = f"RandomMDP_{config_name}"
    register_env(env_name, env_maker)


    rollout_fragment_length = episode_length // 10

    dummy_env = RandomPOMDP(**env_config)

    policies = {
        f"background_deterministic_{bg_policy_seed}": (
            RLlibDeterministicPolicy,
            dummy_env.observation_space[0],
            dummy_env.action_space[0],
            dict(
                config=env_config,
                seed=bg_policy_seed,
                _disable_preprocessor_api=True,
            )


        ) for i, bg_policy_seed in enumerate(bg_policies)
    }
    background_population = list(policies.keys())
    scenarios = ScenarioSet(
        num_players=env_config["num_players"],
        background_population=background_population
    )

    for policy_id in (Scenario.MAIN_POLICY_ID, Scenario.MAIN_POLICY_COPY_ID):
        policies[policy_id] = (None, dummy_env.observation_space[0], dummy_env.action_space[0], {})

    config = make_bf_sgda_config(ImpalaConfig).training(
        beta_lr=1e-1,
        beta_smoothing=2000,
        use_utility=False,
        scenarios=scenarios,
        copy_weights_freq=1,
        beta_eps=2e-2,

        learn_best_responses_only=False,

        # IMPALA
        # opt_type="rmsprop",
        entropy_coeff=1e-4,
        train_batch_size=rollout_fragment_length,
        momentum=0.,
        epsilon=1e-5,
        decay=0.99,
        lr=1e-3,
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
            "fcnet_hiddens": [], # We learn a parameter for each state, simple softmax parametrization
            "vf_share_layers": False,
            #"fcnet_activation": "linear",
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
        env=env_name,
        env_config=env_config
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
    )

    exp = tune.run(
        "IMPALA",
        name="BF_SGDA_v0.5",
        config=config,
        checkpoint_at_end=False,
        checkpoint_freq=30,
        keep_checkpoints_num=3,
        stop={
            "timesteps_total": 1_000_000_000,
        },
        #local_dir="rllib_runs",
        # restore=""
        # resume=True
    )


if __name__ == '__main__':
    fire.Fire(main)

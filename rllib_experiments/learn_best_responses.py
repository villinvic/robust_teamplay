import fire
from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env

from beliefs.rllib_scenario_distribution import Scenario, PPOBFSGDAConfig
from environments.rllib.random_mdp import RandomPOMDP
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy


def env_maker(env_config):

    return RandomPOMDP(*env_config)

register_env("RandomMDP", RandomPOMDP)

num_workers = 4
rollout_fragment_length = 16


def main(
        scenario_bg_policies=[0],
        env_seed=0,
        n_states=5,
        n_actions=3,
        num_players=2,
        episode_length=64,
        history_length=2,
):
    env_config = dict(
        n_states=n_states,
        n_actions=n_actions,
        episode_length=episode_length,
        seed=env_seed,
        num_players=num_players,
        history_length=history_length
    )

    dummy_env = RandomPOMDP(**env_config)

    policies = {
        f"background_{i}": (
            RLlibDeterministicPolicy,
            dummy_env.observation_space,
            dummy_env.action_space,
            dict(
                config=env_config,
                seed=bg_policy_seed
            )

        ) for i, bg_policy_seed in enumerate(scenario_bg_policies)
    }
    policies[Scenario.MAIN_POLICY_ID] = (None, dummy_env.observation_space, dummy_env.action_space, {})

    config = PPOBFSGDAConfig().training(
        beta_lr=2e-2,
        beta_smoothing=2000,
        use_utility=True,

        lambda_=0.95,
        gamma=0.99,
        entropy_coeff=1e-4,
        lr=1e-4,

        kl_coeff=1.,
        kl_target=1e-2,
        clip_param=0.2,
        grad_clip=1.,
        #optimizer="rmsprop",
        train_batch_size=512 * 16,
        sgd_minibatch_size=512,
        num_sgd_iter=32,
        model={
            "fcnet_hiddens"   : [8],
            "fcnet_activation": "relu",
        }
    ).rollouts(
        num_rollout_workers=num_workers,
        sample_async=False,
        create_env_on_local_worker=False,
        num_envs_per_worker=1,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode="truncate_episodes"
    ).environment(
        env="RandomMDP",
        env_config=env_config
    ).resources(num_gpus=1
    ).framework(framework="tf"
    ).multi_agent(
        policies=policies,
        policies_to_train={Scenario.MAIN_POLICY_ID},
    )

    exp = tune.run(
        "PPO",
        name="BF_SGDA_v0.1",
        config=config,
        checkpoint_at_end=False,
        checkpoint_freq=30,
        keep_checkpoints_num=3,
        stop={
            "timesteps_total": 1_000_000_000,
        },
        local_dir="rllib_runs",
        # restore=""
        # resume=True
    )


if __name__ == '__main__':
    fire.Fire(main)

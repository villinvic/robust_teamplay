from ray.rllib.algorithms.ppo import PPOConfig
from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from environments.rllib.random_mdp import RandomPOMDP
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy


def env_maker(env_config):

    return RandomPOMDP(*env_config)

register_env("RandomMDP", RandomPOMDP)

num_workers = 4
rollout_fragment_length = 16


def main(
        scenario_num_copies=1,
        scenario_bg_policies=[0],
        env_seed=0,
        n_states=5,
        n_actions=3,
        num_players=2,
        episode_length=64,
):
    env_config = dict(
        n_states=n_states,
        n_actions=n_actions,
        seed=env_seed,
        num_players=num_players,
        episode_length=episode_length,
    )

    dummy_env = RandomPOMDP(*env_config)

    policies = {
        f"background_{i}": RLlibDeterministicPolicy(
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            config=env_config,
            seed=bg_policy_seed
        ) for i, bg_policy_seed in enumerate(scenario_bg_policies)
    }
    policies["focal"] =  PolicySpec(observation_space=dummy_env.observation_space, action_space=dummy_env.action_space)




    config = PPOConfig().training(
        lambda_=0.95,
        gamma=0.99,
        entropy_coeff=1e-4,
        learner_queue_size=256,
        lr=1e-4,
        statistics_lr=3e-1,
        momentum=0.,
        epsilon=1e-5,
        decay=0.99,
        grad_clip=1.,
        opt_type="rmsprop",
        train_batch_size=512 * 16,
        sgd_minibatch_size=512,
        num_sgd_iter=32,
        model={
            "fcnet_hiddens"   : [64, 64],
            "fcnet_activation": "relu",
        }
    ).rollouts(
        num_rollout_workers=num_workers,
        sample_async=False,
        create_env_on_local_worker=False,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode="truncate_episodes"
    ).environment(
        env="RandomMDP",
        env_config=env_config
    ).resources(num_gpus=1
    ).framework(framework="tf"
    ).multi_agent(
        policies=policies,
        policies_to_train="focal",

    )


if __name__ == '__main__':



tune.run()
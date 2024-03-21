from dataclasses import dataclass, asdict
from typing import Type

from environments.rllib.random_mdp import RandomPOMDP



@dataclass
class EnvConfig:

    _env_name: str = "undefined"


    def as_dict(self):
        return asdict(self)

    def get_env_id(self):
        pass

    def get_maker(self):
        pass


@dataclass
class RandomPOMDPConfig(EnvConfig):

    _env_name: str = "RandomPOMDP"

    seed: int = 0
    n_states: int = 5
    n_actions: int = 3
    num_players: int = 2
    episode_length: int = 100
    history_length: int = 2
    full_one_hot: bool = True

    def as_dict(self):
        d = asdict(self)
        del d["_env_name"]
        return d

    def get_env_id(self):
        config_name = (str(self.as_dict())
                       .replace("'", "")
                       .replace(" ", "")
                       .replace(":", "_")
                       .replace(",", "_")[1:-1]
        )
        return f"{RandomPOMDPConfig._env_name}_{config_name}"


    def get_maker(self):
        def env_maker(config=None):
            return RandomPOMDP(**self.as_dict())

        return env_maker

ENVS = {
    RandomPOMDPConfig._env_name : RandomPOMDPConfig,
}


def get_env_config(environment_name) -> Type[EnvConfig]:
    """
    :param environment_name: base name of the env
    :return: helper function to build config, env maker, and env_id as a function its config
    """
    env = ENVS.get(environment_name, None)
    if env is None:
        raise ValueError(f"Environment '{environment_name}' could not be found, available environments: {list(ENVS.keys())}")
    return ENVS.get(environment_name, None)
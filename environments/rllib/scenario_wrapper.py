from typing import Type

from ray.rllib import MultiAgentEnv


def InfoWrapper(env_cls: Type[MultiAgentEnv]) -> Type[MultiAgentEnv]:
    """
    Multi agent env wrapper
    """

    class InfoWrapper(env_cls):

        info_placeholder = {
            "scenario": 0
        }

        def build_info_dict(self):
            self.infos = {
                agent_id: InfoWrapper.info_placeholder for agent_id in self._agent_ids
            }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.build_info_dict()

        def step(self, action):
            observations, rewards, dones, truncs, _ = self.step(action)

            return observations, rewards, dones, truncs, self.infos.copy()

        def reset(self, *args, **kwargs):
            return self.env.reset(*args, **kwargs), self.infos.copy()

    InfoWrapper.__name__ = env_cls.__name__
    InfoWrapper.__qualname__ =  env_cls.__name__

    return InfoWrapper
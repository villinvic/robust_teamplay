from gymnasium import Wrapper
from ray.rllib import MultiAgentEnv


class InfoWrapper(Wrapper):
    """
    Multi agent env wrapper
    """

    def __init__(self, env: MultiAgentEnv):
        super().__init__(env)
        self.env : MultiAgentEnv
        self.info_placeholder = {
            "scenario": 0
        }

    def step(self, action):
        observations, rewards, dones, truncs, infos = self.env.step(action)

        # Add custom information to the info dictionary
        infos = {
            agent_id: self.info_placeholder for agent_id in self.env._agent_ids
        }

        return observations, rewards, dones, truncs, infos

    def reset(self, **kwargs):
        self.custom_info = {}  # Reset custom info on environment reset
        return self.env.reset(**kwargs)
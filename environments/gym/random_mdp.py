import itertools
from typing import Tuple

from gymnasium import spaces
from ray.rllib import MultiAgentEnv
import numpy as np
from ray.rllib.utils.typing import MultiAgentDict


class RandomMDP(MultiAgentEnv):
    def __init__(
            self,
            episode_length: int = 5,
            n_states: int = 5,
            n_actions: int = 2,
            seed: int = None,
            history_window: int = 3,
            num_players: int = 2
    ):
        self.random = np.random.default_rng(seed=seed)
        self.episode_length = episode_length
        self.history_window = history_window
        self.current_step = 0
        self.current_state = 0
        self.num_players = num_players

        self._agent_ids = {i for i in range(num_players)}

        self.action_space = spaces.Dict(
            {
                i: spaces.Discrete(n_actions) for i in self._agent_ids
            }
        )

        self.observation_space = spaces.Dict(
            {i: spaces.Discrete(n_states) for i in self._agent_ids}
        )

        self.transition_function = {}
        self.reward_function = {}

        action_combinations = itertools.combinations_with_replacement(range(n_actions), num_players)
        for actions in action_combinations:
                rewards = np.random.normal(0, 1, num_players)

                self.reward_function[(n_states-1, frozenset(actions))] =


        self.curr_state_to_opp_state = {i: i for i in range(n_states)}

        self.gamma = 1.
        self.s0 = 0

        self.transition_function /= (self.transition_function.sum(axis=-1, keepdims=True)+1e-8)


    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        actions = action_dict

from typing import Union, List, Optional, Dict, Tuple

import numpy as np
from ray.rllib import Policy, SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType


class RLlibDeterministicPolicy(Policy):

    def __init__(self, observation_space, action_space, config, seed=None):

        self.n_actions = action_space.n
        self.n_states = observation_space.n
        self.history_length = config["history_length"]

        self.policy = np.empty((self.n_actions+1, self.n_states+1) * self.history_length + (self.n_states,) , dtype=np.int8)
        
        super().__init__(observation_space, action_space, config)

        self.initialize(seed=seed)


    def initialize(self, seed=None):
        random = np.random.default_rng(seed=seed)
        self.policy[:] = random.integers(0, self.n_actions, self.policy.shape)

    def compute_single_action(
        self,
        obs: Optional[TensorStructType] = None,
        state: Optional[List[TensorType]] = None,
        *,
        prev_action: Optional[TensorStructType] = None,
        prev_reward: Optional[TensorStructType] = None,
        info: dict = None,
        input_dict: Optional[SampleBatch] = None,
        episode: Optional["Episode"] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:

        action = self.policy[*state, obs]

        state.pop(0)
        state.append((obs, action))

        return action, state, {}

    def get_initial_state(self) -> List[TensorType]:
        return [(self.n_states, self.n_actions) for _ in range(self.history_length)]


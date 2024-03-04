from typing import Union, List, Optional, Dict, Tuple

import numpy as np
from gymnasium.spaces import MultiDiscrete
from ray.rllib import Policy, SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType


class RLlibDeterministicPolicy(Policy):

    def __init__(self, observation_space, action_space, config, seed=None):

        self.n_actions = action_space[0].n
        self.state_shape: MultiDiscrete = observation_space[0].nvec
        self.policy = np.empty(self.state_shape + (self.n_actions,) , dtype=np.int8)
        
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

        action = self.policy[obs]

        return action, state, {}

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        actions = self.policy[obs_batch]

        return actions, state_batches, {}

    # def get_initial_state(self) -> List[TensorType]:
    #     return [(self.n_states, self.n_actions) for _ in range(self.history_length)]


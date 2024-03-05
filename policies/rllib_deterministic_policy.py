from typing import Union, List, Optional, Dict, Tuple

import numpy as np
from gymnasium.spaces import MultiDiscrete
from ray.rllib import Policy, SampleBatch
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.utils.typing import TensorStructType, TensorType, ModelWeights


def restore_obs(obs, space):
    if isinstance(space, MultiDiscrete):
        shape = space.nvec
        indices = np.transpose(np.where(obs == 1)[1])
        B = obs.shape[0]

        offset = np.roll(shape, shift=1)
        offset[0] = 0
        np.cumsum(offset, out=offset)

        return np.stack(np.split(indices, B)) - offset[np.newaxis]



    else:
        return obs

class RLlibDeterministicPolicy(Policy):

    def __init__(self, observation_space, action_space, config, seed=None):

        super().__init__(observation_space, action_space, config)

        self.n_actions = action_space.n
        self.observation_space = observation_space.original_space

        if isinstance(self.observation_space, MultiDiscrete):
            self.state_shape = self.observation_space.nvec

        else:
            self.state_shape = self.observation_space.shape

        self.policy = np.empty(self.state_shape, dtype=np.int8)

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

        action = self.policy[restore_original_dimensions(obs, self.observation_space)]

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

        original_obs = restore_obs(obs_batch, self.observation_space)

        actions = [self.policy[tuple(obs)] for obs in original_obs]

        return actions, state_batches, {}

    # def get_initial_state(self) -> List[TensorType]:
    #     return [(self.n_states, self.n_actions) for _ in range(self.history_length)]

    def get_weights(self) -> ModelWeights:
        return {
            "weights": self.policy.copy()
        }

    def set_weights(self, weights: ModelWeights) -> None:
        self.policy[:] = weights["weights"]






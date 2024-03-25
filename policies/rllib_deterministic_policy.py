from typing import Union, List, Optional, Tuple
from typing import Dict as Dict_t
import numpy as np
from gymnasium.spaces import MultiDiscrete, Discrete, Dict
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

    elif isinstance(space, Discrete):
        return np.where(obs == 1)[1][:, np.newaxis]


    else:
        return obs

class RLlibDeterministicPolicy(Policy):

    def __init__(self, observation_space, action_space, config, seed=None):

        super().__init__(observation_space, action_space, config)

        self.n_actions = action_space.n
        self.observation_space = observation_space if not hasattr(observation_space, "original_space")\
            else observation_space.original_space

        self.dict_obs = False
        if isinstance(self.observation_space, MultiDiscrete):
            self.state_shape = self.observation_space.nvec
        elif isinstance(self.observation_space, Discrete):
            self.state_shape = (self.observation_space.n,)
        elif isinstance(self.observation_space, Dict):
            self.dict_obs = True
            self.state_shape = (self.observation_space[SampleBatch.OBS].n,)
        else:
            self.state_shape = self.observation_space.shape

        self.policy = np.empty(self.state_shape, dtype=np.int8)

        self.initialize(seed=seed)

    def initialize(self, seed=None):
        random = np.random.default_rng(seed=seed)
        self.policy[:] = random.integers(0, self.n_actions, self.policy.shape)

    def get_action(self, obs):
        if self.dict_obs:
            obs = obs[SampleBatch.OBS]
        return self.policy[obs]

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict_t[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict_t[str, TensorType]]:
        if self.dict_obs:
            actions = self.policy[obs_batch[SampleBatch.OBS]]

        else:
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






import gymnasium
import numpy as np

from policies.policy import Policy


class TabularPolicy(Policy):

    def __init__(self, environment: gymnasium.Env):
        super().__init__(environment)

        self.action_probs = np.full((self.n_states, self.n_actions), fill_value=np.nan, dtype=np.float16)


class SingleStatePolicy(Policy):

    def __init__(self, environment: gymnasium.Env, bound_state=0):
        super().__init__(environment)

        self.bound_state = bound_state

        self.action_probs = np.full((self.n_actions,), fill_value=np.nan, dtype=np.float16)

    def make_deterministic(self, action_idx):

        self.action_probs[:] = 0
        self.action_probs[action_idx] = 1



from gymnasium import spaces
from ray.rllib import MultiAgentEnv
import numpy as np

class RandomMDP2P(MultiAgentEnv):
    def __init__(
            self,
            episode_length: int = 5,
            n_states: int = 5,
            n_actions: int = 2,
            seed: int = None,
            history_window: int = 3,
    ):
        self.random = np.random.default_rng(seed=seed)
        self.episode_length = episode_length
        self.history_window = history_window

        self.current_step = 0

        self._agent_ids = {0, 1}


        self.action_space = spaces.Dict(
            {
                i: spaces.Discrete(n_actions) for i in self._agent_ids
            }
        )

        self.observation_space = spaces.Dict(
            {
                i: spaces.Discrete(
                    sum([(n_states * n_actions**2)**t for t in range(history_window+1)])
                ) for i in self._agent_ids
            }
        )
        self.transition_function = np.zeros(
            (self.observation_space[0].n, self.action_space[0].n, self.action_space[0].n, self.observation_space[0].n),
            dtype=np.float32
        )
        self.reward_function = np.zeros((self.observation_space[0].n, self.action_space[0].n, self.action_space[0].n),
                                        dtype=np.float32)

        self.historyless_transition_function = self.random.exponential(1, (n_states, n_actions, n_actions, n_states))
        self.historyless_transition_function /= (self.historyless_transition_function.sum(axis=-1, keepdims=True)+1e-8)
        self.historyless_reward_function = self.random.uniform(0., 1.,(n_states, n_actions, n_actions))

        self.curr_state_to_opp_state = {}

        def get_state_index(history):

            idx = 0
            for i, (s, a1, a2) in enumerate(history):
                idx +=  (n_states * n_actions) ** i * (1 + s + n_states * a1 + n_states * n_actions * a2)

            idx_opp = 0
            for i, (s, a1, a2) in enumerate(history):
                idx_opp +=  (n_states * n_actions) ** i * (1 + s + n_states * a2 + n_states * n_actions * a1)
            self.curr_state_to_opp_state[idx] = idx_opp

            return idx




        def build_env(acc, depth=0):

            curr_state_index = get_state_index(acc)
            history_length = len(acc)

            if history_length == 0:
                underlying_past_state =  0
            else:
                underlying_past_state = acc[-1][0]

            for next_state in range(n_states):
                for action1 in range(n_actions):
                    for action2 in range(n_actions):
                        if history_length == history_window:
                            next_acc = acc[1:]
                        else:
                            next_acc = acc

                        next_acc += ((next_state, action1, action2),)
                        next_state_index = get_state_index(next_acc)

                        self.transition_function[curr_state_index, action1, action2, next_state_index] = (
                            self.historyless_transition_function[underlying_past_state, action1, action2, next_state]
                        )
                        self.reward_function[curr_state_index, action1, action2] = self.historyless_reward_function[underlying_past_state, action1, action2]

                        if depth <= history_length:
                            build_env(next_acc, depth+1)

        build_env(())


        self.gamma = 1.
        self.s0 = 0

        self.transition_function /= (self.transition_function.sum(axis=-1, keepdims=True)+1e-8)

        super(RandomMDP2P, self).__init__()


class HistorylessRandomMDP2P(MultiAgentEnv):
    def __init__(
            self,
            episode_length: int = 5,
            n_states: int = 5,
            n_actions: int = 2,
            seed: int = None,
            *args,
            **kwargs,
    ):
        self.random = np.random.default_rng(seed=seed)
        self.episode_length = episode_length

        self.current_step = 0

        self._agent_ids = {0, 1}

        self.action_space = spaces.Dict(
            {
                i: spaces.Discrete(n_actions) for i in self._agent_ids
            }
        )

        self.observation_space = spaces.Dict(
            {
                i: spaces.Discrete(
                    n_states
                ) for i in self._agent_ids
            }
        )
        self.transition_function = np.zeros((n_states, n_actions, n_actions, n_states), dtype=np.float32)
        self.reward_function = np.zeros((self.observation_space[0].n, self.action_space[0].n, self.action_space[0].n),
                                        dtype=np.float32)
        for action1 in range(n_actions):
            for action2 in range(n_actions):
                p = self.random.exponential(1, (n_states, n_states))
                self.transition_function[:, action1, action2] = p
                self.transition_function[:, action2, action1] = p
                self.reward_function[-1, action1, action2] = 1
                self.reward_function[-1, action2, action1] = 1

        self.curr_state_to_opp_state = {i: i for i in range(n_states)}

        self.gamma = 1.
        self.s0 = 0


        self.transition_function /= (self.transition_function.sum(axis=-1, keepdims=True)+1e-8)

        print("transition", self.transition_function[0, :, :, 0], "rewards", self.reward_function)

        super(HistorylessRandomMDP2P, self).__init__()


if __name__ == '__main__':

    mdp = RandomMDP2P(n_states=2, n_actions=2, history_window=1)

    print(mdp.transition_function[4, :, 0, :])

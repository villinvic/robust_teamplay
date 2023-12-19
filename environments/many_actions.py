from gymnasium import spaces
import numpy as np

class ManyActionEnv:

    def __init__(self, n_actions):

        self._agent_ids = {0, 1}

        self.action_space = spaces.Dict(
            {
                i: spaces.Discrete(n_actions) for i in self._agent_ids
            }
        )

        n_states = 1 + n_actions

        self.observation_space = spaces.Dict(
            {
                i: spaces.Discrete(n_states) for i in self._agent_ids
            }
        )

        self.s0 = 0
        self.s_terminal = n_states - 1
        self.transition_function = np.zeros((n_states, n_actions, n_actions, n_states), dtype=np.float32)
        #self.transition_function[0, :, :, 1] = 1.

        self.reward_function = np.zeros((n_states, n_actions, n_actions), dtype=np.float32)

        for state in range(n_states - 1):
            for action in range(n_actions):
                for action2 in range(n_actions):
                    if state == action or state == action2:
                        if action == action2:
                            self.reward_function[state, action, action2] = action + 1
                        self.transition_function[state, action, action2, self.s_terminal] = 1.

                    elif action == action2:
                        self.transition_function[state, action, action2, state+1] = 1.

                    else:
                        self.transition_function[state, action, action2, self.s_terminal] = 1.

        # print(self.transition_function, self.reward_function)
        # input()
        self.gamma = 1.

        self.episode_length = 1
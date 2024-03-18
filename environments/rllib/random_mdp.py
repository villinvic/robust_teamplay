import itertools
import time
from collections import defaultdict
from copy import copy
from typing import Tuple, Optional

from gymnasium import spaces
from ray.rllib import MultiAgentEnv
import numpy as np
from ray.rllib.utils.typing import MultiAgentDict


class RandomPOMDP(MultiAgentEnv):
    def __init__(
            self,
            episode_length: int = 5,
            n_states: int = 5,
            n_actions: int = 2,
            seed: int = None,
            num_players: int = 2,
            history_length: int = 1,
            full_one_hot: bool = True,
            **kwargs
    ):

        if len(kwargs)> 0:
            print("non understood env args:", kwargs)

        self.random = np.random.default_rng(seed=seed)
        self.episode_length = episode_length
        self.current_step = 0
        self.current_state = 0
        self.n_states = n_states
        self.n_actions = n_actions
        self.num_players = num_players
        self.history_length = history_length
        self.full_one_hot = full_one_hot
        self._obs_space_in_preferred_format = True

        self._agent_ids = {i for i in range(num_players)}

        self.action_space = spaces.Dict(
            {
                i: spaces.Discrete(n_actions) for i in self._agent_ids
            }
        )

        self.base_shape = (self.n_actions,) * self.history_length + (self.n_states,) * (self.history_length + 1)
        if full_one_hot:
            obs_space = spaces.Discrete(np.prod(self.base_shape))
            print(obs_space)
        else:
            obs_space = spaces.MultiDiscrete(self.base_shape)

        self.observation_space = spaces.Dict(
            {i: obs_space
             for i in self._agent_ids}
        )

        self.reset_history()

        self.transition_function = {}
        self.reward_function = {}

        action_combinations = itertools.combinations_with_replacement(range(n_actions), num_players)
        for actions in action_combinations:
            for ordered_actions in itertools.permutations(actions, num_players):
                for player_states in itertools.combinations_with_replacement(range(n_states), num_players):
                    for ordered_player_states in itertools.permutations(player_states, num_players):

                        player_tuples = tuple(sorted([
                            (s, a) for s,a in zip(ordered_player_states, ordered_actions)
                        ]))

                        for player_tuple in player_tuples:

                            transition_probs = self.random.exponential(1, self.n_states)
                            transition_probs[:] = transition_probs / transition_probs.sum()
                            self.transition_function[
                                player_tuple, player_tuples
                            ] = transition_probs

                            state, _ = player_tuple
                            if state == n_states-1:
                                self.reward_function[player_tuple, player_tuples] = self.random.normal(0, 1)
                            else:
                                self.reward_function[player_tuple, player_tuples] = 0.

        self.gamma = 1.
        self.s0 = 0

        super().__init__()

    def reset_history(self):
        self.past_actions = {
            i: [0 for _ in range(self.history_length)] for i in self._agent_ids
        }
        self.past_states = {
            i: [0 for _ in range(self.history_length)] for i in self._agent_ids
        }

    def update_history(self, actions):
        for i in self._agent_ids:
            self.past_actions[i].pop(0)
            self.past_actions[i].append(actions[i])
            self.past_states[i].pop(0)
            self.past_states[i].append(self.player_states[i])


    def get_state_index(self, player_idx):
        last_offset = 1
        index = 0
        for v, offset in zip(self.past_actions[player_idx]
                             + self.past_states[player_idx]
                             + [self.player_states[player_idx]],
                             self.base_shape):
            index += v * last_offset
            last_offset *= offset

        return index


    def get_state(self):
        if self.full_one_hot:
            s = {i: self.get_state_index(i)
                 for i in self._agent_ids}

        else:
            s = {
                i: np.array(self.past_actions[i] + self.past_states[i] + [self.player_states[i]], dtype=np.int64)
                for i in self._agent_ids
            }

        assert self.observation_space.contains(s), s
        return s

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        self.current_step = 0
        self.player_states = {p_id: 0 for p_id in self._agent_ids}
        self.reset_history()

        return self.get_state(), {}

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:

        player_tuples = tuple(sorted([
            (s, a) for s, a in zip(self.player_states.values(), action_dict.values())
        ]))

        self.current_step += 1

        rewards = {}
        for i, (agent_id, action) in enumerate(action_dict.items()):
            player_state = self.player_states[agent_id]
            next_player_state = self.random.choice(self.n_states, p=self.transition_function[(player_state, action), player_tuples])
            self.player_states[agent_id] = next_player_state
            rewards[agent_id] = self.reward_function[(player_state, action), player_tuples]

        done = self.current_step >= self.episode_length
        dones = {
            agent_id: done for agent_id in self._agent_ids
        }
        dones["__all__"] = done

        state = self.get_state()
        self.update_history(action_dict)

        return state, rewards, dones, dones, {}






if __name__ == '__main__':

    np.random.seed(0)
    n_states = 5
    n_actions = 3
    p1 = np.random.random((n_states, n_actions))
    p2 = np.random.random((n_states, n_actions))
    p3 = np.random.random((n_states, n_actions))

    players = [p2, p3]

    for p in players:
        p[:] = p / p.sum(axis=1, keepdims=True)

    env = RandomPOMDP(
        history_length=2,
        n_states=n_states, n_actions=n_actions, num_players=len(players), episode_length=64, seed=0)

    rewards = defaultdict(int)

    obs, _ = env.reset()
    done = {"__all__": False}
    states = []
    while not done["__all__"]:
        # print({
        #     p_id: #np.random.choice(n_actions, p=p[obs[p_id]]) for p_id, p in enumerate(players)
        #         np.argmax(p[obs[p_id]]) for p_id, p in enumerate(players)
        # })
        obs, step_rewards, done, trunc, info = env.step({
            p_id: #np.random.choice(n_actions, p=p[obs[p_id]]) for p_id, p in enumerate(players)
                np.argmax(p[obs[p_id]]) for p_id, p in enumerate(players)
        })

        states.append(list(obs.values()))

        for p_id, r in step_rewards.items():
            rewards[p_id] += r

    print(rewards)



import itertools
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
            **kwargs
    ):
        self.random = np.random.default_rng(seed=seed)
        self.episode_length = episode_length
        self.current_step = 0
        self.current_state = 0
        self.n_states = n_states
        self.n_global_states = sum([self.n_states ** (i+1) for i in range(num_players)])
        self.n_actions = n_actions
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


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        self.current_step = 0
        self.player_states = {p_id: 0 for p_id in self._agent_ids}

        return copy(self.player_states), {}

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


        states = copy(self.player_states)

        done = self.current_step >= self.episode_length
        dones = {
            agent_id: done for agent_id in self._agent_ids
        }
        dones["__all__"] = done

        return states, rewards, dones, dones, {}






if __name__ == '__main__':

    np.random.seed(0)
    n_states = 5
    n_actions = 3
    p1 = np.random.random((n_states, n_actions))
    p2 = np.random.random((n_states, n_actions))
    p3 = np.random.random((n_states, n_actions))


    players = [p1, p2, p3]

    for p in players:
        p[:] = p / p.sum(axis=1, keepdims=True)


    env = RandomPOMDP(n_states=n_states, n_actions=n_actions, num_players=len(players), episode_length=50000, seed=0)

    rewards = defaultdict(int)

    obs, _ = env.reset()
    done = {"__all__":False}
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



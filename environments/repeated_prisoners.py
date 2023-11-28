import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Any, TypedDict

from gymnasium.core import ObsType
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class StateTreeNode:
    def __init__(self, index):
        self.index = index
        self.children = [None, None, None, None]

# Build the tree
def build_tree(depth):
    if depth == 0:
        return None
    root = StateTreeNode(0)
    for i in range(4):
        root.children[i] = build_tree_recursive(root.index * 4 + i + 1, depth - 1)
    return root

def build_tree_recursive(index=0, depth=5):
    if depth == 0:
        return StateTreeNode(index)
    node = StateTreeNode(index)
    for i in range(4):
        node.children[i] = build_tree_recursive(index * 4 + i + 1, depth - 1)
    return node


def navigate_tree(node, actions):
    idx = actions[0] + actions[1] * 2
    return node.children[idx]


class RepeatedPrisonersDilemmaEnv(MultiAgentEnv):
    def __init__(self, episode_length: int):

        self.episode_length = episode_length
        self.current_step = 0
        self.max_reward = 5  # Maximum reward for each player
        self._agent_ids = {0, 1}

        # Action space: both players can either cooperate (0) or defect (1)
        self.action_space = spaces.Dict(
            {
                i: spaces.Discrete(2) for i in self._agent_ids
            }
        )

        # Observation space: one-hot encoding of the state index
        self.observation_space = spaces.Dict(
            {
                i: spaces.Discrete(sum([4**(t) for t in range(episode_length)])) for i in self._agent_ids
            }
        )

        # History of actions
        self.tree = build_tree_recursive(depth=episode_length)
        self.transition_function = np.zeros(
            (self.observation_space[0].n, self.action_space[0].n, self.action_space[0].n, self.observation_space[0].n), dtype=np.float16
        )

        def setup_transitions_rec(node=self.tree, depth=self.episode_length):
            if depth == 1:
                return

            else:
                for action1 in range(self.action_space[0].n):
                    for action2 in range(self.action_space[0].n):
                        idx = action1 + 2 * action2
                        next_node = node.children[idx]
                        self.transition_function[node.index, action1, action2, next_node.index] = 1
                        setup_transitions_rec(next_node, depth-1)

        setup_transitions_rec()

        self.curr_nodes = [None, None]

        super(RepeatedPrisonersDilemmaEnv, self).__init__()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.curr_nodes = [self.tree, self.tree]
        return self._get_observation()

    def step(self, actions: dict):

        # Update the state based on actions
        for i, curr_node in enumerate(self.curr_nodes):
            a = actions.values()
            if i == 1:
                a = reversed(a)
            self.curr_nodes[i] = navigate_tree(curr_node, list(a))

        reward = self._calculate_reward(actions)

        # Move to the next step
        self.current_step += 1

        # Check if it's the last step of the episode
        done = self.current_step == self.episode_length

        obs = {
            i: 0 for i in self._agent_ids
        } if done else self._get_observation()

        return obs, reward, done, done, {}


    def _get_observation(self):

        return {
            i: curr_node.index for i, curr_node in enumerate(self.curr_nodes)
        }

    def _calculate_reward(self, actions: dict) -> Tuple[float, float]:
        # Calculate the reward based on the actions

        if actions[0] == 0 and actions[1] == 0:
            return (self.max_reward - 1, self.max_reward - 1)  # Both players cooperate
        elif actions[0] == 0 and actions[1] == 1:
            return (0, self.max_reward)  # Player 1 cooperates, player 2 defects
        elif actions[0] == 1 and actions[1] == 0:
            return (self.max_reward, 0)  # Player 1 defects, player 2 cooperates
        else:
            return (1, 1)  # Both players defect


if __name__ == "__main__":

    # Example usage:
    episode_length = 2
    env = RepeatedPrisonersDilemmaEnv(episode_length)

    # Reset the environment
    obs = env.reset()

    for _ in range(episode_length):
        # Take random actions for both players
        print(obs)
        actions = env.action_space.sample()
        # Step through the environment
        obs, reward, done, _, _ = env.step({
            0:1,
            1:1
        })
        #print(f"Step: {env.current_step}, Actions: {actions}, Reward: {reward}, Done: {done}")

    # Close the environment
    env.close()

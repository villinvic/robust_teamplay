from collections import defaultdict

import numpy as np

from beliefs.rllib_scenario_distribution import Scenario
from environments.rllib.random_mdp import RandomPOMDP
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy


def force_tuple(data):
    if isinstance(data, (int, np.int64)):
        return (data,)

    return tuple(data)

class SARSA:

    def __init__(self, environment, lr=1e-3, epsilon=1., epsilon_min=0.01, epsilon_decay=0.9999):

        self.n_states = environment.n_states
        self.n_actions = environment.n_actions
        self.env = environment
        self.epsilon = epsilon
        self.lr = lr

        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def decay_epsilon(self):

        self.epsilon = np.maximum(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy_value(self, Q):

        return np.max([
            Q[self.env.s0, a] for a in range(self.n_actions)
        ])

    def get_action(self, Q, state):

        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)

        return np.argmax([
            Q[state, a] for a in range(self.n_actions)
        ])

    def learn(self, scenario: Scenario):
        Q = defaultdict(float)

        players = scenario.get_policies()

        old_Q = None
        iter = 0
        average_utility = 0
        stat_lr = 0.9999
        while old_Q != Q:
            old_Q = Q.copy()

            episode_r = self.run_episode_and_learn(Q, players)
            self.decay_epsilon()

            average_utility = average_utility * stat_lr + episode_r * (1 - stat_lr)

            iter += 1
            print(f"Iter {iter}, V(pi)={self.get_policy_value(Q)}, episodic_r={average_utility} epsilon={self.epsilon}")

            v = np.array(list(Q.values()))
            print(len(np.argwhere(np.nonzero(v))))

        return self.get_pi(Q)

    def run_episode_and_learn(self, Q, players):

        done = False
        obss, _ = self.env.reset()

        num_focal = sum([int(p  in (Scenario.MAIN_POLICY_COPY_ID, Scenario.MAIN_POLICY_ID)) for p in players])
        r = 0
        while not done:

            actions = {
                i: self.get_action(Q, force_tuple(obs))
                if player in (Scenario.MAIN_POLICY_COPY_ID, Scenario.MAIN_POLICY_ID)
                else
                player.get_action(force_tuple(obs))

                for (i, obs), player in zip(obss.items(), players)
            }

            obss_, rewards, dones, truncs, infos = self.env.step(actions)

            done = dones["__all__"]

            for i, player in zip(self.env._agent_ids, players):
                if player in (Scenario.MAIN_POLICY_COPY_ID, Scenario.MAIN_POLICY_ID):
                    self.update(Q, actions[i], obss[i], rewards[i], obss_[i], done)
                    r += rewards[i]
            obss = obss_

        return r / num_focal

    def update(self, Q, action, state, reward, next_state, done):

        next_action = self.get_action(Q, force_tuple(next_state))

        Q[force_tuple(state), action] = Q[force_tuple(state), action] + self.lr * (
                    reward + done * Q[force_tuple(next_state), next_action] - Q[force_tuple(state), action])

    def get_pi(self, Q):

        pi = np.zeros((self.n_states, self.n_actions), np.int8)

        for state in range(self.n_states):
            a_star = np.argmax([
                Q[state, action] for action in range(self.n_actions)
            ])

            pi[state, a_star] = 1

        return pi


if __name__ == '__main__':
    # run algo against set seed opponents and env, compare with RLLIB

    # TODO TEST num possible observations !

    env_config = dict(
        seed=0,
        n_states=5,
        n_actions=3,
        num_players=2,
        episode_length=100,
        history_length=2,
        full_one_hot=True
    )

    env = RandomPOMDP(**env_config)

    opponent = RLlibDeterministicPolicy(
        env.observation_space[0], env.action_space[0], {}, seed=0
    )

    algo = SARSA(env)

    s = Scenario(1, [opponent])
    # [1, 0] ~1.7
    # [1, 1] ~ 1.4 ?
    # [2, _] ~ 8.3?

    pi = algo.learn(scenario=s)

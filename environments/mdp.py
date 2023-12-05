import numpy as np

def compute_multiagent_mdp(transition_function, reward_function, policy, joint_rewards=False):
    # two players
    # transition_function (state, action1, action2, next_state)
    state_dim = transition_function.shape[0]
    action_dim =  transition_function.shape[1]
    single_agent_transition_function = np.zeros((state_dim, action_dim, state_dim))
    single_agent_reward_function = np.zeros((state_dim, action_dim))


    for state in range(state_dim):
        for action1 in range(action_dim):
            for action2 in range(action_dim):
                for next_state in range(state_dim):
                    single_agent_transition_function[state, action1, next_state] += (
                            policy[state, action2] * transition_function[state, action1, action2, next_state]
                    )

                if joint_rewards:
                    single_agent_reward_function[state, action1] +=  0.5 * (
                            policy[state, action2] * reward_function[state, action1, action2]
                    ) + 0.5 * (
                            policy[state, action1] * reward_function[state, action2, action1]
                    )
                else:

                    single_agent_reward_function[state, action1] += (
                            policy[state, action2] * reward_function[state, action1, action2]
                    )

    return single_agent_transition_function, single_agent_reward_function




if __name__ == "__main__":

    players = 2
    actions = 2
    states = 5

    transition_function = np.random.random((states,) + (actions,)*players + (states,))
    normalization = np.sum(transition_function, axis=0, keepdims=True)
    for i in range(players):
        normalization = np.sum(normalization, axis=i+1, keepdims=True)
    transition_function /= normalization
    policy = np.random.random((states, actions))
    policy /= np.sum(policy, axis= 1, keepdims=True)

    single_agent_tf_1 = compute_multiagent_mdp(transition_function, policy)
    print(single_agent_tf_1.shape, transition_function.shape)
    print(np.sum(single_agent_tf_1[:, :, 0]))
import gymnasium
import numpy as np


class Policy:

    def __init__(self, environment: gymnasium.Env):

        self.n_actions = environment.action_space[0].n
        self.n_states = environment.observation_space[0].n

        self.action_logits: np.ndarray = np.full((self.n_states, self.n_actions), fill_value=np.nan, dtype=np.float32)

        self.environment = environment


    def initialize_uniformly(self):

        self.action_logits[:] = 0

    def initialize_randomly(self):

        self.action_logits[:] = np.random.normal(0, 0.2, self.action_logits.shape)


    def sample_action(self, state):

        return np.random.choice(self.n_actions, p=self.get_probs()[state])

    def sample_deterministic_action(self, state):
        return np.argmax(self.action_logits[state])

    def get_params(self):
        return self.action_logits

    def get_probs(self, explore=True):
        exp = np.exp(self.action_logits - self.action_logits.max(axis=-1, keepdims=True))
        p = exp / exp.sum(axis=-1, keepdims=True)

        return p

    def compute_pg(self, Q, V, transition_function, lambda_):

        action_probs = self.get_probs()

        state_visitation = self.get_state_visitation(transition_function)

        #log_pi_grad = np.zeros((self.n_states, self.n_actions, self.n_states, self.n_actions), dtype=np.float32)

        # for state1 in range(self.n_states):
        #     for action1 in range(self.n_actions):
        #         for state2 in range(self.n_states):
        #             for action2 in range(self.n_actions):
        #                 if state1 != state2:
        #                     grad = 0.
        #
        #                 else:
        #                     if action1 == action2:
        #                         grad = 1- action_probs[state1, action1]
        #                     else:
        #                         grad = 0.#-action_probs[state1, action1]
        #
        #                 log_pi_grad[state1, action1, state2, action2] = grad

        A = Q - V[:, np.newaxis]
        #gradients =  (log_pi_grad * Q[:, :, np.newaxis, np.newaxis] * state_visitation[:, np.newaxis, np.newaxis, np.newaxis]).sum(axis=-1).sum(axis=-1)
        #gradients = state_visitation[:, np.newaxis] * advantages *

        gradients = state_visitation[:, np.newaxis] * action_probs * (A - lambda_ * np.log(action_probs+1e-8))

        return gradients

    def apply_gradient(self, gradient, lr, normalize=True):
        if normalize:
            mean_grad = np.mean(gradient)
            std = np.maximum(np.std(gradient), 1e-2)
            gradient = (gradient - mean_grad) / std

        self.action_logits[:] = lr * gradient  + self.action_logits


    def get_state_visitation(self, transition_function):

        max_iterations = self.environment.episode_length
        state_frequencies = np.zeros(self.n_states)
        state_frequencies[self.environment.s0] = 1.
        action_probs = self.get_probs()

        for iteration in range(max_iterations):
            # Store the current state frequencies for comparison
            #prev_state_frequencies = np.copy(state_frequencies)

            # Update state frequencies using the transition function and policy
            state_frequencies[:] += ((state_frequencies[:, np.newaxis, np.newaxis] * action_probs[:, :, np.newaxis] * transition_function)
                                     .sum(axis=0).sum(axis=0))
            # for s_prime in range(self.n_states):
            #     for s in range(self.n_states):
            #         for a in range(self.n_actions):
            #             state_frequencies[s_prime] += state_frequencies[s] * action_probs[s, a] * transition_function[s, a, s_prime]

        state_frequencies /= np.sum(state_frequencies)

        return state_frequencies

    def  get_state_action_visitation(self, transition_function):
        action_probs = self.get_probs()
        state_frequencies = self.get_state_visitation(transition_function)
        state_action_frequencies = state_frequencies[:, np.newaxis] * action_probs

        return state_action_frequencies




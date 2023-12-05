import gymnasium
import numpy as np


class Policy:

    def __init__(self,environment: gymnasium.Env):

        self.n_actions = environment.action_space[0].n
        self.n_states = environment.observation_space[0].n

        self.action_logits: np.ndarray = None


    def sample_action(self, state):

        return np.random.choice(self.n_actions, p=self.get_probs()[state])

    def sample_deterministic_action(self, state):
        return np.argmax(self.action_logits[state])

    def get_params(self):
        return self.action_logits

    def get_probs(self):
        exp = np.exp(self.action_logits - self.action_logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keep_dims=True)


import gymnasium
import numpy as np


class Policy:

    def __init__(self,environment: gymnasium.Env):

        self.n_actions = environment.action_space[0].n
        self.n_states = environment.observation_space[0].n

        self.action_logits: np.ndarray = np.full((self.n_states, self.n_actions), fill_value=np.nan, dtype=np.float32)


    def initialize_uniformly(self):

        self.action_logits[:] = 0

    def initialize_randomly(self):

        self.action_logits[:] = np.random.random(self.action_logits.shape)


    def sample_action(self, state):

        return np.random.choice(self.n_actions, p=self.get_probs()[state])

    def sample_deterministic_action(self, state):
        return np.argmax(self.action_logits[state])

    def get_params(self):
        return self.action_logits

    def get_probs(self):
        exp = np.exp(self.action_logits - self.action_logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def apply_loss(self, loss, lr):

        v = self.get_probs()

        gradients = v * (1 - v) * loss

        self.action_logits[:] = lr * gradients + self.action_logits


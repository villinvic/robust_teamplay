import numpy as np


class Prior:

    def __init__(self, dim, learning_rate=5e-2):
        self.dim = dim
        self.beta_logits: np.ndarray
        self.learning_rate = learning_rate

    def initialize_uniformly(self):

        self.beta_logits = np.full((self.dim,), fill_value=0, dtype=np.float32)

    def initialize_certain(self, idx=0):

        self.beta_logits = np.full((self.dim,), fill_value=-1e3, dtype=np.float32)
        self.beta_logits[idx] = 1e3

    def get_probs(self):
        exp = np.exp(self.beta_logits - self.beta_logits.max())
        return exp / exp.sum()

    def __call__(self):
        return self.get_probs()

    def update_prior(self, gradient, regret=True):

        # Score is the value of the policy
        # Thus, we want to minimize it, ie maximize regret

        if not regret:
            gradient = - gradient

        self.beta[:] = self.beta * np.exp(self.learning_rate * gradient)



if __name__ == "__main__":

    prior = Prior(5)

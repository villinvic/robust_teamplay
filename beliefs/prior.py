import numpy as np


class Prior:

    def __init__(self, dim, learning_rate=5e-2):
        self.dim = dim
        self.beta_logits : np.ndarray
        self.learning_rate = learning_rate

    def initialize_uniformly(self):

        self.beta_logits = np.full((self.dim,), fill_value=0, dtype=np.float32)

    def initialize_randomly(self):

        self.beta_logits = np.random.random((self.dim,))

    def initialize_certain(self, idx=0):

        self.beta_logits = np.full((self.dim,), fill_value=-1e3, dtype=np.float32)
        self.beta_logits[idx] = 1e3

    def get_probs(self):
        exp = np.exp(self.beta_logits - self.beta_logits.max())
        return exp / exp.sum()

    def __call__(self):
        return self.get_probs()

    def update_prior(self, loss, regret=True):

        # Score is the value of the policy
        # Thus, we want to minimize it, ie maximize regret

        if not regret:
            loss = - loss

        v = self.get_probs()

        gradients = v * (1 - v) * loss

        self.beta_logits[:] = self.learning_rate * gradients + self.beta_logits



if __name__ == "__main__":

    prior = Prior(5, learning_rate=1e-1)
    prior.initialize_randomly()

    loss = 1 + np.arange(5)

    print(loss)

    for j in range(100):

        prior.update_prior(loss)

        print(prior.get_probs(), prior.beta_logits)


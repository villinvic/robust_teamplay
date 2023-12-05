import numpy as np


class Prior:

    def __init__(self, dim, learning_rate=5e-2):
        self.dim = dim
        self.beta: np.ndarray
        self.learning_rate = learning_rate

    def initialize_uniformly(self):

        self.beta = np.full((self.dim,), fill_value=1/self.dim, dtype=np.float32)

    def initialize_certain(self, idx=0):

        self.beta = np.zeros((self.dim,), dtype=np.float32)
        self.beta[idx] = 1.



    def __call__(self):
        return self.beta

    def project(self):

        self.beta[:] /= self.beta.sum()


    def update_prior(self, gradient, regret=True):

        # Score is the value of the policy
        # Thus, we want to minimize it, ie maximize regret

        if not regret:
            gradient = - gradient

        self.beta[:] = self.beta * np.exp(self.learning_rate * gradient)



if __name__ == "__main__":

    prior = Prior(5)

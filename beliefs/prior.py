import numpy as np


class Prior:

    def __init__(self, dim, learning_rate=1e-3):
        self.dim = dim
        self.beta: np.ndarray
        self.learning_rate = learning_rate

    def init_uniformly(self):

        self.beta = np.full((self.dim,), fill_value=1/self.dim, dtype=np.float16)


    def __call__(self):
        return self.beta

    def project_prior(self):

        min_beta = np.min(self.beta)
        if min_beta < 0:
            self.beta += -min_beta + 1e-8
        self.beta /= np.sum(self.beta)


    def update_prior(self, score, regret=True):
        if regret:
            score = - score

        self.beta += self.learning_rate * score * self.beta



if __name__ == "__main__":

    prior = Prior(5)

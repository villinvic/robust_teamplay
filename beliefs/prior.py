import numpy as np


def project_onto_simplex(v, z=1):
    n_features = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class Prior:

    def __init__(self, dim, learning_rate=5e-2, smoothing_window=2000):
        self.dim = dim
        self.beta_logits : np.ndarray
        self.learning_rate = learning_rate

        self.smoothing_window = smoothing_window
        self.past_priors = []
        self.initialize_uniformly()

    def initialize_uniformly(self):

        self.beta_logits = np.full((self.dim,), fill_value=1/self.dim, dtype=np.float32)

        self.past_priors = [self.beta_logits.copy()]

    def initialize_randomly(self):

        self.beta_logits = np.random.random((self.dim,)) * 4
        self.beta_logits /= self.beta_logits.sum(keepdims=True)
        self.past_priors = [self.beta_logits.copy()]


    def sample_test_set(self, num_scenarios=3):

        self.beta_logits[np.random.choice(self.dim, size=num_scenarios, replace=False)] = 1.
        self.beta_logits /= num_scenarios
        self.past_priors = [self.beta_logits.copy()]


    def initialize_certain(self, idx=0):

        #self.beta_logits = np.full((self.dim,), fill_value=-10, dtype=np.float32)
        #self.beta_logits[idx] = 10

        self.beta_logits = np.zeros((self.dim,), dtype=np.float32)
        self.beta_logits[idx] = 1
        self.past_priors = [self.beta_logits.copy()]

    def get_probs(self, smooth=False):

        if smooth:
            return sum(self.past_priors) / len(self.past_priors)
        else:
            return self.beta_logits
        #exp = np.exp((self.beta_logits - self.beta_logits.max())*0.5)
        #return exp / exp.sum()

    def __call__(self, smooth=False):
        return self.get_probs(smooth=smooth)

    def update_prior(self, loss, regret=True):

        # Score is the value of the policy
        # Thus, we want to minimize it, ie maximize regret

        #loss -= np.mean(loss)
        if not regret:
            loss = np.max(loss) - loss + np.min(loss)

        normalized_loss = loss

        #next_beta = np.maximum(self.beta_logits + normalized_loss * self.learning_rate, 0.)

        #print("prior loss:", loss * self.learning_rate)
        #print("prior:", self.beta_logits)


        #self.beta_logits[self.beta_logits < 0] = 1e-3

        #self.beta_logits[:] /= self.beta_logits.sum()

        self.beta_logits[:] = project_onto_simplex(self.beta_logits + normalized_loss * self.learning_rate)

        #self.beta_logits[:] = project_to_simplex(next_beta)

        #print("prior post projection:", self.beta_logits)

        self.past_priors.append(self.beta_logits.copy())
        if len(self.past_priors) > self.smoothing_window:
            self.past_priors.pop(0)


    def sample(self, size):
        return np.random.choice(self.dim, size, p=self())



if __name__ == "__main__":

    prior = Prior(5, learning_rate=1e-3)
    prior.initialize_uniformly()

    loss = np.array([
        4,-5,1,5,1
    ])

    for j in range(100):

        prior.update_prior(loss)

        print(prior.get_probs())

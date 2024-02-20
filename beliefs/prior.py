import numpy as np


def project_to_simplex(p):
    p_sorted = -np.sort(-p)
    cumulative_sum = np.cumsum(p_sorted)
    rho = np.argmax(p_sorted > (cumulative_sum - 1) / np.arange(1, len(p) + 1))
    theta = np.max([0, (cumulative_sum[rho] - 1) / (rho + 1)])
    x = np.maximum(p - theta, 0)
    return x / x.sum()


class Prior:

    def __init__(self, dim, learning_rate=5e-2):
        self.dim = dim
        self.beta_logits : np.ndarray
        self.learning_rate = learning_rate

        self.initialize_uniformly()

    def initialize_uniformly(self):

        self.beta_logits = np.full((self.dim,), fill_value=1/self.dim, dtype=np.float32)

    def initialize_randomly(self):

        self.beta_logits = np.random.random((self.dim,)) * 4
        self.beta_logits /= self.beta_logits.sum(keepdims=True)

    def sample_test_set(self, num_scenarios=3):

        self.beta_logits[np.random.choice(self.dim, size=num_scenarios, replace=False)] = 1.
        self.beta_logits /= num_scenarios


    def initialize_certain(self, idx=0):

        #self.beta_logits = np.full((self.dim,), fill_value=-10, dtype=np.float32)
        #self.beta_logits[idx] = 10

        self.beta_logits = np.zeros((self.dim,), dtype=np.float32)
        self.beta_logits[idx] = 1

    def get_probs(self):
        return self.beta_logits
        #exp = np.exp((self.beta_logits - self.beta_logits.max())*0.5)
        #return exp / exp.sum()

    def __call__(self):
        return self.get_probs()

    def update_prior(self, loss, regret=True):

        # Score is the value of the policy
        # Thus, we want to minimize it, ie maximize regret

        #loss -= np.mean(loss)
        if not regret:
            loss = np.max(loss) - loss + np.min(loss)

        normalized_loss = loss

        next_beta = np.maximum(self.beta_logits + normalized_loss * self.learning_rate, 0.)

        #print("prior loss:", loss * self.learning_rate)
        #print("prior:", self.beta_logits)


        #self.beta_logits[self.beta_logits < 0] = 1e-3

        #self.beta_logits[:] /= self.beta_logits.sum()

        self.beta_logits[:] = next_beta / next_beta.sum()

        #self.beta_logits[:] = project_to_simplex(next_beta)

        #print("prior post projection:", self.beta_logits)



if __name__ == "__main__":

    prior = Prior(5, learning_rate=1e-3)
    prior.initialize_uniformly()

    loss = np.array([
        4,-5,1,5,1
    ])

    for j in range(100):

        prior.update_prior(loss)

        print(prior.get_probs())

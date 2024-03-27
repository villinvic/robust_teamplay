import numpy as np


class SmoothMetric:
    def __init__(self, init_value=0., lr=0.993):
        assert  0 <= lr < 1
        self.lr =lr
        self._v = init_value

    def update(self, value, weight=1.):

        if not np.any(np.isnan(value)):
            lr = np.maximum(1. - (1. - self.lr) * weight, 0.)
            self._v = lr * self._v + (1. - lr) * value

        return self._v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


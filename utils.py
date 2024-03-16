import numpy as np


class SmoothMetric:
    def __init__(self, init_value=0, lr=0.993):

        self.lr =lr
        self._v = init_value

    def update(self, value):

        if not np.isnan(value):
         self._v = self.lr * self._v + (1-self.lr) * value

        return self._v

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


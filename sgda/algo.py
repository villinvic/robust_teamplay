

class SGDA:
    def __init__(self,
                 min_function,
                 max_function,
                 n_iter=1000
                 ):
        self.n_iter = n_iter

        self.min_function = min_function
        self.max_function = max_function

    def __call__(self, lr1, lr2):

        for i in range(self.n_iter):

            self.max_function()
            self.min_function()
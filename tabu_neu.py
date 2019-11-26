import numpy as np


class TabuSearch:

    def __init__(self, prob):

        # parameters
        self.n_iter = 10**6
        self.n_neigh = 100
        self.n_tabu_tenure = 10
        self.n_term = 10**4

        # input
        self.prob = prob

        # search arrays
        self.neigh = np.empty([self.n_neigh, self.prob.n], dtype=int)
        self.tabu_tenure = np.empty([self.n_tabu_tenure, self.prob.n], dtype=int)

        # output
        self.best = np.empty(self.prob.n, dtype=int)
        self.best_obj = None

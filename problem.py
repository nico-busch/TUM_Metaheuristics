import numpy as np
import pandas as pd
import math


# this class only creates instances (objects) of the problem
class Problem:

    def __init__(self,
                 n,
                 m):

        self.n = n
        self.m = m

        # index ranges
        self.i = range(1, n + 1)
        self.k = range(1, m + 1)

        # input dataframes
        self.a_i = pd.DataFrame()
        self.b_i = pd.DataFrame()
        self.d_i = pd.DataFrame()
        self.p_i = pd.DataFrame()
        self.w_kl = pd.DataFrame()
        self.f_ij = pd.DataFrame()

        # create initial data
        self.create_data()

    # creates data with regards to the number of flights n and number of gates m
    def create_data(self):
        # parameter
        des = 0.7

        # setting of random values
        a = np.random.uniform(1, self.n * 70 / self.m, self.n)
        b = a + np.random.uniform(45, 74)
        d = des * (b - a)
        p = np.random.uniform(10, 14, self.n)
        f = np.where(a.reshape(self.n, 1) < a, np.random.randint(6, 60, [self.n, self.n]), 0)

        # gate distances
        w = self.create_distance_matrix()

        # create dataframes
        self.a_i = pd.DataFrame(data={'a': a, 'i': self.i}).set_index('i')
        self.b_i = pd.DataFrame(data={'b': b, 'i': self.i}).set_index('i')
        self.d_i = pd.DataFrame(data={'d': d, 'i': self.i}).set_index('i')
        self.p_i = pd.DataFrame(data={'p': p, 'i': self.i}).set_index('i')
        self.w_kl = pd.DataFrame(data={'w': w.flatten()},
                                 index=pd.MultiIndex.from_product([self.k, self.k],
                                                                  names=['k', 'l']))
        self.f_ij = pd.DataFrame(data={'f': f.flatten()},
                                 index=pd.MultiIndex.from_product([self.i, self.i],
                                                                  names=['i', 'j']))

    # distance matrix
    def create_distance_matrix(self):
        w = np.empty([self.m, self.m])
        for k in range(self.m):
            kk = k + 1
            for l in range(self.m):
                ll = l + 1
                if (kk % 2 == 0) == (ll % 2 == 0):
                    w[k][l] = math.sqrt(((kk - ll) * 0.5) ** 2)
                elif kk % 2 == 0:
                    w[k][l] = 3 + math.sqrt(((kk - 2) * 0.5) ** 2) + math.sqrt(((ll - 1) * 0.5) ** 2)
                elif ll % 2 == 0:
                    w[k][l] = 3 + math.sqrt(((ll - 2) * 0.5) ** 2) + math.sqrt(((kk - 1) * 0.5) ** 2)
                else:
                    w[k][l] = math.sqrt(((kk - ll) * 0.5) ** 2)
        return w

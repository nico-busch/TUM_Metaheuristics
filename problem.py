import numpy as np
import pandas as pd
import math


# this class only creates instances (objects) of the problem
class Problem:

    def __init__(self,
                 n,
                 m):

        # index lengths
        self.n = n
        self.m = m

        # index ranges
        self.i = range(1, n + 1)
        self.k = range(1, m + 1)

        # numpy arrays
        self.a = np.empty(self.n)
        self.b = np.empty(self.n)
        self.d = np.empty(self.n)
        self.p = np.empty(self.n)
        self.f = np.empty([self.n, self.n])
        self.w = np.empty([self.m, self.m])

        # pandas dataframes
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

        # create arrays randomly
        self.a = np.random.uniform(1, self.n * 70 / self.m, self.n)
        self.b = self.a + np.random.uniform(45, 74, self.n)
        self.d = des * (self.b - self.a)
        self.p = np.random.uniform(10, 14, self.n)
        self.f = np.where(self.a[:, np.newaxis] < self.a, np.random.randint(6, 60, [self.n, self.n]), 0)
        self.w = self.create_distance_matrix()

        # create dataframes
        self.a_i = pd.DataFrame(data={'a': self.a, 'i': self.i}).set_index('i')
        self.b_i = pd.DataFrame(data={'b': self.b, 'i': self.i}).set_index('i')
        self.d_i = pd.DataFrame(data={'d': self.d, 'i': self.i}).set_index('i')
        self.p_i = pd.DataFrame(data={'p': self.p, 'i': self.i}).set_index('i')
        self.f_ij = pd.DataFrame(data={'f': self.f.flatten()},
                                 index=pd.MultiIndex.from_product([self.i, self.i],
                                                                  names=['i', 'j']))
        self.w_kl = pd.DataFrame(data={'w': self.w.flatten()},
                                 index=pd.MultiIndex.from_product([self.k, self.k],
                                                                  names=['k', 'l']))

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

import numpy as np
import pandas as pd
import itertools
import math

class Problem:

    def __init__(self,
                 n,
                 m,
                 a,
                 b,
                 d,
                 p,
                 w,
                 f):

        self.i = range(1, n + 1)
        self.k = range(1, m + 1)
        self.a_i = pd.DataFrame(data={'a': a, 'i': self.i}).set_index('i')
        self.b_i = pd.DataFrame(data={'b': b, 'i': self.i}).set_index('i')
        self.d_i = pd.DataFrame(data={'d': d, 'i': self.i}).set_index('i')
        self.p_i = pd.DataFrame(data={'p': p, 'i': self.i}).set_index('i')
        self.w_kl = pd.DataFrame(data={'w': w},
                                 index=pd.MultiIndex.from_product([self.k, self.k],
                                                                  names=['k', 'l']))
        self.f_ij = pd.DataFrame(data={'f': f},
                                 index=pd.MultiIndex.from_product([self.i, self.i],
                                                                  names=['i', 'j']))
        self.c_i = pd.DataFrame()
        self.x_ik = pd.DataFrame()
        self.sched = pd.DataFrame()

    def create_sched(self):
        self.sched = self.x_ik.join(
            pd.concat([self.c_i, self.a_i, self.b_i, self.d_i, self.p_i],
                      axis=1), how='inner').sort_values(by=['k', 'c'])

    def shift_left(self, k, i):
        for (idx1, idx2), row in self.sched.loc[pd.IndexSlice[i:, k], :].iterrows():
            self.sched.loc[(idx1, k), 'c'] = max(row['a'],
                                                 self.sched.shift(1).loc[(idx1, k), 'c']
                                                 + self.sched.shift(1).loc[(idx1, k), 'd'])
    def shift_right(self, k, i, t):
        return 'hello world'

    def attempt_shift_right(self):
        return 'hello world'

    def shift_interval(self):
        return 'hello world'

    def attempt_shift_interval(self):
        return 'hello world'

    def attempt_shift_interval_right(self):
        return 'hello world'

    def insert(self):
        return 'hello world'

    def tabu_search(self):
        return 'hello world'

    def genetic_algorithm(self):
        return 'hello world'

    def memetic_algorithm(self):
        return 'hello world'

    def solve(self):
        return 'hello world'



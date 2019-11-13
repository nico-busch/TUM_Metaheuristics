import numpy as np
import pandas as pd
import math


class Problem:

    def __init__(self,
                 n,
                 m):

        self.n = n
        self.m = m
        self.i = range(1, n + 1)
        self.k = range(1, m + 1)
        self.a_i = pd.DataFrame()
        self.b_i = pd.DataFrame()
        self.d_i = pd.DataFrame()
        self.p_i = pd.DataFrame()
        self.w_kl = pd.DataFrame()
        self.f_ij = pd.DataFrame()

        self.c_i = pd.DataFrame()
        self.x_ik = pd.DataFrame()
        self.y_ij = pd.DataFrame()
        self.z_ijkl = pd.DataFrame()

        feasible = False
        while not feasible:
            self.create_data()
            feasible = self.create_initial_solution()

    # creates data with regards to the number of flights n and number of gates m
    def create_data(self):
        a = np.empty([self.n])
        b = np.empty([self.n])
        d = np.empty([self.n])
        p = np.empty([self.n])
        f = np.empty([self.n, self.n])
        # parameter
        des = 0.7
        # setting of random values
        for i in range(self.n):
            a[i] = np.random.uniform(1, self.n * 70 / self.m)
            b[i] = a[i] + np.random.uniform(45, 74)
            d[i] = des * (b[i] - a[i])
            p[i] = np.random.uniform(10, 14)
            for j in range(self.n):
                if a[i] < a[j]:
                    f[i][j] = np.random.random_integers(6, 60)
                else:
                    f[i][j] = 0
        # gate distances
        w = self.create_distance_matrix()
        # print(self.a_i) # Todo delete if not necessary anymore

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
        # print(self.w_kl) Todo delete if not necessary anymore

    # creates a feasible initial solution
    def create_initial_solution(self):
        self.x_ik = pd.DataFrame(data={'x': 0},
                                 index=pd.MultiIndex.from_product([self.i, self.k],
                                                                  names=['i', 'k']))
        self.c_i = pd.DataFrame(data={'c': 0, 'i': self.i}).set_index('i')
        temp = self.a_i.sort_values(by=['a'])
        for idx, row in temp.iterrows():
            successful = False
            count = 0
            while not successful:
                k = round(np.random.uniform(1, self.m))
                temp = self.x_ik.loc[pd.IndexSlice[:, k], :].loc[self.x_ik['x'] == 1].index.get_level_values(0)
                if max(self.c_i.loc[temp, 'c']
                       + self.d_i.loc[temp, 'd'], default=0) <= self.a_i.loc[idx, 'a']:
                    self.c_i.loc[idx] = self.a_i.loc[idx, 'a']
                    self.x_ik.loc[(idx, k), :] = 1
                    successful = True
                elif max(self.c_i.loc[temp, 'c']
                         + self.d_i.loc[temp, 'd']
                         + self.d_i.loc[idx, 'd']) <= self.b_i.loc[idx, 'b']:
                    self.c_i.loc[idx] = max(self.c_i.loc[temp, 'c'] + self.d_i.loc[temp, 'd'])
                    self.x_ik.loc[(idx, k), :] = 1
                    successful = True
                else:
                    successful = False
                    count += 1
                    if count > 1000:
                        return False
        return True

    def shift_left(self, i, k):
        pos = self.x_ik.loc[pd.IndexSlice[:, k], :].loc[self.x_ik['x'] == 1].index.get_level_values(0)
        temp = self.c_i.loc[pos].sort_values(by=['c'])
        loc = temp.index.get_loc(i)
        x = 1
        prev = None
        for idx, row in temp.iterrows():
            if x >= loc:
                if x == 1:
                    self.c_i.loc[idx] = self.a_i.loc[idx, 'a']
                else:
                    self.c_i.loc[idx] = max(self.a_i.loc[idx, 'a'],
                                            self.c_i.loc[prev, 'c']
                                            + self.d_i.loc[prev, 'd'])
            prev = idx
            x += 1

    def shift_right(self, k, i, t):
        return 'hello world'

    def attempt_shift_right(self, k, i):
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

# Hilfsklasse für die Berechnung von Ergebnissewerten (objective Function)
# Generelle Trennung in Daten-Speicher-Klasse Problem (Input-Parameter) und Lösungsspeicher-Klasse Solution (Output decision variables)
# todo abstimmen mit Nico
class Solution:

    def __init__(self, x_ik, c_i):
        self.x_ik = x_ik
        self.c_i = c_i

    # todo abstimmen mit Nico
    def calculateObjectiveValue(self, Problem):
        sumDelayPenalty = 0
        for i in range(Problem.n):
            sumDelayPenalty += Problem.p_i[i]*(self.c_i[i] - Problem.a_i[i])
        sumWalkingDistance = 0
        for i in range(Problem.n):
            for j in range(Problem.n):
                for k in range(Problem.m):
                    for l in range(Problem.m):
                        if(self.x_ik[k][i] > -1 and self.x_ik[l][j] > -1):
                            sumWalkingDistance += Problem.w_kl[k][l]*Problem.f_ij[self.x_ik[k][i]][self.x_ik[l][j]]
        return sumWalkingDistance + sumDelayPenalty
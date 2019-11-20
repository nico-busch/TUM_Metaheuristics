import timeit

import numpy as np
import pandas as pd

import random


class GeneticAlgorithm:

    def __init__(self, prob, pop_n=None, crossover_n=None, iter_n=None, p1=None, term=None):

        self.prob = prob
        self.pop_n = pop_n or 3
        self.crossover_n = crossover_n or 10
        self.iter_n = iter_n or 100
        self.p1 = p1 or 0.2
        self.term = term or 10
        self.pop = {}
        self.obj = {}
        self.best = None

    def solve(self):

        self.create_initial_population()
        n = self.pop_n
        z = 0

        for x in range(self.iter_n):
            for y in range(self.crossover_n):

                par1 = random.choice(list(self.pop.values()))
                par2 = random.choice(list(self.pop.values()))

                off1, off2 = self.one_point_crossover(par1, par2)

                if np.random.rand() <= self.p1:
                    off1 = self.mutation(off1)
                if np.random.rand() <= self.p1:
                    off2 = self.mutation(off2)

                off1, obj1 = self.generate_solution(off1)
                if obj1 is not None:
                    self.pop[n] = off1
                    self.obj[n] = obj1
                    n += 1

                off2, obj2 = self.generate_solution(off2)
                if obj2 is not None:
                    self.pop[n] = off2
                    self.obj[n] = obj2
                    n += 1

            top = [key for key in sorted(self.obj, key=self.obj.get)[:self.pop_n]]
            self.pop = {key: self.pop[key] for key in top}
            print(self.pop)
            if x == 0:
                self.best = min(self.obj.values())
            elif min(self.obj.values()) < self.best:
                self.best = min(self.obj.values())
            elif z <= self.term:
                z += 1
            else:
                break

        return self.best

    def create_initial_population(self):

        for n in range(self.pop_n):

            indv = None
            obj = None

            print(n)

            while obj is None:
                indv = np.random.randint(1, self.prob.m + 1, self.prob.n)
                indv, obj = self.generate_solution(indv)

            self.pop[n] = indv
            self.obj[n] = obj

    def generate_solution(self, indv):

        sol = Solution(self.prob)

        for i, k in enumerate(indv, 1):

            g = k
            x = 0
            successful = False

            while x < self.prob.m:

                sched = sol.get_gate_schedule(g)

                if sched.empty:
                    sol.c_i.loc[i] = self.prob.a_i.loc[i, 'a']
                    sol.x_ik.loc[(i, g)] = 1
                    indv[i - 1] = g
                    successful = True
                    break

                space = self.prob.b_i.loc[i, 'b'] - self.prob.a_i.loc[i, 'a'] \
                        + (sched['c'].clip(lower=self.prob.a_i.loc[i, 'a'])
                           - (sched['c'] + sched['d']).clip(upper=self.prob.b_i.loc[i, 'b'])).clip(upper=0).sum()

                if space >= self.prob.d_i.loc[i, 'd']:

                    sol.c_i.loc[i] = (sched['c'] + sched['d']).clip(lower=self.prob.a_i.loc[i, 'a']) \
                        .where(sched['c'] < self.prob.a_i.loc[i, 'a'], self.prob.a_i.loc[i, 'a']).max()

                    sol.x_ik.loc[(i, g)] = 1
                    indv[i - 1] = g
                    successful = True
                    break

                g = g % self.prob.m + 1
                x += 1

            if not successful:
                return indv, None

        return indv, sol.calculate_objective_value()

    def one_point_crossover(self, par1, par2):

        cross = np.random.randint(0, self.prob.n)
        off1 = np.concatenate([par1[:cross], par2[cross:]])
        off2 = np.concatenate([par2[:cross], par1[cross:]])
        return off1, off2

    def two_point_crossover(self, par1, par2):

        cross1 = np.random.randint(0, self.prob.n)
        cross2 = np.random.randint(0, self.prob.n)
        off1 = np.concatenate([par1[:cross1], par2[cross1:cross2], par1[cross2:]])
        off2 = np.concatenate([par2[:cross1], par1[cross1:cross2], par2[cross2:]])
        return off1, off2

    def mutation(self, indv):
        switch1 = np.random.randint(0, self.prob.n)
        switch2 = np.random.randint(0, self.prob.n)
        mut = np.copy(indv)
        mut[switch1], mut[switch2] = indv[switch2], indv[switch1]
        return mut


class Solution:

    def __init__(self,
                 prob,
                 x_ik=None,
                 c_i=None):

        self.prob = prob
        self.x_ik = x_ik or pd.DataFrame(data={'x': 0},
                                         index=pd.MultiIndex.from_product([self.prob.i, self.prob.k],
                                                                          names=['i', 'k']))
        self.c_i = c_i or pd.DataFrame(data={'c': 0, 'i': self.prob.i}).set_index('i')
        self.obj_value = 0

    def calculate_objective_value(self):

        sum_delay_penalty = np.sum(self.prob.p_i.to_numpy()
                                   * (self.c_i.to_numpy()
                                      - self.prob.a_i.to_numpy()))

        sum_walking_distance = np.sum(self.x_ik.to_numpy().reshape(self.prob.n, 1, self.prob.m, 1)
                                      * self.x_ik.to_numpy().reshape(1, self.prob.n, 1, self.prob.m)
                                      * self.prob.f_ij.to_numpy().reshape(self.prob.n, self.prob.n, 1, 1)
                                      * self.prob.w_kl.to_numpy().reshape(1, 1, self.prob.m, self.prob.m))

        return sum_delay_penalty + sum_walking_distance

    def get_schedule(self):
        sched = self.x_ik.loc[self.x_ik['x'] == 1].join(
            pd.concat([self.prob.a_i, self.prob.b_i, self.prob.d_i, self.c_i], axis=1), how='inner').sort_values(
            by=['k', 'c']).drop(['x'], axis=1)
        return sched

    def get_gate_schedule(self, k):
        sched = self.x_ik.loc[pd.IndexSlice[:, k], :].loc[self.x_ik['x'] == 1].join(
            pd.concat([self.prob.a_i, self.prob.b_i, self.prob.d_i, self.c_i], axis=1), how='inner').sort_values(
            by=['k', 'c']).drop(['x'], axis=1).reset_index(level=1, drop=True)
        return sched

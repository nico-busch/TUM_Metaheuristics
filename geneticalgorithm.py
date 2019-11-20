import timeit

import numpy as np
import pandas as pd


class GeneticAlgorithm:

    def __init__(self, prob, pop_n=None, crossover_n=None, iter_n=None, p1=None):

        self.prob = prob
        self.pop_n = pop_n or 100
        self.crossover_n = crossover_n or 10
        self.iter_n = iter_n or 100
        self.p1 = p1 or .2
        self.s_i = np.empty([self.prob.n])
        self.pop = np.empty([self.pop_n, self.prob.n])

    def one_point_crossover(self, par1, par2):

        cross = np.random.randint(0, self.prob.n)
        off1 = np.concatenate([self.pop[par1][:cross], self.pop[par2][cross:]])
        off2 = np.concatenate([self.pop[par2][:cross], self.pop[par1][cross:]])
        return off1, off2

    def two_point_crossover(self, par1, par2):

        cross1 = np.random.randint(0, self.prob.n)
        cross2 = np.random.randint(0, self.prob.n)
        off1 = np.concatenate([self.pop[par1][:cross1], self.pop[par2][cross1:cross2], self.pop[par1][cross2:]])
        off2 = np.concatenate([self.pop[par2][:cross1], self.pop[par1][cross1:cross2], self.pop[par2][cross2:]])
        return off1, off2

    def mutation(self, indv):
        switch1 = np.random.randint(0, self.prob.n)
        switch2 = np.random.randint(0, self.prob.n)
        mut = np.copy(indv)
        mut[switch1], mut[switch2] = indv[switch2], indv[switch1]
        return mut

    def solve(self):

        self.create_initial_population()

        for x in range(self.iter_n):
            for y in range(self.crossover_n):

                par1 = np.random.randint(0, self.pop.shape[0])
                par2 = np.random.randint(0, self.pop.shape[0])

                off1, off2 = self.one_point_crossover(par1, par2)

                if np.random.rand() <= self.p1:
                    off1 = self.mutation(off1)
                if np.random.rand() <= self.p1:
                    off2 = self.mutation(off2)

                sol1 = self.generate_solution(off1)


    def create_initial_population(self):

        self.pop = np.random.randint(1, self.prob.m + 1, (self.pop_n, self.prob.n))

    def generate_solution(self, indv):

        sol = Solution(self.prob)

        for i, k in enumerate(indv, 1):

            g = k
            x = 0

            while x < self.prob.m:

                sched = sol.get_gate_schedule(g)

                if sched.empty:
                    sol.c_i.loc[i] = self.prob.a_i.loc[i, 'a']
                    sol.x_ik.loc[(i, g)] = 1
                    break

                space = self.prob.b_i.loc[i, 'b'] - self.prob.a_i.loc[i, 'a'] \
                        + (sched['c'].clip(lower=self.prob.a_i.loc[i, 'a'])
                           - (sched['c'] + sched['d']).clip(upper=self.prob.b_i.loc[i, 'b'])).clip(upper=0).sum()

                if space >= self.prob.d_i.loc[i, 'd']:

                    sol.c_i.loc[i] = (sched['c'] + sched['d']).clip(lower=self.prob.a_i.loc[i, 'a']) \
                        .where(sched['c'] < self.prob.a_i.loc[i, 'a'], self.prob.a_i.loc[i, 'a']).max()

                    sol.x_ik.loc[(i, g)] = 1
                    break

                g = g % self.prob.m + 1
                x += 1

        return sol


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





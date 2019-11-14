import numpy as np
import pandas as pd
import math


# this class can solve objects of class Problem
class Solve:

    def __init__(self, prob):

        self.prob = prob
        self.c_i = pd.DataFrame()
        self.x_ik = pd.DataFrame()
        self.y_ij = pd.DataFrame()
        self.z_ijkl = pd.DataFrame()

        self.create_initial_solution()

    # creates a feasible initial solution
    def create_initial_solution(self):
        # creates dataframes filled with zeros
        self.x_ik = pd.DataFrame(data={'x': 0},
                                 index=pd.MultiIndex.from_product([self.prob.i, self.prob.k],
                                                                  names=['i', 'k']))
        self.c_i = pd.DataFrame(data={'c': 0, 'i': self.prob.i}).set_index('i')

        temp = self.prob.a_i.sort_values(by=['a'])
        for idx, row in temp.iterrows():
            successful = False
            count = 0
            # if the random chosen gate cannot fit the flight, another random gate is chosen (max 1000 times)
            while not successful:
                k = round(np.random.uniform(1, self.prob.m))
                temp = self.x_ik.loc[pd.IndexSlice[:, k], :].loc[self.x_ik['x'] == 1].index.get_level_values(0)
                if max(self.c_i.loc[temp, 'c']
                       + self.prob.d_i.loc[temp, 'd'], default=0) <= self.prob.a_i.loc[idx, 'a']:
                    self.c_i.loc[idx] = self.prob.a_i.loc[idx, 'a']
                    self.x_ik.loc[(idx, k), :] = 1
                    successful = True
                elif max(self.c_i.loc[temp, 'c']
                         + self.prob.d_i.loc[temp, 'd']
                         + self.prob.d_i.loc[idx, 'd']) <= self.prob.b_i.loc[idx, 'b']:
                    self.c_i.loc[idx] = max(self.c_i.loc[temp, 'c'] + self.prob.d_i.loc[temp, 'd'])
                    self.x_ik.loc[(idx, k), :] = 1
                    successful = True
                else:
                    successful = False
                    count += 1
                    if count > 1000:
                        return False
        return True

    # returns the complete current schedule
    def get_schedule(self):
        sched = self.x_ik.loc[self.x_ik['x'] == 1].join(
            pd.concat([self.prob.a_i, self.prob.b_i, self.prob.d_i, self.c_i], axis=1), how='inner').sort_values(
            by=['k', 'c']).drop(['x'], axis=1)
        return sched

    # returns the schedule for one gate
    def get_gate_schedule(self, k):
        sched = self.x_ik.loc[pd.IndexSlice[:, k], :].loc[self.x_ik['x'] == 1].join(
            pd.concat([self.prob.a_i, self.prob.b_i, self.prob.d_i, self.c_i], axis=1), how='inner').sort_values(
            by=['k', 'c']).drop(['x'], axis=1).reset_index(level=1, drop=True)
        return sched

    # shift left subroutine
    def shift_left(self, i, k):
        sched = self.get_gate_schedule(k)
        loc = sched.index.get_loc(i)
        x = 1
        prev = None
        for idx, row in sched.iterrows():
            if x >= loc:
                if x == 1:
                    self.c_i.loc[idx] = row['a']
                else:
                    self.c_i.loc[idx] = max(row['a'],
                                            self.c_i.loc[prev, 'c']
                                            + sched.loc[prev, 'd'])
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

    def solve_optimal(self):
        return 'hello world'

    # todo to be fixed
    def calculate_objective_value(self, Problem):
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

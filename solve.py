import numpy as np
import pandas as pd
#from gurobipy import *
import math
from problem import Problem


# this class can solve objects of class Problem
class Solve:

    def __init__(self, prob):

        self.prob = prob
        self.c_i = pd.DataFrame()
        self.x_ik = pd.DataFrame()
        # self.y_ij = pd.DataFrame()
        # self.z_ijkl = pd.DataFrame()

        self.create_initial_solution()

    # creates a feasible initial solution
    # TODO bug fixen. Manchmal wird c als 0 gesetzt
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
    """ Bis jetzt erst ein Ansatz. Funktioniert noch nicht immer. (OLI)
    def create_initial_solution_alternative(self):
        # creates dataframes filled with zeros
        self.x_ik = pd.DataFrame(data={'x': 0},
                                 index=pd.MultiIndex.from_product([self.prob.i, self.prob.k],
                                                                  names=['i', 'k']))
        self.c_i = pd.DataFrame(data={'c': 0, 'i': self.prob.i}).set_index('i')

        temp = self.prob.a_i.sort_values(by=['a'])
        k = 0
        start = 0
        prevIdx = np.empty([self.prob.m])
        currIdx = np.empty([self.prob.m])
        for idx, row in temp.iterrows():

            print(idx)
            print(self.x_ik)
            start += 1
            k += 1

            if(k > self.prob.m):
                k = 1
                prevIdx = np.copy(currIdx)
                currIdx = np.empty([self.prob.m])
                print(prevIdx)
                print(currIdx)

            if(start <= self.prob.m):
                self.c_i.loc[idx,'c'] = self.prob.a_i.loc[idx,'a']
                self.x_ik.loc[(idx,k),'x'] = 1
                currIdx[k-1] = idx
            elif(self.prob.b_i.loc[idx,'b']-self.prob.d_i.loc[idx,'d']
                 >= self.c_i.loc[prevIdx[k-1],'c']+self.prob.d_i.loc[prevIdx[k-1],'d']):
                currIdx[k-1]=idx
                self.x_ik.loc[(idx,k),'x'] = 1
                if(temp.loc[idx,'a'] >= self.c_i.loc[prevIdx[k-1], 'c']+self.prob.d_i.loc[prevIdx[k-1],'d']):
                    self.c_i.loc[idx,'c']=temp.loc[idx,'a']
                else:
                    self.c_i.loc[idx, 'c'] = self.c_i.loc[prevIdx[k-1], 'c']+self.prob.d_i.loc[prevIdx[k-1],'d']
            else:
                if(start <= self.prob.n):
                    print("Feasibility Error: initial data will be newly created")
                    self.prob = Problem(self.prob.n,self.prob.m)
                    self.create_initial_solution_alternative()
        """

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
        prev = None
        for x, (idx, row) in enumerate(sched.iterrows()):
            if x >= loc:
                if x == 0:
                    self.c_i.loc[idx] = row['a']
                else:
                    self.c_i.loc[idx] = max(row['a'],
                                            self.c_i.loc[prev, 'c']
                                            + sched.loc[prev, 'd'])
            prev = idx

    # shift right subroutine
    def shift_right(self, i, k, t):
        sched = self.get_gate_schedule(k)
        loc = sched.index.get_loc(i)
        prev = None
        for x, (idx, row) in enumerate(sched.iterrows()):
            if x >= loc:
                if x == loc:
                    self.c_i.loc[idx] = t
                else:
                    self.c_i.loc[idx] = max(row['c'],
                                            self.c_i.loc[prev, 'c']
                                            + sched.loc[prev, 'd'])
            prev = idx

    def attempt_shift_right(self, i, k):
        sched = self.get_gate_schedule(k).iloc[::-1]
        loc = sched.index.get_loc(i)
        temp = self.c_i.copy()
        prev = None
        for x, (idx, row) in enumerate(sched.iterrows()):
            if x <= loc:
                if x == 0:
                    temp.loc[idx] = row['b'] - row['d']
                else:
                    temp.loc[idx] = min(row['b'] - row['d'],
                                        temp.loc[prev, 'c'] - row['d'])
            prev = idx
        return temp.loc[i, 'c']

    def shift_interval(self, i, j, k, t):
        sched = self.get_gate_schedule(k)
        loc1 = sched.index.get_loc(i)
        loc2 = sched.index.get_loc(j)
        prev = None
        for x, (idx, row) in enumerate(sched.iterrows()):
            if loc1 <= x <= loc2:
                if x == loc1:
                    self.c_i.loc[idx] = t
                else:
                    self.c_i.loc[idx] = max(row['c'],
                                            self.c_i.loc[prev, 'c']
                                            + sched.loc[prev, 'd'])
            prev = idx

    def attempt_shift_interval(self, i, j, k, t):
        sched = self.get_gate_schedule(k)
        loc1 = sched.index.get_loc(i)
        loc2 = sched.index.get_loc(j)
        temp = self.c_i.copy()
        prev = None
        for x, (idx, row) in enumerate(sched.iterrows()):
            if loc1 <= x <= loc2:
                if x == loc1:
                    temp.loc[idx] = t
                else:
                    temp.loc[idx] = max(row['c'],
                                        temp.loc[prev, 'c']
                                        + sched.loc[prev, 'd'])
            prev = idx
        return temp.loc[j, 'c'] + sched.loc[j, 'd']

    def attempt_shift_interval_right(self, i, j, k):
        sched = self.get_gate_schedule(k).iloc[::-1]
        loc1 = sched.index.get_loc(i)
        loc2 = sched.index.get_loc(j)
        temp = self.c_i.copy()
        prev = None
        for x, (idx, row) in enumerate(sched.iterrows()):
            if loc2 <= x <= loc1:
                if x == 0:
                    temp.loc[idx] = row['b'] - row['d']
                else:
                    temp.loc[idx] = min(row['b'] - row['d'],
                                        temp.loc[prev, 'c'] - row['d'])
            prev = idx
        return temp.loc[i, 'c']

    # todo
    def insert(self):
        return 'hello world'

    # todo Oli
    def tabu_search(self):
        return 'hello world'

    # todo Nico
    def genetic_algorithm(self):
        return 'hello world'

    # todo ?
    def memetic_algorithm(self):
        return 'hello world'

    # todo Dani
    def solve_optimal(self):

        model = Model()
        model.Params.M = self.prob.b_i

        #Creation of decision variables
        x = {}
        for i in range(self.prob.n):
            for k in range(self.prob.m):
                if i != k:
                    x[i, k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'x[{i},{k}]')

        y = {}
        for i, j in range(self.prob.n):
            if i != j:
                y[i, j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'y[{i},{j}]')

        z = {}
        for i, j in range(self.prob.n):
            for k, l in range(self.prob.m):
                z[i, j, k, l] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'z[{i},{j},{k},{l}]')

        c = {}
        for i in range(self.prob.n):
            c[i] = model.addVar(lb=0.0, ub=null, vtype=GRB.CONTINUOUS, name=f'c[{i}]')

        #objective
        objective = LinExpr()

        objective = \
                    quicksum(self.prob.f[i, j]*self.prob.w[k, l]*z[i, j, k, l] for i, j in range(self.prob.n) for k, l in range(self.prob.m)) \
                    + quicksum(self.prob.p[i]*(self.prob.c[i] - self.prob.a[i]) for i in range(self.prob.n))
        #minimize objective function
        model.setObjective(objective, GRB.MINIMIZE)

        #constraints

        #c1
        for i in range(self.prob.n):
            model.addConstr(quicksum(x[i, k] for k in range(self.prob.m)), GBR.EQUAL, 1)

        #c2
        for i, j in range(self.prob.n):
            for k, l in range(self.prob.m):
                model.addConstr(z[i, j, k, l] <= x[i, k])

        #c3
        for i, j in range(self.prob.n):
            for k, l in range(self.prob.m):
                model.addConstr(z[i, j, k, l] <= x[j, l])

        #c4
        for i, j in range(self.prob.n):
            for k, l in range(self.prob.m):
                model.addConstr(x[i,  k] + x[j, l] - 1 <= z[i, j, k, l])
        #c5
        for i in range(self.prob.n):
            model.addConstr(self.prob.c[i] >= self.prob.a[i])
        #c6
        for i in range(self.prob.n):
            model.addConstr(c[i] <= self.prob.b[i] - self.prob.d[i])

        #c7 - BIG M 1
        for i, j in range(self.prob.n):
            model.addConstr((self.prob.c[i] + self.prob.d[i]) + self.prob.c[j] - y[i, j]*model.M, GBR.GREATER, 0)

        #c8 - BIG M 2
        for i, j in range(self.prob.n):
            model.addConstr((self.prob.c[i] + self.prob.d[i]) - self.prob.c[j] - (1 - y[i, j])*model.M <= 0)

        #c9
        for k in range(self.prob.m):
            if i != j:
                model.addConstr(y[i, j] + y[j, i] >= z[i, j, k, l])

        model.update()
        model.optimize()

        return model.objVal

    def calculate_objective_value(self):
        sumDelayPenalty = 0
        df_temp = pd.DataFrame()
        df_temp = pd.concat([self.c_i,self.prob.a_i, self.prob.p_i], axis=1)
        for index, row in df_temp.iterrows():
            sumDelayPenalty += row['p']*(row['c']-row['a'])
        sumWalkingDistance = 0
        for idx1, row1 in self.x_ik.iterrows():
            for idx2, row2 in self.x_ik.iterrows():
                if(idx1[0] != idx2[0] and idx1[1] != idx2[1]):
                    w_tmp = self.prob.w_kl.loc[(idx1[1],idx2[1]), 'w']
                    f_tmp = self.prob.f_ij.loc[(idx1[0], idx2[0]),'f']
                    x_mul = self.x_ik.loc[idx1,'x']*self.x_ik.loc[idx2,'x']
                    sumWalkingDistance += w_tmp*f_tmp*x_mul
        self.objective_value = sumWalkingDistance + sumDelayPenalty
        return self.objective_value

import numpy as np
import pandas as pd


# this class can solve objects of class Problem
class TabuSearch:

    def __init__(self, prob):

        self.prob = prob
        self.c_i = pd.DataFrame()
        self.x_ik = pd.DataFrame()

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

    def calculate_objective_value(self):

        sum_delay_penalty = np.sum(self.prob.p_i.to_numpy()
                                   * (self.c_i.to_numpy()
                                      - self.prob.a_i.to_numpy()))

        sum_walking_distance = np.sum(self.x_ik.to_numpy().reshape(self.prob.n, 1, self.prob.m, 1)
                                      * self.x_ik.to_numpy().reshape(1, self.prob.n, 1, self.prob.m)
                                      * self.prob.f_ij.to_numpy().reshape(self.prob.n, self.prob.n, 1, 1)
                                      * self.prob.w_kl.to_numpy().reshape(1, 1, self.prob.m, self.prob.m))

        return sum_delay_penalty + sum_walking_distance

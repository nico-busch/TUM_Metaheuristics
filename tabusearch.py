import numpy as np
import pandas as pd
from problem import Problem
import random


# this class can solve objects of class Problem
class TabuSearch:

    def __init__(self, prob):

        self.prob = prob
        self.c_i = pd.DataFrame()
        self.x_ik = pd.DataFrame()

        self.create_initial_solution()

        self.A = pd.DataFrame(data={'k': 0, 'i': self.prob.i}).set_index('i')
        self.create_A()


    """
    def __init__(self, prob, c_i, x_ik):

        self.prob = prob
        self.c_i = pd.DataFrame()
        self.x_ik = pd.DataFrame()
        self.c_i = c_i
        self.x_ik = x_ik

        self.A = pd.DataFrame(data={'k': 0, 'i': self.prob.i}).set_index('i')
        self.create_A()
    """

    def tabu_search(self):

        # set parameter
        max_num_iter = 10**6
        neigh = 100
        neigh_range = range(1, neigh + 1)
        tabu_tenure = 10
        max_num_without_impr = 10**4

        # create array of neighbours
        N = []
        for i in range(neigh):
            N.append(Solution(self.x_ik, self.c_i, self.prob.i))

        # create array for objective fct calculation for N
        Obj_val = pd.DataFrame(data={'O': 0},
                         index=pd.MultiIndex.from_product([neigh_range],
                                                          names=['n']))

        # Save current solution
        # todo check if alternative solution to object creation
        curr_sol = Solution(self.x_ik, self.c_i, self.prob.i)
        curr_obj_val = curr_sol.calculate_objective_value(self.prob)
        self.sol = curr_sol
        self.sol_obj_val = curr_obj_val

        # create array for tabu moves
        Tabu = []
        for i in range(tabu_tenure):
            Tabu.append(TabuMove(curr_sol.A))

        # Generate set of neighborhood solutions
        count = 0
        count_iter = 0
        count_unchanged = 0

        while(count_iter < max_num_iter and count_unchanged < max_num_without_impr):

            # update forbidden solutions
            Tabu[count % tabu_tenure] = TabuMove(curr_sol.A)

            # generate 'neigh' number of neighbours
            # todo locale variable anpassen
            while (count < neigh):
                # choose method randomly (insert = true / interval exchange move = false)
                method = bool(random.getrandbits(1))
                if (method == True):
                    if (self.insert()):
                        N[count] = Solution(self.x_ik, self.c_i, self.prob.i)
                        count += 1
                else:
                    if (self.interval_exchange()):
                        N[count] = Solution(self.x_ik, self.c_i, self.prob.i)
                        count += 1
            count = 0

            # calculate all objective values of neighbours
            # todo vectorize
            for i in range(neigh):
                Obj_val.loc[i+1, 'O'] = N[i].calculate_objective_value(self.prob)

            # Choose best solution
            idx_best = Obj_val['O'].idxmin(axis=0)

            # Check if best solution is forbidden
            # todo vectorize
            forbidden = False
            for i in range(tabu_tenure):
                if(N[idx_best[0]-1].A.equals(Tabu[i].A)):
                    forbidden = True
                    break

            # Check if better than current solution or not forbidden
                if(Obj_val.loc[idx_best, 'O'] < curr_obj_val or forbidden==False):
                    curr_sol = Solution(N[idx_best[0]-1].x_ik, N[idx_best[0]-1].c_i, self.prob.i)
                    curr_obj_val = Obj_val.loc[idx_best, 'O']
                    if(curr_obj_val < self.sol_obj_val):
                        self.sol_obj_val = curr_obj_val
                        self.sol = curr_sol
                        count_unchanged = 0
                    else:
                        count_unchanged += 1
                else:
                    count_unchanged += 1

            count_iter += 1
            print(self.sol_obj_val)

        # save the final solution
        self.sol = curr_sol
        self.sol_obj_val = curr_obj_val
        print(self.sol_obj_val)


    # creates a feasible initial solution
    # TODO bug fixen. Manchmal wird c als 0 gesetzt
    # Todo Manchmal werden noch falsche Lösungen ausgegeben. z.B. nicht alle Flüge zu Gates zugewiesen
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

        # todo change to self to Solution method

        # Choose randomly a flight i and gate k
        i = random.randint(1, self.prob.n)
        k = random.randint(1, self.prob.m)
        while k == self.A.loc[i,'k']:
            k = random.randint(1, self.prob.m)

        # calculate the endtime of gate occupation 'end' for every flight
        tmp = self.get_gate_schedule(k)
        tmp['end'] = tmp['c']+tmp['d']

        # find the possible precessor of flight i at new gate k
        starttime_i = self.prob.a_i.loc[i,'a']
        precessor_i = tmp[tmp['end'] <= starttime_i]
        gap_start = precessor_i['end'].max()
        if(pd.isnull(gap_start)):
            gap_start = starttime_i

        # find the possible successor of flight i at new gate k
        gap_end = 0
        tmp = tmp[tmp['end'] > starttime_i]
        if(tmp.empty):
            gap_end = float('inf')
        else:
            successor_i = tmp['c'].idxmin(axis=0)
            gap_end = self.attempt_shift_right(successor_i, k)

        # check if flight i fits into the gap
        if(gap_end - gap_start >= self.prob.d_i.loc[i, 'd']):
            # update new x
            self.x_ik.loc[(i, self.A.loc[i, 'k']), 'x'] = 0
            self.x_ik.loc[(i, k), 'x'] = 1
            # update new c
            self.c_i.loc[i, 'c'] = gap_start
            if( gap_start < starttime_i):
                self.c_i.loc[i, 'c'] = starttime_i
            # update all successors
            if(gap_end != float('inf')):
                self.shift_right(successor_i, k, gap_start+self.prob.d_i.loc[i,'d'])
                self.shift_left(successor_i, k)
            # update A
            self.create_A()
            return True
        # return false if insert move could not be realized
        else:
            return False

    def interval_exchange(self):

        # todo check if there are more than two flights at that gate
        # todo check if schedules at k are unequal to 0

        # choose random gates that are not the same
        k_1 = random.randint(1, self.prob.m)
        k_2 = random.randint(1, self.prob.m)
        while k_1 == k_2:
            k_2 = random.randint(1, self.prob.m)
        empty_k_1 = False
        empty_k_2 = False

        # choose random flights of gate k_1
        sched_k_1 = self.get_gate_schedule(k_1)
        if(sched_k_1.empty):
            return False
        pos_i_1 = random.randint(1, len(sched_k_1))
        pos_j_1 = random.randint(pos_i_1, len(sched_k_1))-1
        j_1 = sched_k_1.index[pos_j_1]
        pos_i_1 -= 1
        i_1 = sched_k_1.index[pos_i_1]

        # choose random flights of gate k_2
        sched_k_2 = self.get_gate_schedule(k_2)
        if(sched_k_2.empty):
            return False
        pos_i_2 = random.randint(1, len(sched_k_2))
        j_2 = sched_k_2.index[random.randint(pos_i_2, len(sched_k_2))-1]
        pos_i_2 -= 1
        i_2 = sched_k_2.index[pos_i_2]

        # determine interval parameter for gate k_1
        sched_k_1['end'] = sched_k_1['d'] + sched_k_1['c']
        t_11 = 0
        if(pos_i_1 != 0):
            t_11 = sched_k_1['end'].iloc[pos_i_1-1]
        idx = pos_j_1+1
        t_12 = float('inf')
        if(j_1 != sched_k_1['c'].idxmax(axis=0)):
            for x, (jj, row) in enumerate(sched_k_1.iterrows()):
                if x == idx:
                    t_12 = self.attempt_shift_right(jj, k_1)
                    break
        t_13 = self.c_i.loc[i_1, 'c']
        t_14 = sched_k_1.loc[j_1, 'end']
        t_15 = self.attempt_shift_interval_right(i_1, j_1, k_1)
        t_16 = sched_k_1.loc[j_1, 'b']

        # determine interval parameter for gate k_2
        sched_k_2 = pd.DataFrame()
        sched_k_2 = self.get_gate_schedule(k_2)[['d', 'c']]
        sched_k_2['end'] = sched_k_2['d'] + sched_k_2['c']
        t_21 = 0
        idx = sched_k_2.index.get_loc(i_2)
        if(idx != 0):
            t_21 = sched_k_2['end'].iloc[idx-1]
        idx = sched_k_2.index.get_loc(j_2)+1
        t_22 = float('inf')
        if(j_2 != sched_k_2['c'].idxmax(axis=0)):
            for x, (jj, row) in enumerate(sched_k_2.iterrows()):
                if x == idx:
                    t_22 = self.attempt_shift_right(jj, k_2)
                    break
        t_23 = self.c_i.loc[i_2, 'c']
        t_24 = self.c_i.loc[j_2, 'c'] + self.prob.d_i.loc[j_2, 'd']
        t_25 = self.attempt_shift_interval_right(i_2, j_2, k_2)
        t_26 = self.prob.b_i.loc[j_2, 'b']

        # check if intervals are incompatible
        succ = False
        if(t_21 > t_15 or t_11 > t_25 or t_12 < t_24 or t_22 < t_14):
            succ = False
        else:

            # special case if selecting the first
            if (t_21 == 0):
                t_21 = self.prob.a_i.loc[i_1, 'a']
            if (t_11 == 0):
                t_11 = self.prob.a_i.loc[i_2, 'a']

            result_1 = self.attempt_shift_interval(i_1, j_1, k_1, t_21)
            result_2 = self.attempt_shift_interval(i_2, j_2, k_2, t_11)

            if(result_1 <= t_22 and result_2 <= t_12):

                self.shift_interval(i_1, j_1, k_1, t_21)
                self.shift_interval(i_2, j_2, k_2, t_11)

                # perform exchange move
                pos_i_1 = sched_k_1.index.get_loc(i_1)
                pos_j_1 = sched_k_1.index.get_loc(j_1)
                for x, (idx, row) in enumerate(sched_k_1.iterrows()):
                    if pos_i_1 <= x <= pos_j_1:
                        self.x_ik.loc[(idx, k_1), 'x'] = 0
                        self.x_ik.loc[(idx, k_2), 'x'] = 1
                pos_i_2 = sched_k_2.index.get_loc(i_2)
                pos_j_2 = sched_k_2.index.get_loc(j_2)
                for x, (idx, row) in enumerate(sched_k_2.iterrows()):
                    if pos_i_2 <= x <= pos_j_2:
                        self.x_ik.loc[(idx, k_1), 'x'] = 1
                        self.x_ik.loc[(idx, k_2), 'x'] = 0
                # update A
                self.create_A()
                succ = True
        return succ

    # todo OLI: make more efficient if possible
    def create_A(self):
        tmp = pd.DataFrame()
        tmp = self.x_ik[self.x_ik['x'] == 1]
        for idx, row in tmp.iterrows():
            self.A.loc[(idx[0], 'k')] = idx[1]
        return self.A

    def calculate_objective_value(self):

        sum_delay_penalty = np.sum(self.prob.p_i.to_numpy()
                                   * (self.c_i.to_numpy()
                                      - self.prob.a_i.to_numpy()))

        sum_walking_distance = np.sum(self.x_ik.to_numpy().reshape(self.prob.n, 1, self.prob.m, 1)
                                      * self.x_ik.to_numpy().reshape(1, self.prob.n, 1, self.prob.m)
                                      * self.prob.f_ij.to_numpy().reshape(self.prob.n, self.prob.n, 1, 1)
                                      * self.prob.w_kl.to_numpy().reshape(1, 1, self.prob.m, self.prob.m))

        return sum_delay_penalty + sum_walking_distance

class Solution:

    def __init__(self, x_ik, c_i, i):

        self.x_ik = x_ik.copy()
        self.c_i = c_i.copy()
        self.A = pd.DataFrame(data={'k': 0, 'i': i}).set_index('i')
        self.create_A()

    def calculate_objective_value(self, prob):

        sum_delay_penalty = np.sum(prob.p_i.to_numpy()
                                   * (self.c_i.to_numpy()
                                      - prob.a_i.to_numpy()))

        sum_walking_distance = np.sum(self.x_ik.to_numpy().reshape(prob.n, 1, prob.m, 1)
                                      * self.x_ik.to_numpy().reshape(1, prob.n, 1, prob.m)
                                      * prob.f_ij.to_numpy().reshape(prob.n, prob.n, 1, 1)
                                      * prob.w_kl.to_numpy().reshape(1, 1, prob.m, prob.m))

        return sum_delay_penalty + sum_walking_distance

    def create_A(self):
        tmp = pd.DataFrame()
        tmp = self.x_ik[self.x_ik['x'] == 1]
        for idx, row in tmp.iterrows():
            self.A.loc[(idx[0], 'k')] = idx[1]
        return self.A

class TabuMove:

    def __init__(self, A):
        self.A = A

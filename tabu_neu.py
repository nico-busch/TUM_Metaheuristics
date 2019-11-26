import numpy as np
import random


class TabuSearch:

    def __init__(self, prob):

        # parameters
        self.n_iter = 10**6
        self.n_neigh = 100
        self.n_tabu_tenure = 10
        self.n_term = 10**4

        # input
        self.prob = prob

        # search arrays
        self.neigh = np.empty([self.n_neigh, self.prob.n], dtype=int)
        self.neigh_obj_val = np.empty(self.n_neigh, dtype=float)
        self.neigh_c = np.empty([self.n_neigh, self.prob.n], dtype=float)
        self.tabu_tenure = np.empty([self.n_tabu_tenure, self.prob.n], dtype=int)

        # output
        self.best = np.empty(self.prob.n, dtype=int)
        self.best_obj = None
        self.best_c = np.empty(self.prob.n, dtype=float)

    def solve(self):

        # create initial solution
        # self.best creation

        # local best solution
        curr_sol = self.best
        curr_obj_val = self.best_obj
        curr_sol_c = self.best_c
        working_sol = self.best
        working_c = self.best_c

        # initialize tabu tenure
        for i in self.tabu_tenure:
            i = curr_sol

        # Generate set of neighborhood solutions
        count = 0
        count_iter = 0
        count_tabu = 0
        count_unchanged = 0

        while (count_iter < self.n_iter and count_unchanged < self.n_term):

            # update forbidden solutions
            self.tabu_tenure[count_tabu % self.n_tabu_tenure] = curr_sol

            # generate neighbours
            while (count < self.n_neigh):
                # choose method randomly (insert = true / interval exchange move = false)
                method = bool(random.getrandbits(1))
                if (method == True):
                    working_sol, working_c, succ = self.insert(curr_sol, curr_sol_c)
                    if (succ):
                        self.neigh[count, :] = working_sol
                        self.neigh_c[count, :] = working_c
                        count += 1
                else:
                    working_sol, working_c, succ = self.interval_exchange(curr_sol, curr_sol_c)
                    if (succ):
                        self.neigh[count, :] = working_sol
                        self.neigh_c[count, :] = working_c
                        count += 1
            count = 0

            # calculate all objective values of neighbours
            self.neigh_obj_val = apply_along_axis(self.create_initial_solution, 1, self.neigh)

            # Choose best solution
            idx_best = np.argmin(self.neigh_obj_val)

            # Check if best solution is forbidden
            forbidden = False
            for i in range(self.tabu_tenure):
                if (np.array_equal(self.tabu_tenure[i], self.neigh[idx_best, :])):
                    forbidden = True
                    break

            # Check if better than current solution or not forbidden
            if (self.neigh_obj_val[idx_best] < curr_obj_val or forbidden == False):
                curr_sol = self.neigh[idx_best, :]
                curr_obj_val = self.neigh_obj_val[idx_best]
                curr_sol_c = self.neigh_c[idx_best, :]
                count_tabu += 1
                if (curr_obj_val < self.best_obj):
                    self.best_obj = curr_obj_val
                    self.best = curr_sol
                    self.best_c = curr_sol_c
                    count_unchanged = 0
                else:
                    count_unchanged += 1
            else:
                count_unchanged += 1

            count_iter += 1
            print(self.sol_obj_val)

    def insert(self, curr_sol, curr_sol_c):

        # Choose randomly a flight i and gate k
        i = random.randint(0, self.prob.n - 1)
        k = random.randint(0, self.prob.m - 1)
        while k == curr_sol[i]:
            k = random.randint(1, self.prob.m)

        # todo special case if no flights are at the gate ! ! ! !

        # get gate specific duration end times
        curr_sol_end = curr_sol_c + self.prob.d
        gate_end_times = np.where(curr_sol == k, curr_sol, -1*curr_sol)*curr_sol_end
        idx, = np.where(curr_sol == k)
        sched = idx[np.argsort(curr_sol_c[idx])]

        # find the possible predecessor of flight i at new gate k
        gate_predecessors = np.where(gate_end_times < starttime_i, gate_end_times, -1*gate_end_times)
        gap_start = 0
        predecessor_idx = None
        if(np.sum(gate_precessors) >= 0):
            predecessor_idx = gate_predecessors.argmax()
            gap_start = curr_sol_end[predecessor_idx]


        # find the possible successor of flight i at new gate k
        gate_successors = np.where(gate_end_times > starttime_i, gate_end_times, -1*gate_end_times)
        gap_end = float('inf')
        successor_idx = None
        if(np.sum(gate_successors) >= 0):
            successor_idx = np.where(gate_successors > 0, gate_successors, float('inf')).argmin()
            gap_end = self.attempt_shift_right(np.argwhere(sched == successor_idx), k)

        # check if flight i fits into the gap
        if (gap_end - gap_start >= self.prob.d[i]):
            # update new x
            new_sol = np.copy(curr_sol)
            new_sol[i] = k
            # update new c
            new_sol_c = np.copy(curr_sol_c)
            if (gap_start < self.prob.a[i]):
                new_sol_c[i] = self.prob.a[i]
            else:
                new_sol_c[i] = gap_start
            # update all successors
            if (gap_end != float('inf')):
                self.shift_right(k, np.argwhere(sched == successor_idx), gap_start + self.prob.d[i])
                self.shift_left(np.argwhere(sched == successor_idx), k)
            return True
        # return false if insert move could not be realized
        else:
            return False


    # todo Nico löschen
    def shift_left(self, s, c, k, i):
        idx, = np.where(s == k)
        sched = idx[np.argsort(c[idx])]
        loc = np.where(sched == i)[0][0]
        for x, y in enumerate(sched[loc:], loc):
            c_new = self.prob.a[y] if x == 0 \
                else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def shift_right(self, s, c, k, i, t):
        idx, = np.where(s == k)
        sched = idx[np.argsort(c[idx])]
        loc, = np.where(sched == i)
        for x, y in enumerate(sched[loc[0]:], loc[0]):
            c_new = t if x == loc[0] else np.maximum(c[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c
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
        self.best_c = None
        self.best_obj = None

    def create_initial_solution(self):

        s = np.random.randint(0, self.prob.m, self.prob.n)
        c = np.zeros(self.prob.n)
        for i in self.prob.a.argsort():
            k = s[i]
            successful = False
            x = 0
            while x < self.prob.m:
                c_max = self.prob.a[i] if s[s == k].size == 0 \
                    else np.amax(np.maximum(c + self.prob.d, self.prob.a[i])[s == k])
                if self.prob.b[i] - c_max >= self.prob.d[i]:
                    c[i] = c_max
                    s[i] = k
                    successful = True
                    break
                else:
                    k = k % (self.prob.m - 1) + 1
                    x += 1
            if not successful:
                return s, None
        return s, c

    def solve(self):

        # create initial solution
        z = 0
        while self.best_c is None:
            if z > 10000:
                print("Greedy heuristic cannot find feasible solution")
                return None
            self.best, self.best_c = self.create_initial_solution()
            z += 1
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
        loc = np.where(sched == i)[0][0]
        for x, y in enumerate(sched[loc:], loc):
            c_new = t if x == loc else np.maximum(c[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def attempt_shift_right(self, s, c, k, i):
        idx, = np.where(s == k)
        sched = idx[np.argsort(c[idx])]
        loc = np.where(sched == i)[0][0]
        for x, y in enumerate(sched[loc:][::-1]):
            c_new = self.prob.b[y] - self.prob.d[y] if x == 0 \
                else np.minimum(self.prob.b[y], c[sched[-x + 1]]) - self.prob.d[y]
            c[y] = c_new
        return c[i]

    def shift_interval(self, s, c, k, i, j, t):
        idx, = np.where(s == k)
        sched = idx[np.argsort(c[idx])]
        loc_i = np.where(sched == i)[0][0]
        loc_j = np.where(sched == j)[0][0]
        for x, y in enumerate(sched[loc_i:loc_j], loc_i):
            c_new = t if x == loc_i else np.maximum(c[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def attempt_shift_interval(self, s, c, k, i, j, t):
        idx, = np.where(s == k)
        sched = idx[np.argsort(c[idx])]
        loc_i = np.where(sched == i)[0][0]
        loc_j = np.where(sched == j)[0][0]
        for x, y in enumerate(sched[loc_i:loc_j], loc_i):
            c_new = t if x == loc_i else np.maximum(c[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c[j] + self.prob.d[j]

    def attempt_shift_interval_right(self, s, c, k, i, j):
        idx, = np.where(s == k)
        sched = idx[np.argsort(c[idx])]
        loc_i = np.where(sched == i)[0][0]
        loc_j = np.where(sched == j)[0][0]
        for x, y in enumerate(sched[loc_i:loc_j][::-1], loc_j):
            c_new = self.prob.b[y] - self.prob.d[y] if x == 0 \
                else np.minimum(self.prob.b[y], c[sched[-x + 1]]) - self.prob.d[y]
            c[y] = c_new
        return c[i]

    def insert(self, s_old, c):

        s = s_old.copy()

        # Choose randomly a flight i and gate k
        i = np.random.randint(0, self.prob.n)
        k = np.random.choice(np.setdiff1d(range(self.prob.m), s_old[i]))

        idx, = np.where(s == k)
        sched = idx[np.argsort(c[idx])]
        loc = np.where(c[sched] + self.prob.d[sched] > self.prob.a[i])[0][0]

        successful = False

        for x, y in enumerate(sched[loc:], loc):
            t_1 = 0 if x == 0 else c[sched[x - 1]] + self.prob.d[sched[x - 1]]
            t_2 = float('inf') if x == sched.size - 1 else self.attempt_shift_right(s, c, k, y)
            if t_2 - t_1 >= self.prob.d[i]:
                c = self.shift_right(s, c, k, y, t_1 + self.prob.d[i])
                c[i] = t_1
                s[i] = k
                idx_old, = np.where(s_old == s_old[i])
                sched_old = idx_old[np.argsort(c[idx_old])]
                loc_old = np.where(sched_old == i)[0][0]
                c = self.shift_left(s, c, s_old[i], loc_old)
                successful = True
                break

        return s, c, successful

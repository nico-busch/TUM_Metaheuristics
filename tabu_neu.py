import numpy as np


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

        self.shift_left(self.best, self.best_c, 0, 1)

    #     # local best solution
    #     curr_sol = self.best
    #     curr_obj_val = self.best_obj
    #     working_sol = self.best
    #
    #     # initialize tabu tenure
    #     for i in self.tabu_tenure:
    #         i = curr_sol
    #
    #     # Generate set of neighborhood solutions
    #     count = 0
    #     count_iter = 0
    #     count_unchanged = 0
    #
    #     while (count_iter < self.n_iter and count_unchanged < self.n_term):
    #
    #         # update forbidden solutions
    #         self.tabu_tenure[count_iter % self.n_tabu_tenure] = curr_sol
    #
    #         # generate neighbours
    #         while (count < self.n_neigh):
    #             # choose method randomly (insert = true / interval exchange move = false)
    #             method = bool(random.getrandbits(1))
    #             if (method == True):
    #                 working_sol = self.insert()
    #                 if (self.insert()):
    #                     self.neigh[count, :] =
    #                     N[count] = Solution(self.x_ik, self.c_i, self.prob.i)
    #                     count += 1
    #             else:
    #                 if (self.interval_exchange()):
    #                     N[count] = Solution(self.x_ik, self.c_i, self.prob.i)
    #                     count += 1
    #         count = 0
    #
    #         # calculate all objective values of neighbours
    #         # todo vectorize
    #         for i in range(neigh):
    #             Obj_val.loc[i + 1, 'O'] = N[i].calculate_objective_value(self.prob)
    #
    #         # Choose best solution
    #         idx_best = Obj_val['O'].idxmin(axis=0)
    #
    #         # Check if best solution is forbidden
    #         # todo vectorize
    #         forbidden = False
    #         for i in range(tabu_tenure):
    #             if (N[idx_best[0] - 1].A.equals(Tabu[i].A)):
    #                 forbidden = True
    #                 break
    #
    #             # Check if better than current solution or not forbidden
    #             if (Obj_val.loc[idx_best, 'O'] < curr_obj_val or forbidden == False):
    #                 curr_sol = Solution(N[idx_best[0] - 1].x_ik, N[idx_best[0] - 1].c_i, self.prob.i)
    #                 curr_obj_val = Obj_val.loc[idx_best, 'O']
    #                 if (curr_obj_val < self.sol_obj_val):
    #                     self.sol_obj_val = curr_obj_val
    #                     self.sol = curr_sol
    #                     count_unchanged = 0
    #                 else:
    #                     count_unchanged += 1
    #             else:
    #                 count_unchanged += 1
    #
    #         count_iter += 1
    #         print(self.sol_obj_val)
    #
    #     # save the final solution
    #     self.sol = curr_sol
    #     self.sol_obj_val = curr_obj_val
    #     print(self.sol_obj_val)
    #
    # def shift_left(self, s, c, i, k):
    #     loc = sched.index.get_loc(i)
    #     prev = None
    #     for x, (idx, row) in enumerate(sched.iterrows()):
    #         if x >= loc:
    #             if x == 0:
    #                 self.c_i.loc[idx] = row['a']
    #             else:
    #                 self.c_i.loc[idx] = max(row['a'],
    #                                         self.c_i.loc[prev, 'c']
    #                                         + sched.loc[prev, 'd'])
    #         prev = idx

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





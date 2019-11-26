import numpy as np


class TabuSearch:

    def __init__(self,
                 prob,
                 n_iter=10**6,
                 n_neigh=100,
                 n_tabu_tenure=10,
                 n_term=10**4):

        # parameters
        self.n_iter = n_iter
        self.n_neigh = n_neigh
        self.n_tabu_tenure = n_tabu_tenure
        self.n_term = n_term

        # input
        self.prob = prob

        # search arrays
        self.neigh = np.empty([self.n_neigh, self.prob.n], dtype=int)
        self.neigh_c = np.empty([self.n_neigh, self.prob.n], dtype=float)
        self.neigh_obj = np.empty(self.n_neigh, dtype=float)
        self.tabu_tenure = -np.ones([self.n_tabu_tenure, self.prob.n], dtype=int)

        # output
        self.best = np.empty(self.prob.n, dtype=int)
        self.best_c = None
        self.best_obj = None

    def solve(self):

        # create initial solution
        count_feas = 0
        while self.best_c is None:
            if count_feas > 10000:
                print("Greedy heuristic cannot find feasible solution")
                return None
            self.best, self.best_c = self.create_initial_solution()
            count_feas += 1
        self.best_obj = self.calculate_objective_value(self.best, self.best_c)

        # local best solution
        curr = self.best.copy()
        curr_c = self.best_c.copy()
        curr_obj = self.best_obj.copy()

        # Generate set of neighborhood solutions
        count_term = 0
        count_tabu = 0
        for count_iter in range(self.n_iter):
            # terminal condition
            if count_term >= self.n_term:
                break
            count_neigh = 0
            while count_neigh <= self.n_neigh:
                move = np.random.choice([True, False])
                if move:
                    s, c, successful = self.insert(curr, curr_c)
                    if successful:
                        self.neigh[count_neigh] = s
                        self.neigh_c[count_neigh] = c
                        self.neigh_obj[count_neigh] = self.calculate_objective_value(s, c)
                        count_neigh += 1
                else:
                    s, c, successful = self.interval_exchange(curr, curr_c)
                    if successful:
                        self.neigh[count_neigh] = s
                        self.neigh_c[count_neigh] = c
                        self.neigh_obj[count_neigh] = self.calculate_objective_value(s, c)
                        count_neigh += 1

            top = np.argmin(self.neigh_obj)

            if not (self.tabu_tenure == self.neigh[top]).any or self.neigh_obj[top] < self.best_obj:
                curr = self.neigh.copy()[top]
                curr_c = self.neigh_c.copy()[top]
                curr_obj = self.neigh_obj.copy()[top]
                self.tabu_tenure[count_tabu % self.n_tabu_tenure] = curr
                count_tabu += 1
                if self.neigh_obj[top] < self.best_obj:
                    self.best = curr.copy()
                    self.best_c = curr_c.copy()
                    self.best_obj = curr_obj.copy()
                    count_term = 0
            else:
                count_term +=1

        return self.best

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

    def shift_left(self, s, c, k, i):
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        loc = np.nonzero(sched == i)[0][0]
        for x, y in enumerate(sched[loc:], loc):
            c_new = self.prob.a[y] if x == 0 \
                else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def shift_right(self, s, c, k, i, t):
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        loc = np.nonzero(sched == i)[0][0]
        for x, y in enumerate(sched[loc:], loc):
            c_new = t if x == loc else np.maximum(c[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def attempt_shift_right(self, s, c, k, i):
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        loc = np.nonzero(sched == i)[0][0]
        for x, y in enumerate(sched[loc:][::-1]):
            c_new = self.prob.b[y] - self.prob.d[y] if x == 0 \
                else np.minimum(self.prob.b[y], c[sched[-x + 1]]) - self.prob.d[y]
            c[y] = c_new
        return c[i]

    def shift_interval(self, s, c, k, i, j, t):
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        loc_i = np.nonzero(sched == i)[0][0]
        loc_j = np.nonzero(sched == j)[0][0]
        for x, y in enumerate(sched[loc_i:loc_j], loc_i):
            c_new = t if x == loc_i else np.maximum(c[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def attempt_shift_interval(self, s, c, k, i, j, t):
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        loc_i = np.nonzero(sched == i)[0][0]
        loc_j = np.nonzero(sched == j)[0][0]
        for x, y in enumerate(sched[loc_i:loc_j], loc_i):
            c_new = t if x == loc_i else np.maximum(c[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c[j] + self.prob.d[j]

    def attempt_shift_interval_right(self, s, c, k, i, j):
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        loc_i = np.nonzero(sched == i)[0][0]
        loc_j = np.nonzero(sched == j)[0][0]
        for x, y in enumerate(sched[loc_i:loc_j][::-1], loc_j):
            c_new = self.prob.b[y] - self.prob.d[y] if x == 0 \
                else np.minimum(self.prob.b[y], c[sched[-x + 1]]) - self.prob.d[y]
            c[y] = c_new
        return c[i]

    def insert(self, s, c):

        s_new = s.copy()
        c_new = c.copy()

        # Choose randomly a flight i and gate k
        i = np.random.randint(0, self.prob.n)
        k = s[i]
        k_new = np.random.choice(np.setdiff1d(np.arange(self.prob.m), k))

        idx, = np.nonzero(s_new == k)
        sched = idx[np.argsort(c_new[idx])]
        loc = np.nonzero(sched == i)[0][0]

        idx_new, = np.nonzero(s_new == k_new)
        sched_new = idx_new[np.argsort(c_new[idx_new])]
        loc_new = np.nonzero(c_new[sched_new] + self.prob.d[sched_new] > self.prob.a[i])[0][0]

        successful = False
        for x, y in enumerate(sched_new[loc_new], loc_new):
            t_1 = 0 if x == 0 else c_new[sched_new[x - 1]] + self.prob.d[sched_new[x - 1]]
            t_2 = float('inf') if x == sched_new.size - 1 else self.attempt_shift_right(s_new, c_new, k_new, y)
            if t_2 - t_1 >= self.prob.d[i]:
                c_new = self.shift_right(s_new, c_new, k_new, y, t_1 + self.prob.d[i])
                c_new[i] = t_1
                s_new[i] = k_new
                c_new = self.shift_left(s_new, c_new, k, loc)
                successful = True
                break

        return s_new, c_new, successful

    def interval_exchange(self, s, c):

        s_new = s.copy()
        c_new = c.copy()

        k_1, k_2 = np.random.choice(self.prob.m, 2, replace=False)

        idx_1, = np.nonzero(s_new == k_1)
        sched_1 = idx_1[np.argsort(c_new[idx_1])]
        loc_i_1 = np.random.randint(0, sched_1.size)
        i_1 = sched_1[loc_i_1]
        loc_j_1 = np.random.randint(loc_i_1, sched_1.size)
        j_1 = sched_1[loc_j_1]

        idx_2, = np.nonzero(s_new == k_2)
        sched_2 = idx_2[np.argsort(c_new[idx_2])]
        loc_i_2 = np.random.randint(0, sched_2.size)
        i_2 = sched_2[loc_i_2]
        loc_j_2 = np.random.randint(loc_i_2, sched_2.size)
        j_2 = sched_2[loc_j_2]

        t_11 = 0 if loc_i_1 == 0 else c_new[sched_1[loc_i_1 - 1]] + self.prob.d[sched_1[loc_i_1 - 1]]
        t_12 = float('inf') if loc_j_1 == sched_1.size - 1 else self.attempt_shift_right(s, c, k_1, loc_j_1 + 1)
        t_13 = c_new[i_1]
        t_14 = c_new[j_1] + self.prob.d[j_1]
        t_15 = self.attempt_shift_interval_right(s_new, c_new, k_1, loc_i_1, loc_j_1)
        t_16 = self.prob.b[j_1]

        t_21 = 0 if loc_i_2 == 0 else c_new[sched_2[loc_i_2 - 1]] + self.prob.d[sched_2[loc_i_2 - 1]]
        t_22 = float('inf') if loc_j_2 == sched_2.size - 1 else self.attempt_shift_right(s_new, c_new, k_2, loc_j_2 + 1)
        t_23 = c_new[i_2]
        t_24 = c_new[j_2] + self.prob.d[j_2]
        t_25 = self.attempt_shift_interval_right(s_new, c_new, k_2, loc_i_2, loc_j_2)
        t_26 = self.prob.b[j_2]

        if t_21 > t_15 or t_11 > t_25 or t_12 < t_24 or t_22 < t_14:

            result_1 = self.attempt_shift_interval(s_new, c_new, k_1, loc_i_1, loc_j_1, t_21)
            result_2 = self.attempt_shift_interval(s_new, c_new, k_2, loc_i_2, loc_j_2, t_11)

            if result_1 <= t_22 and result_2 <= t_12:

                c_new = self.shift_interval(s_new, c_new, k_1, loc_i_1, loc_j_1, t_21)
                c_new = self.shift_interval(s_new, c_new, k_2, loc_i_2, loc_j_2, t_11)
                s_new[sched_1[loc_i_1:loc_j_1]] = k_2
                s_new[sched_2[loc_i_2:loc_j_2]] = k_1

                return s_new, c_new, True

        else:
            return s, c, False

    def calculate_objective_value(self, s, c):
        sum_delay_penalty = np.sum(self.prob.p * (c - self.prob.a))
        x = np.zeros([self.prob.n, self.prob.m])
        x[np.arange(self.prob.n), s] = 1
        sum_walking_distance = np.sum(x[:, np.newaxis, :, np.newaxis]
                                      * x[:, np.newaxis, :]
                                      * self.prob.f[:, :, np.newaxis, np.newaxis]
                                      * self.prob.w)
        return sum_delay_penalty + sum_walking_distance

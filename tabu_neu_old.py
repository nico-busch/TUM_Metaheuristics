import numpy as np


class TabuSearch:

    def __init__(self, prob, n_iter=10**6, n_neigh=100, n_tabu_tenure=10, n_term=10**4):

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

        print("None: " + str(self.best_c))

        # create initial solution
        count_feas = 0
        while self.best_c is None:
            if count_feas > 10000:
                print("Greedy heuristic cannot find feasible solution")
                return None
            self.best, self.best_c = self.create_initial_solution()
            count_feas += 1
        self.best_obj = self.calculate_objective_value(self.best, self.best_c)

        print("initial solution: " + str(self.best))

        # local best solution
        curr = np.copy(self.best)
        curr_c = np.copy(self.best_c)
        curr_obj = np.copy(self.best_obj)

        # Generate set of neighborhood solutions
        count_term = 0
        count_tabu = 0
        for count_iter in range(self.n_iter):
            # terminal condition
            if count_term >= self.n_term:
                break
            count_neigh = 0
            while count_neigh < self.n_neigh:
                move = np.random.choice([True, False])
                if move:
                    s, c, successful = self.insert(curr, curr_c)
                    if successful:
                        self.neigh[count_neigh] = s
                        self.neigh_c[count_neigh] = c
                        self.neigh_obj[count_neigh] = self.calculate_objective_value(s, c)

                        #print("insert obj: " + str(self.neigh_obj[count_neigh]))
                        #print("insert c: " + str(self.neigh_c[count_neigh]))
                        #print("insert a: " + str(self.prob.a))

                        count_neigh += 1


                else:
                    s, c, successful = self.interval_exchange(curr, curr_c)
                    if successful:
                        self.neigh[count_neigh] = s
                        self.neigh_c[count_neigh] = c
                        self.neigh_obj[count_neigh] = self.calculate_objective_value(s, c)

                        #print("interval obj: " + str(self.neigh_obj[count_neigh]))
                        #print("interval c: " + str(self.neigh_c[count_neigh]))
                        #print("interval a: " + str(self.prob.a))

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

                    print("best solution: " + str(self.best_obj))

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

    # CONTROLLED
    def shift_left(self, s_old, c_old, k, i):
        c = np.copy(c_old)
        s = np.copy(s_old)

        #print("shift left IN: " + str(c))

        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched[i:], i):
            c_new = self.prob.a[y] if x == 0 \
                else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new

        #print("shift left OUT: " + str(c))

        return c

    # CONTROLLED
    def shift_right(self, s_old, c_old, k, i, t):
        c = np.copy(c_old)
        s = np.copy(s_old)

        #print("shift right IN: " + str(c))
        #print("TTTT: " + str(t))

        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched[i:], i):
            c_new = np.maximum(t, self.prob.a[sched[i]]) if x == i else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new

        #print("shift right OUT: " + str(c))

        return c

    # CONTROLLED
    def attempt_shift_right(self, s_old, c_old, k, i):
        c = np.copy(c_old)
        s = np.copy(s_old)

        #print("attemp shift right IN: " + str(c))


        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched, 0):
            z = sched.size - 1 - (x % sched.size)
            if z >= i:
                c_new = self.prob.b[sched[z]] - self.prob.d[y] if z == sched.size-1 \
                    else np.minimum(self.prob.b[sched[z]], c[sched[z + 1]]) - self.prob.d[sched[z]]
                c[sched[z]] = c_new

        #print("attempt shift right OUT: " + str(c))

        return c[sched[i]]

    # CONTROLLED
    def attempt_shift_interval_right(self, s_old, c_old, k, i, j):
        s = np.copy(s_old)
        c = np.copy(c_old)

        #print("attemp shift interval right IN: " + str(c))

        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched, 0):
            z = sched.size - 1 - (x % sched.size)
            if z >= i and z <= j:
                c_new = self.prob.b[y] - self.prob.d[y] if z == sched.size - 1 \
                    else np.minimum(self.prob.b[sched[z]], c[sched[z+1]]) - self.prob.d[sched[z]]
                c[sched[z]] = c_new

        #print("attemp shift interval right OUT: " + str(c))

        return c[sched[i]]

    # CONTROLLED
    def shift_interval(self, s_old, c_old, k, i, j, t):
        c = np.copy(c_old)
        s = np.copy(s_old)

        #print("shift interval IN: " + str(c))

        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched[i:(j+1)], i):
            c_new = np.maximum(t, self.prob.a[sched[x]]) if x == i else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new

        #print("shift interval OUT: " + str(c))

        return c

    # CONTROLLED
    def attempt_shift_interval(self, s_old, c_old, k, i, j, t):
        c = np.copy(c_old)
        s = np.copy(s_old)

        #print("attemp shift interval IN: " + str(c))

        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched[i:(j+1)], i):
            c_new = np.maximum(t, self.prob.a[sched[x]]) if x == i else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new

        #print("attemp shift interval OUT: " + str(c))

        return c[sched[j]] + self.prob.d[sched[j]]

    # CONTROLLED
    def insert(self, s, c):

        s_new = np.copy(s)
        c_new = np.copy(c)

        # Choose randomly a flight i and gate k
        i = np.random.randint(0, self.prob.n)
        k = s[i]
        k_new = np.random.choice(np.setdiff1d(np.arange(self.prob.m), k))

        idx, = np.nonzero(s_new == k)
        sched = idx[np.argsort(c_new[idx])]

        idx_new, = np.nonzero(s_new == k_new)
        sched_new = idx_new[np.argsort(c_new[idx_new])]
        if(sched_new.size == 0):
            c_new[i] = self.prob.a[i]
            s_new[i] = k_new
            return s_new, c_new, True

        successful = False
        for x, y in enumerate(sched_new):
            if(c_new[x] + self.prob.d[x] > self.prob.a[i]):
                t_1 = self.prob.a[i] if x == 0 else np.maximum(c_new[sched_new[x - 1]] + self.prob.d[sched_new[x - 1]], self.prob.a[y])
                t_2 = np.inf if x == sched_new.size - 1 else self.attempt_shift_right(s_new, c_new, k_new, x)
                if t_2 - t_1 >= self.prob.d[i]:
                    c_new = self.shift_right(s_new, c_new, k_new, x, t_1 + self.prob.d[i])
                    c_new[i] = t_1
                    s_new[i] = k_new
                    c_new = self.shift_left(s_new, c_new, k, 0)
                    successful = True
                    break

        return s_new, c_new, successful

    def interval_exchange(self, s, c):

        s_new = np.copy(s)
        c_new = np.copy(c)

        k_1, k_2 = np.random.choice(self.prob.m, 2, replace=False)

        idx_1, = np.nonzero(s_new == k_1)
        sched_1 = idx_1[np.argsort(c_new[idx_1])]
        if(sched_1.size == 0):
            return s, c, False
        loc_i_1 = np.random.randint(0, sched_1.size)
        i_1 = sched_1[loc_i_1]
        loc_j_1 = np.random.randint(loc_i_1, sched_1.size)
        j_1 = sched_1[loc_j_1]

        t_11 = self.prob.a[i_1] if loc_i_1 == 0 else c_new[sched_1[loc_i_1 - 1]] + self.prob.d[sched_1[loc_i_1 - 1]]
        t_12 = np.inf if loc_j_1 == sched_1.size - 1 else self.attempt_shift_right(s, c, k_1, loc_j_1 + 1)
        t_13 = c_new[i_1]
        t_14 = c_new[j_1] + self.prob.d[j_1]
        t_15 = self.attempt_shift_interval_right(s_new, c_new, k_1, loc_i_1, loc_j_1)
        t_16 = self.prob.b[j_1]

        idx_2, = np.nonzero(s_new == k_2)
        sched_2 = idx_2[np.argsort(c_new[idx_2])]
        if(sched_2.size == 0):
            return s, c, False
        loc_i_2 = np.random.randint(0, sched_2.size)
        i_2 = sched_2[loc_i_2]
        loc_j_2 = np.random.randint(loc_i_2, sched_2.size)
        j_2 = sched_2[loc_j_2]

        t_21 = self.prob.a[i_2] if loc_i_2 == 0 else c_new[sched_2[loc_i_2 - 1]] + self.prob.d[sched_2[loc_i_2 - 1]]
        t_22 = np.inf if loc_j_2 == sched_2.size - 1 else self.attempt_shift_right(s_new, c_new, k_2, loc_j_2 + 1)
        t_23 = c_new[i_2]
        t_24 = c_new[j_2] + self.prob.d[j_2]
        t_25 = self.attempt_shift_interval_right(s_new, c_new, k_2, loc_i_2, loc_j_2)
        t_26 = self.prob.b[j_2]

        if t_21 <= t_15 and t_11 <= t_25 and t_12 >= t_24 and t_22 >= t_14:

            result_1 = self.attempt_shift_interval(s_new, c_new, k_1, loc_i_1, loc_j_1, t_21)
            result_2 = self.attempt_shift_interval(s_new, c_new, k_2, loc_i_2, loc_j_2, t_11)

            if result_1 <= t_22 and result_2 <= t_12:

                c_new = self.shift_interval(s_new, c_new, k_1, loc_i_1, loc_j_1, t_21)
                c_new = self.shift_interval(s_new, c_new, k_2, loc_i_2, loc_j_2, t_11)
                s_new[sched_1[loc_i_1:(loc_j_1+1)]] = k_2
                s_new[sched_2[loc_i_2:(loc_j_2+1)]] = k_1

                return s_new, c_new, True

        return s, c, False

    def calculate_objective_value(self, s, c):

        #print("s: " + str(s))
        #print("c: " + str(c))
        #print("a: " + str(self.prob.a))

        sum_delay_penalty = np.sum(self.prob.p * (c - self.prob.a))
        x = np.zeros([self.prob.n, self.prob.m])
        x[np.arange(self.prob.n), s] = 1
        sum_walking_distance = np.sum(x[:, np.newaxis, :, np.newaxis]
                                      * x[:, np.newaxis, :]
                                      * self.prob.f[:, :, np.newaxis, np.newaxis]
                                      * self.prob.w)

        if(sum_delay_penalty < 0):
            print("ERROR: c > a")
            exit()
        #print("sum_delay: " + str(sum_delay_penalty))
        #print("sum_waling: " + str(sum_walking_distance))
        #print("obj: " + str(sum_delay_penalty + sum_walking_distance))

        return sum_delay_penalty + sum_walking_distance


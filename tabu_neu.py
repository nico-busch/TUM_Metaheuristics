import numpy as np
import gantt


class TabuSearch:

    def __init__(self, prob, n_iter=10**5, n_neigh=10, n_tabu_tenure=10, n_term=10**4):

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

    #_________________________________________________________________________________________________
    # Main functions

    def solve(self):

        # create initial solution
        count_infeasible = 0
        while self.best_c is None:
            if count_infeasible >= 10000:
                print("Greedy heuristic cannot find feasible solution")
                return None
            self.best, self.best_c = self.create_solution()
            count_infeasible += 1
        self.best_obj = self.calculate_objective_value(self.best[np.newaxis, :], self.best_c[np.newaxis, :])

        # local best solution
        curr = self.best.copy()
        curr_c = self.best_c.copy()
        curr_obj = self.best_obj.copy()

        # Generate set of neighborhood solutions
        count_term = 0
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
                        count_neigh += 1
                else:
                    s, c, successful = self.interval_exchange(curr, curr_c)
                    if successful:
                        self.neigh[count_neigh] = s
                        self.neigh_c[count_neigh] = c
                        count_neigh += 1
            self.neigh_obj = self.calculate_objective_value(self.neigh, self.neigh_c)

            # Condition for current solution update
            top = np.argmin(self.neigh_obj[np.logical_or((self.neigh[:, np.newaxis] != self.tabu_tenure).any(-1).all(-1),
                                                         self.neigh_obj < self.best_obj)])
            curr = self.neigh.copy()[top]
            curr_c = self.neigh_c.copy()[top]
            curr_obj = self.neigh_obj.copy()[top]
            if curr_obj < self.best_obj:
                self.best = curr.copy()
                self.best_c = curr_c.copy()
                self.best_obj = curr_obj.copy()
                count_term = 0
            else:
                count_term += 1

            # Update memory
            self.tabu_tenure = np.roll(self.tabu_tenure, 1, axis=0)
            self.tabu_tenure[0] = curr

            print(str(count_iter) + " TS global best solution: " + str(self.best_obj))

        return self.best

    def create_solution(self):
        s = -np.ones(self.prob.n, dtype=np.int64)
        c = np.zeros(self.prob.n)
        for i in self.prob.a.argsort():
            successful = False
            for k in np.roll(np.arange(self.prob.m), -np.random.randint(0, self.prob.m)):
                c_max = self.prob.a[i] if s[s == k].size == 0 \
                    else np.amax(np.maximum(c + self.prob.d, self.prob.a[i])[s == k])
                if self.prob.b[i] - c_max >= self.prob.d[i]:
                    c[i] = c_max
                    s[i] = k
                    successful = True
                    break
            if not successful:
                return None, None
        return s, c

    def calculate_objective_value(self, s, c):
        sum_delay_penalty = np.einsum('i,ni->n', self.prob.p, (c - self.prob.a))

        x = np.zeros([s.size, self.prob.m], dtype=np.int64)
        x[np.arange(s.size), s.ravel()] = 1
        x.shape = s.shape + (self.prob.m,)
        sum_walking_distance = np.einsum('nik,njl,ij,kl->n', x, x, self.prob.f, self.prob.w, optimize=True)

        return sum_delay_penalty + sum_walking_distance

    #_________________________________________________________________________________________________
    # Routines: insert & interval exchange

    # todo possibly optimizable
    def insert(self, s, c):
        # Choose randomly a flight i and gate k
        i = np.random.randint(0, self.prob.n)
        k = s[i]
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        loc = np.nonzero(sched == i)[0][0]

        k_new = np.random.choice(np.setdiff1d(np.arange(self.prob.m), k))
        idx_new, = np.nonzero(s == k_new)
        sched_new = idx_new[np.argsort(c[idx_new])]

        s_new = s.copy()
        c_new = c.copy()
        successful = False
        if sched_new.size == 0:
            s_new[i] = k_new
            c_new[i] = self.prob.a[i]
            successful = True
        else:
            for x, y in enumerate(sched_new, 0):
                if c_new[y] + self.prob.d[y] > self.prob.a[i]:
                    t_1 = self.prob.a[i] if x == 0 \
                        else np.maximum(c_new[sched_new[x - 1]] + self.prob.d[sched_new[x - 1]], self.prob.a[i])
                    t_2 = np.minimum(self.prob.b[y] - self.prob.d[y], self.prob.b[i]) if x == sched_new.size - 1 \
                        else np.minimum(self.attempt_shift_right(s_new, c_new, k_new, x), self.prob.b[i])
                    if t_2 - t_1 >= self.prob.d[i]:
                        c_new = self.shift_right(s_new, c_new, k_new, x, t_1 + self.prob.d[i])
                        c_new[i] = t_1
                        s_new[i] = k_new
                        c_new = self.shift_left(s_new, c_new, k, loc)
                        successful = True
                        break
                    if t_1 >= self.prob.b[i]:
                        break
                    # special case last flight in sequence
                    if x == sched_new.size - 1 and self.prob.b[i] - self.prob.d[i] >= c_new[y] + self.prob.d[y]:
                        c_new[i] = c_new[y] + self.prob.d[y]
                        s_new[i] = k_new
                        c_new = self.shift_left(s_new, c_new, k, loc)
                        successful = True
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

        idx_2, = np.nonzero(s_new == k_2)
        sched_2 = idx_2[np.argsort(c_new[idx_2])]
        if(sched_2.size == 0):
            return s, c, False
        loc_i_2 = np.random.randint(0, sched_2.size)
        i_2 = sched_2[loc_i_2]
        loc_j_2 = np.random.randint(loc_i_2, sched_2.size)
        j_2 = sched_2[loc_j_2]

        # calculation of max gap sizes and min interval sizes
        latest_start_int_1 = self.attempt_shift_interval_right(s_new, c_new, k_1, loc_i_1, loc_j_1)
        latest_start_int_2 = self.attempt_shift_interval_right(s_new, c_new, k_2, loc_i_2, loc_j_2)
        earliest_end_int_1 = c_new[j_1]+self.prob.d[j_1]
        earliest_end_int_2 = c_new[j_2] + self.prob.d[j_2]
        earliest_start_gap_1 = self.prob.a[i_2] if loc_i_1 == 0 else np.maximum(self.prob.a[i_2],
                                                                                c_new[sched_1[loc_i_1 - 1]] +
                                                                                self.prob.d[sched_1[loc_i_1 - 1]])
        earliest_start_gap_2 = self.prob.a[i_1] if loc_i_2 == 0 else np.maximum(self.prob.a[i_1],
                                                                                c_new[sched_2[loc_i_2-1]] +
                                                                                self.prob.d[sched_2[loc_i_2-1]])
        latest_end_gap_1 = self.prob.b[j_2] if loc_j_1 == sched_1.size - 1 else np.minimum(self.prob.b[j_2],
                                                                                c_new[sched_1[loc_j_1+1]])
        latest_end_gap_2 = self.prob.b[j_1] if loc_j_2 == sched_2.size - 1 else np.minimum(self.prob.b[j_1],
                                                                                c_new[sched_2[loc_j_2 + 1]])
        # general infeasibility check
        if(earliest_start_gap_1 > latest_start_int_2
                or earliest_start_gap_2 > latest_start_int_1
                or earliest_end_int_1 > latest_end_gap_2
                or earliest_end_int_2 > latest_end_gap_1):
            return s, c, False
        else:
            # fitting test
            #todo maybe optimize: fitting_test return also c in order to avoid calculating it again with shift_interval
            succ_1, start_1 = self.fitting_test(earliest_start_gap_1, latest_end_gap_1,
                                                c_new, sched_2[loc_i_2:(loc_j_2+1)])
            succ_2, start_2 = self.fitting_test(earliest_start_gap_2, latest_end_gap_2,
                                                c_new, sched_1[loc_i_1:(loc_j_1 + 1)])
            if(succ_1 and succ_2):
                c_new = self.shift_interval(s_new, c_new, k_1, loc_i_1, loc_j_1, start_2)
                c_new = self.shift_interval(s_new, c_new, k_2, loc_i_2, loc_j_2, start_1)
                s_new[sched_1[loc_i_1:(loc_j_1 + 1)]] = k_2
                s_new[sched_2[loc_i_2:(loc_j_2 + 1)]] = k_1
                return s_new, c_new, True
            else:
                return s, c, False

    #_________________________________________________________________________________________________
    # Sub-routines

    # assumption that start >= a
    def fitting_test(self, start, end, c, sched_slice):
        c_new = c.copy()
        for x, y in enumerate(sched_slice, 0):
            if(x==0):
                c_new[y] = start
            else:
                c_new[y] = np.maximum(self.prob.a[y],
                                      c_new[sched_slice[x-1]] + self.prob.d[sched_slice[x-1]])
                if(c_new[y]+self.prob.d[y] > self.prob.b[y]):
                    return False, 0
        if(c_new[sched_slice[sched_slice.size-1]] + self.prob.d[sched_slice[sched_slice.size-1]] > end):
            return False, 0

        # todo optimize
        """ possibly faster way to test validity (not tested)
        infeas = np.greater(self.prob.c_new[sched_slice] + self.prob.d[sched_slice], self.prob.b[sched_slice])
        if (True in infeas 
                or c_new[sched_slice[sched_slice.size-1]] + self.prob.d[sched_slice[sched_slice.size-1]] > end):
            return False, 0
        """
        return True, c_new[sched_slice[0]]

    def shift_left(self, s, c, k, i):
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched[i:], i):
            c_new = self.prob.a[y] if x == 0 \
                else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def shift_right(self, s, c, k, i, t):
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched[i:], i):
            c_new = np.maximum(c[y], t) if x == i \
                else np.maximum(c[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def attempt_shift_right(self, s, c, k, i):
        c_temp = c.copy()
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c_temp[idx])]
        for x, y in enumerate(sched[i:][::-1]):
            c_new = self.prob.b[y] - self.prob.d[y] if x == 0 \
                else np.minimum(self.prob.b[y], c_temp[sched[sched.size - x]]) - self.prob.d[y]
            c_temp[y] = c_new
        return c_temp[sched[i]]

    # todo possibly optimizable
    def attempt_shift_interval_right(self, s_old, c_old, k, i, j):
        c = c_old.copy()
        s = s_old.copy()
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        sched_revers = sched[i:(j+1)][::-1]
        for x, y in enumerate(sched_revers, 0):
            if x==0 and j==(sched.size-1):
                c[sched_revers[x]] = self.prob.b[sched_revers[x]] - self.prob.d[sched_revers[x]]
            else:
                c[sched_revers[x]] = np.minimum(self.prob.b[sched_revers[x]], c[sched[j-x+1]]) - self.prob.d[sched_revers[x]]
        return c[sched[i]]

    def shift_interval(self, s_old, c_old, k, i, j, t):
        c = c_old.copy()
        s = s_old.copy()
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched[i:(j+1)], i):
            c_new = np.maximum(t, self.prob.a[sched[x]]) if x == i else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c

    def attempt_shift_interval(self, s_old, c_old, k, i, j, t):
        c = c_old.copy()
        s = s_old.copy()
        idx, = np.nonzero(s == k)
        sched = idx[np.argsort(c[idx])]
        for x, y in enumerate(sched[i:(j+1)], i):
            c_new = np.maximum(t, self.prob.a[sched[x]]) if x == i \
                else np.maximum(self.prob.a[y], c[sched[x - 1]] + self.prob.d[sched[x - 1]])
            c[y] = c_new
        return c[sched[j]] + self.prob.d[sched[j]]

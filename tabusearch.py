import numpy as np
from numba import njit
import timeit


class TabuSearch:

    def __init__(self,
                 prob,
                 n_iter=10**6,
                 n_neigh=100,
                 n_tabu_tenure=10,
                 n_term=10**4,
                 initial=None,
                 initial_c=None,
                 initial_obj=None):

        # parameters
        self.n_iter = n_iter
        self.n_neigh = n_neigh
        self.n_tabu_tenure = n_tabu_tenure
        self.n_term = n_term

        # input
        self.prob = prob

        # initial solution
        self.initial = initial
        self.initial_c = initial_c
        self.initial_obj = initial_obj

        # output
        self.best = None
        self.best_c = None
        self.best_obj = None
        self.solutions = np.empty(0)
        self.runtimes = np.empty(0)

    # Main function
    def solve(self, show_print=True):

        start_time = timeit.default_timer()
        if show_print:
            print('Beginning Tabu Search')

        # search arrays
        tabu_tenure = -np.ones([self.n_tabu_tenure, self.prob.n], dtype=np.int64)

        if self.initial is None:
            # create initial solution
            count_infeasible = 0
            while self.best_c is None:
                if count_infeasible >= 10000:
                    print("Greedy heuristic cannot find feasible solution")
                    return None, None, None
                self.best, self.best_c = self.create_solution()
                count_infeasible += 1
            self.best_obj = self.calculate_objective_value(self.best[np.newaxis, :], self.best_c[np.newaxis, :])[0]
        else:
            self.best, self.best_c, self.best_obj = self.initial, self.initial_c, self.initial_obj

        self.solutions = np.append(self.solutions, self.best_obj)
        self.runtimes = np.append(self.runtimes, timeit.default_timer() - start_time)
        if show_print:
            print('{:<10}{:>15}{:>10}'.format('Iter', 'Best Obj', 'Time'))
            print('{:<10}{:>15.4f}{:>9.0f}{}'.format('init',
                                                     self.best_obj,
                                                     timeit.default_timer() - start_time, 's'))

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

            neigh, neigh_c = self.generate_neighbours(curr, curr_c)
            neigh_obj = self.calculate_objective_value(neigh, neigh_c)

            # Condition for current solution update
            allowed = neigh_obj[np.logical_or((neigh[:, np.newaxis] != tabu_tenure).any(-1).all(-1),
                                              neigh_obj < self.best_obj)]
            if allowed.size > 0:
                top = np.argmin(allowed)
                curr = neigh.copy()[top]
                curr_c = neigh_c.copy()[top]
                curr_obj = neigh_obj.copy()[top]
                tabu_tenure = np.roll(tabu_tenure, 1, axis=0)
                tabu_tenure[0] = curr
                if curr_obj < self.best_obj:
                    self.best = curr.copy()
                    self.best_c = curr_c.copy()
                    self.best_obj = curr_obj.copy()
                    self.solutions = np.append(self.solutions, self.best_obj)
                    self.runtimes = np.append(self.runtimes, timeit.default_timer() - start_time)
                    if show_print:
                        print('{:<10}{:>15.4f}{:>9.0f}{}'.format(count_iter + 1,
                                                                 self.best_obj,
                                                                 timeit.default_timer() - start_time, 's'))
                    count_term = 0
                else:
                    count_term += 1
            else:
                count_term += 1

        if show_print:
            self.solutions = np.append(self.solutions, self.best_obj)
            self.runtimes = np.append(self.runtimes, timeit.default_timer() - start_time)
            print('Termination criterion reached')
            print('{}{}'.format('Best objective value is ', self.best_obj))
            print('{}{}'.format('Time is ', timeit.default_timer() - start_time))

        return self.best, self.best_c, self.best_obj

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

    def generate_neighbours(self, curr, curr_c):
        return self._generate_neighbours(self.n_neigh,
                                         self.prob.n, self.prob.m, self.prob.a, self.prob.b, self.prob.d,
                                         curr, curr_c)

    @staticmethod
    @njit
    def _generate_neighbours(n_neigh, n, m, a, b, d, curr, curr_c):
        neigh = np.empty((n_neigh, n), dtype=np.int64)
        neigh_c = np.empty((n_neigh, n))
        count_neigh = 0
        while count_neigh < n_neigh:
            move = np.random.rand() >= 0.5
            if move:
                s, c, successful = _insert(n, m, a, b, d, curr, curr_c)
                if successful:
                    neigh[count_neigh] = s
                    neigh_c[count_neigh] = c
                    count_neigh += 1
            else:
                s, c, successful = _interval_exchange(m, a, b, d, curr, curr_c)
                if successful:
                    neigh[count_neigh] = s
                    neigh_c[count_neigh] = c
                    count_neigh += 1
        return neigh, neigh_c


@njit
def _interval_exchange(m, a, b, d, s, c):

    k_1, k_2 = np.random.choice(m, 2, replace=False)

    idx_1, = np.nonzero(s == k_1)
    sched_1 = idx_1[np.argsort(c[idx_1])]
    if sched_1.size == 0:
        return s, c, False
    loc_i_1, loc_j_1 = np.sort(np.random.randint(0, sched_1.size, 2))
    i_1 = sched_1[loc_i_1]
    j_1 = sched_1[loc_j_1]

    idx_2, = np.nonzero(s == k_2)
    sched_2 = idx_2[np.argsort(c[idx_2])]
    if sched_2.size == 0:
        return s, c, False
    loc_i_2, loc_j_2 = np.sort(np.random.randint(0, sched_2.size, 2))
    i_2 = sched_2[loc_i_2]
    j_2 = sched_2[loc_j_2]

    # calculation of max gap sizes and min interval sizes
    latest_start_int_1 = _attempt_shift_interval_right(b, d, s, c, k_1, loc_i_1, loc_j_1)
    latest_start_int_2 = _attempt_shift_interval_right(b, d, s, c, k_2, loc_i_2, loc_j_2)
    earliest_end_int_1 = c[j_1] + d[j_1]
    earliest_end_int_2 = c[j_2] + d[j_2]
    earliest_start_gap_1 = a[i_2] if loc_i_1 == 0 else np.maximum(a[i_2],
                                                                  c[sched_1[loc_i_1 - 1]] + d[sched_1[loc_i_1 - 1]])
    earliest_start_gap_2 = a[i_1] if loc_i_2 == 0 else np.maximum(a[i_1],
                                                                  c[sched_2[loc_i_2 - 1]] + d[sched_2[loc_i_2 - 1]])
    latest_end_gap_1 = b[j_2] if loc_j_1 == sched_1.size - 1 else np.minimum(b[j_2], c[sched_1[loc_j_1 + 1]])
    latest_end_gap_2 = b[j_1] if loc_j_2 == sched_2.size - 1 else np.minimum(b[j_1], c[sched_2[loc_j_2 + 1]])

    # general infeasibility check
    if earliest_start_gap_1 > latest_start_int_2 \
            or earliest_start_gap_2 > latest_start_int_1 \
            or earliest_end_int_1 > latest_end_gap_2 \
            or earliest_end_int_2 > latest_end_gap_1:
        return s, c, False
    else:
        result_1 = _attempt_shift_interval(d, s, c, k_1, loc_i_1, loc_j_1, earliest_start_gap_2)
        result_2 = _attempt_shift_interval(d, s, c, k_2, loc_i_2, loc_j_2, earliest_start_gap_1)
        if result_1 <= latest_end_gap_2 and result_2 <= latest_end_gap_1:
            s_new = np.copy(s)
            c_new = np.copy(c)
            c_new = _shift_interval(d, s_new, c_new, k_1, loc_i_1, loc_j_1, earliest_start_gap_2)
            c_new = _shift_interval(d, s_new, c_new, k_2, loc_i_2, loc_j_2, earliest_start_gap_1)
            s_new[sched_1[loc_i_1:loc_j_1 + 1]] = k_2
            s_new[sched_2[loc_i_2:loc_j_2 + 1]] = k_1
            c_new = _shift_left(a, d, s_new, c_new, k_1, 0)
            c_new = _shift_left(a, d, s_new, c_new, k_2, 0)
            return s_new, c_new, True
        else:
            return s, c, False


@njit
def _insert(n, m, a, b, d,  s, c):

    # Choose randomly a flight i and gate k
    i = np.random.randint(0, n)
    k = s[i]
    idx, = np.nonzero(s == k)
    sched = idx[np.argsort(c[idx])]
    loc = np.nonzero(sched == i)[0][0]

    k_new = np.random.choice(np.delete(np.arange(m), k))
    idx_new, = np.nonzero(s == k_new)
    sched_new = idx_new[np.argsort(c[idx_new])]

    s_new = s.copy()
    c_new = c.copy()
    successful = False
    if sched_new.size == 0:
        s_new[i] = k_new
        c_new[i] = a[i]
        successful = True
    else:
        for x, y in enumerate(sched_new, 0):
            if c[y] + d[y] > a[i]:
                t_1 = a[i] if x == 0 \
                    else np.maximum(c[sched_new[x - 1]] + d[sched_new[x - 1]], a[i])
                t_2 = np.minimum(_attempt_shift_right(b, d, s, c, k_new, x), b[i])
                if t_2 - t_1 >= d[i]:
                    c_new = _shift_right(d, s_new, c_new, k_new, x, t_1 + d[i])
                    c_new[i] = t_1
                    s_new[i] = k_new
                    c_new = _shift_left(a, d, s_new, c_new, k, loc)
                    successful = True
                    break
                if t_1 >= b[i]:
                    break
                # special case last flight in sequence
                if x == sched_new.size - 1 and b[i] - d[i] >= c[y] + d[y]:
                    c_new[i] = c_new[y] + d[y]
                    s_new[i] = k_new
                    c_new = _shift_left(a, d, s_new, c_new, k, loc)
                    successful = True
    return s_new, c_new, successful


@njit
def _shift_left(a, d, s, c, k, i):
    idx, = np.nonzero(s == k)
    sched = idx[np.argsort(c[idx])]
    for x, y in enumerate(sched[i:], i):
        c_new = a[y] if x == 0 \
            else np.maximum(a[y], c[sched[x - 1]] + d[sched[x - 1]])
        c[y] = c_new
    return c


@njit
def _shift_right(d, s, c, k, i, t):
    idx, = np.nonzero(s == k)
    sched = idx[np.argsort(c[idx])]
    for x, y in enumerate(sched[i:], i):
        c_new = np.maximum(c[y], t) if x == i \
            else np.maximum(c[y], c[sched[x - 1]] + d[sched[x - 1]])
        c[y] = c_new
    return c


@njit
def _attempt_shift_right(b, d, s, c, k, i):
    c_temp = c.copy()
    idx, = np.nonzero(s == k)
    sched = idx[np.argsort(c_temp[idx])]
    for x, y in enumerate(sched[i:][::-1]):
        c_new = b[y] - d[y] if x == 0 \
            else np.minimum(b[y], c_temp[sched[sched.size - x]]) - d[y]
        c_temp[y] = c_new
    return c_temp[sched[i]]


@njit
def _shift_interval(d, s, c, k, i, j, t):
    idx, = np.nonzero(s == k)
    sched = idx[np.argsort(c[idx])]
    for x, y in enumerate(sched[i:j + 1], i):
        c_new = t if x == i else np.maximum(c[y], c[sched[x - 1]] + d[sched[x - 1]])
        c[y] = c_new
    return c


@njit
def _attempt_shift_interval(d, s, c, k, i, j, t):
    c_temp = c.copy()
    idx, = np.nonzero(s == k)
    sched = idx[np.argsort(c_temp[idx])]
    for x, y in enumerate(sched[i:j + 1], i):
        c_new = t if x == i else np.maximum(c_temp[y], c_temp[sched[x - 1]] + d[sched[x - 1]])
        c_temp[y] = c_new
    return c_temp[sched[j]] + d[sched[j]]


@njit
def _attempt_shift_interval_right(b, d, s, c, k, i, j):
    c_temp = c.copy()
    idx, = np.nonzero(s == k)
    sched = idx[np.argsort(c_temp[idx])]
    for x, y in enumerate(sched[i:j + 1][::-1]):
        c_new = b[y] - d[y] if x == 0 and j == sched.size - 1 \
            else np.minimum(b[y], c_temp[sched[j - x + 1]]) - d[y]
        c_temp[y] = c_new
    return c_temp[sched[i]]

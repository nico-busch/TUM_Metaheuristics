import numpy as np
from numba import njit
import timeit


class BeeColony:

    def __init__(self,
                 problem,
                 n_iter=10**5,
                 n_term=100,
                 n_bees=10**3):

        # Parameters
        self.n_iter = n_iter
        self.n_term = n_term
        self.n_bees = n_bees

        # Input
        self.prob = problem

        # Output
        self.best = None
        self.best_c = None
        self.best_obj = np.inf
        self.solutions = np.empty(0)
        self.runtimes = np.empty(0)

    def solve(self, show_print=True):

        start_time = timeit.default_timer()
        if show_print:
            print('Beginning Bee Colony Optimization')
            print('{:<10}{:>15}{:>10}'.format('Iter', 'Best Obj', 'Time'))

        count_term = 0
        for x in range(self.n_iter):

            # Determine terminal condition
            if count_term >= self.n_term:
                break

            # Initialize bees
            bees = -np.ones([self.n_bees, self.prob.n], dtype=np.int64)
            bees_c = np.empty([self.n_bees, self.prob.n])
            bees_obj = np.empty(self.n_bees)

            for n, i in enumerate(self.prob.a.argsort(), 1):

                # Generate new step
                bees[:, i] = np.random.randint(0, self.prob.m, self.n_bees)
                bees, bees_c = self.generate_solution(i, bees, bees_c)

                # Get feasible bees
                feasible = bees[:, i] != -1
                if not feasible.any():
                    print("The algorithm cannot find a feasible solution")
                    return None

                # Calculate objective value
                bees_obj[feasible] = self.calculate_objective_value(bees[feasible], bees_c[feasible])
                bees_obj[~feasible] = np.inf
                if np.ptp(bees_obj[feasible]) == 0:
                    continue

                # Calculate normalized objective value
                bees_obj_norm = np.empty(self.n_bees)
                bees_obj_norm[feasible] = (np.amax(bees_obj[feasible]) - bees_obj[feasible]) / \
                                          (np.amax(bees_obj[feasible]) - np.amin(bees_obj[feasible]))

                # Decide loyalty
                p_loy = np.empty(self.n_bees)
                p_loy[feasible] = np.exp((bees_obj_norm[feasible] - np.amax(bees_obj_norm[feasible])))
                p_loy[~feasible] = 0
                r_loy = np.random.rand(self.n_bees)
                rec = r_loy <= p_loy
                fol = r_loy > p_loy

                # Determine followers
                p_fol = bees_obj_norm[rec] / np.sum(bees_obj_norm[rec])
                choice = np.random.choice(np.arange(self.n_bees)[rec], bees[fol].shape[0], replace=True, p=p_fol)
                bees[fol] = bees[choice]
                bees_c[fol] = bees_c[choice]

            # Update best solution
            if np.amin(bees_obj) < self.best_obj:
                best_idx = np.argmin(bees_obj)
                self.best = bees[best_idx]
                self.best_c = bees_c[best_idx]
                self.best_obj = bees_obj[best_idx]
                self.solutions = np.append(self.solutions, self.best_obj)
                self.runtimes = np.append(self.runtimes, timeit.default_timer() - start_time)
                if show_print:
                    print('{:<10}{:>15.4f}{:>9.0f}{}'.format(x + 1,
                                                             self.best_obj,
                                                             timeit.default_timer() - start_time, 's'))
                count_term = 0
            else:
                count_term += 1

        self.solutions = np.append(self.solutions, self.best_obj)
        self.runtimes = np.append(self.runtimes, timeit.default_timer() - start_time)
        if show_print:
            print('Termination criterion reached')
            print('{}{}'.format('Best objective value is ', self.best_obj))
            print('{}{}'.format('Time is ', timeit.default_timer() - start_time))

        return self.best, self.best_c, self.best_obj

    def generate_solution(self, i, s, c):
        return self._generate_solution(i, self.prob.m, self.prob.a, self.prob.b, self.prob.d, s, c)

    '''
    Algorithm to calculate feasible start time for a single flight to gate assignment.
    This method used numba's JIT compilation in nopython mode for faster for loops.
    '''
    @staticmethod
    @njit
    def _generate_solution(i, m, a, b, d, s, c):
        s_new = s.copy()
        s_new[:, i] = -1
        for x in range(s.shape[0]):
            for k in np.roll(np.arange(m), -s[x, i]):
                c_max = a[i] if s_new[x][s_new[x] == k].size == 0 \
                    else np.amax(np.maximum(c[x] + d, a[i])[s_new[x] == k])
                if b[i] - c_max >= d[i]:
                    c[x, i] = c_max
                    s_new[x, i] = k
                    break
        return s_new, c

    '''
    Method to calculate the objective values for all bees for a single step.
    Utilizes np.einsum for fast tensor multiplication.
    '''
    def calculate_objective_value(self, s, c):
        delay = np.where(s != -1, c - self.prob.a, 0)
        sum_delay_penalty = np.einsum('i,ni->n', self.prob.p, delay)

        x = np.zeros([s.size, self.prob.m], dtype=np.int64)
        x[np.arange(s.size), s.ravel()] = 1
        x.shape = s.shape + (self.prob.m,)
        x[s == -1] = 0
        sum_walking_distance = np.einsum('nik,njl,ij,kl->n', x, x, self.prob.f, self.prob.w, optimize=True)

        return sum_delay_penalty + sum_walking_distance


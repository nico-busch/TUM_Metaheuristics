import numpy as np
from numba import njit


class BeeColony:

    def __init__(self,
                 problem,
                 n_iter=10**4,
                 n_term=500,
                 n_bees=300):

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

    def solve(self):

        print('{:<10}{:>10}'.format('Iter', 'Best Obj'))

        count_term = 0
        for x in range(self.n_iter):

            # Determine terminal condition
            if count_term >= self.n_term:
                break

            # Initialize bees
            bees = -np.ones((self.n_bees, self.prob.n), dtype=np.int64)
            bees_c = np.zeros((self.n_bees, self.prob.n))
            bees_obj = np.empty(self.n_bees)

            for n, i in enumerate(self.prob.a.argsort(), 1):

                # Generate new step
                bees[:, i] = np.random.randint(0, self.prob.m, self.n_bees)
                bees, bees_c, infeas = self.generate_solution(i, bees, bees_c)

                # Get feasible bees
                feas = np.ones(self.n_bees, dtype=np.bool_)
                feas[infeas] = 0
                if not feas.any():
                    print("The algorithm cannot find a feasible solution")
                    return None

                # Calculate objective value
                bees_obj[feas] = self.calculate_objective_value(bees[feas], bees_c[feas])
                bees_obj[infeas] = np.inf

                if np.ptp(bees_obj[feas]) == 0:
                    continue

                # Calculate normalized objective value
                bees_obj_norm = np.empty(self.n_bees)
                bees_obj_norm[feas] = (np.amax(bees_obj[feas]) - bees_obj[feas]) \
                                      / (np.amax(bees_obj[feas]) - np.amin(bees_obj[feas]))

                # Decide loyalty
                p_loy = np.empty(self.n_bees)
                p_loy[feas] = np.exp((bees_obj_norm[feas] - np.amax(bees_obj_norm[feas])) / n)
                p_loy[infeas] = 0
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
                count_term = 0
            else:
                count_term += 1

            print('{:<10}{:>10.4f}'.format(x + 1, self.best_obj))

        return self.best_obj

    def generate_solution(self, i, s, c):
        return self._generate_solution(i, self.prob.m, self.prob.a, self.prob.b, self.prob.d, s, c)

    '''
    Algorithm to calculate feasible start time for a single flight to gate assignment.
    This method used numba's JIT compilation in nopython mode for faster for loops.
    '''
    @staticmethod
    @njit
    def _generate_solution(i, m, a, b, d, s, c):
        infeasible = np.empty(0, dtype=np.int64)
        s_new = s.copy()
        s_new[:, i] = -1
        for x in range(s.shape[0]):
            k = s[x, i]
            successful = False
            count_k = 0
            while count_k < m:
                c_max = a[i] if s_new[x][s_new[x] == k].size == 0 \
                    else np.amax(np.maximum(c[x] + d, a[i])[s_new[x] == k])
                if b[i] - c_max >= d[i]:
                    c[x, i] = c_max
                    s_new[x, i] = k
                    successful = True
                    break
                else:
                    k = k % (m - 1) + 1
                    count_k += 1
            if not successful:
                infeasible = np.append(infeasible, x)
        return s_new, c, infeasible

    '''
    Method to calculate the objective values for all bees for a single step.
    Utilizes np.einsum for fast tensor multiplication.
    '''
    def calculate_objective_value(self, s, c):
        sum_delay_penalty = np.einsum('i,ni->n', self.prob.p, c - self.prob.a)

        x = np.zeros([s.size, self.prob.m], dtype=np.int64)
        x[np.arange(s.size), s.ravel()] = 1
        x.shape = s.shape + (self.prob.m,)
        x[s == -1] = 0
        sum_walking_distance = np.einsum('nik,njl,ij,kl->n', x, x, self.prob.f, self.prob.w, optimize=True)

        return sum_delay_penalty + sum_walking_distance

import numpy as np
from numba import njit


class BeeColony:

    def __init__(self,
                 problem,
                 n_iter=10,
                 n_term=500,
                 n_bees=100):

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

        # Create the initial population
        count_term = 0

        print('{:<10}{:>10}'.format('Iter', 'Best Obj'))

        for x in range(self.n_iter):

            # Determine terminal condition
            if count_term >= self.n_term:
                break

            bees = -np.ones((self.n_bees, self.prob.n), dtype=np.int64)
            bees_c = np.zeros((self.n_bees, self.prob.n))
            bees_obj = np.empty(self.n_bees)

            for n, i in enumerate(self.prob.a.argsort(), 1):

                bees[:, i] = np.random.randint(0, self.prob.m, self.n_bees)

                bees, bees_c, infeas = self.generate_solution(i, bees, bees_c)
                feas = np.ones(self.n_bees, dtype=np.bool_)
                feas[infeas] = 0

                if not feas.any():
                    print("The algorithm cannot find a feasible solution")
                    return None

                bees_obj[feas] = self.calculate_objective_value(bees[feas], bees_c[feas])
                bees_obj[infeas] = np.inf

                if np.amax(bees_obj[feas]) == 0:
                    continue

                bees_obj_norm = np.empty(self.n_bees)
                bees_obj_norm[feas] = (np.amax(bees_obj[feas]) - bees_obj[feas]) \
                                      / (np.amax(bees_obj[feas]) - np.amin(bees_obj[feas]))

                p_loy = np.empty(self.n_bees)
                p_loy[feas] = np.exp((bees_obj_norm[feas] - np.amax(bees_obj_norm[feas])) / n)
                p_loy[infeas] = 0
                r_loy = np.random.rand(self.n_bees)
                rec = r_loy <= p_loy
                fol = r_loy > p_loy
                p_fol = np.empty(self.n_bees)
                p_fol[feas] = bees_obj_norm[rec] / np.sum(bees_obj_norm[rec])
                choice = np.random.choice(np.arange(bees[rec].shape[0]), bees[fol].shape[0], replace=True, p=p_fol)
                bees[fol] = bees[rec][choice]
                bees_c[fol] = bees_c[rec][choice]

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
    Greedy algorithm to calculate feasible start times for a given gate assignment.
    This method used numba's JIT compilation in nopython mode for faster for loops.
    '''
    @staticmethod
    @njit
    def _generate_solution(i, m, a, b, d, s, c):
        infeasible = np.empty(0, dtype=np.int64)
        for x in range(s.shape[0]):
            k = s[x, i]
            successful = False
            count_k = 0
            while count_k < m:
                c_max = a[i] if s[x][s[x] == k].size == 0 \
                    else np.amax(np.maximum(c[x] + d, a[i])[s[x] == k])
                if b[i] - c_max >= d[i]:
                    c[x, i] = c_max
                    s[x, i] = k
                    successful = True
                    break
                else:
                    k = k % (m - 1) + 1
                    count_k += 1
            if not successful:
                infeasible = np.append(infeasible, x)
        return s, c, infeasible

    '''
    Method to calculate the objective values for a whole population.
    Utilizes np.einsum for fast tensor multiplication.
    '''
    def calculate_objective_value(self, s, c):
        sum_delay_penalty = np.einsum('i,ni->n', self.prob.p, np.maximum(c - self.prob.a, 0))

        x = np.zeros([s.size, self.prob.m], dtype=np.int64)
        x[np.arange(s.size), s.ravel()] = 1
        x.shape = s.shape + (self.prob.m,)
        x[s == -1] = 0
        sum_walking_distance = np.einsum('nik,njl,ij,kl->n', x, x, self.prob.f, self.prob.w, optimize=True)

        return sum_delay_penalty + sum_walking_distance

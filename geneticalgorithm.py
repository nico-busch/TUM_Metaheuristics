import numpy as np
from numba import njit


class GeneticAlgorithm:

    def __init__(self,
                 problem,
                 n_iter=10**4,
                 n_term=500,
                 n_pop=300,
                 n_crossover=500,
                 crossover_type='2p',
                 p1=0.2):

        # Parameters
        self.n_iter = n_iter
        self.n_term = n_term
        self.n_pop = n_pop
        self.n_crossover = n_crossover
        self.crossover_type = crossover_type
        self.p1 = p1

        # Input
        self.prob = problem

        # Output
        self.best = None
        self.best_c = None
        self.best_obj = None

    def solve(self):

        # Create the initial population
        count_infeasible = 0
        pop = np.empty((0, self.prob.n), dtype=np.int64)
        pop_c = np.empty((0, self.prob.n))
        while pop.shape[0] < 300:
            if count_infeasible >= 10000:
                print("Greedy heuristic cannot find enough feasible solutions")
                return self.best_obj
            else:
                s, c = self.generate_solution(
                    np.random.randint(0, self.prob.m, [self.n_pop - pop.shape[0], self.prob.n]))
                count_infeasible += self.n_pop - pop.shape[0] - s.shape[0]
                pop = np.vstack([pop, s])
                pop_c = np.vstack([pop_c, c])
        pop_obj = self.calculate_objective_value(pop, pop_c)

        # Find the initial best solution
        best_idx = np.argmin(pop_obj)
        self.best = pop[best_idx]
        self.best_c = pop_c[best_idx]
        self.best_obj = pop_obj[best_idx]

        print('{:<10}{:>10}'.format('Iter', 'Best Obj'))
        print('{:<10}{:>10.4f}'.format('init', self.best_obj))

        count_term = 0
        for x in range(self.n_iter):

            # Determine terminal condition
            if count_term >= self.n_term:
                break

            # Perform crossover
            par1 = pop[np.random.randint(0, self.n_pop, self.n_crossover), :]
            par2 = pop[np.random.randint(0, self.n_pop, self.n_crossover), :]
            if self.crossover_type == '1p':
                cross = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
                r = np.arange(self.prob.n)
                off1 = np.where(r < cross, par1, par2)
                off2 = np.where(r < cross, par2, par1)
            elif self.crossover_type == '2p':
                cross1 = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
                cross2 = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
                r = np.arange(self.prob.n)
                off1 = np.where((r < cross1) & (r > cross2), par1, par2)
                off2 = np.where((r < cross1) & (r > cross2), par2, par1)
            else:
                raise KeyError("Invalid crossover type argument passed")
            off = np.vstack([off1, off2])

            # Perform mutation with probability p1
            switch1 = np.random.randint(0, self.prob.n, self.n_crossover * 2)
            switch2 = np.random.randint(0, self.prob.n, self.n_crossover * 2)
            mut = np.copy(off)
            mut[np.arange(self.n_crossover * 2), switch1], mut[np.arange(self.n_crossover * 2), switch2] \
                = off[np.arange(self.n_crossover * 2), switch2], off[np.arange(self.n_crossover * 2), switch1]
            off = np.where(np.random.rand(self.n_crossover * 2, 1) <= self.p1, mut, off)

            # Calculate objective values for offspring
            off, off_c = self.generate_solution(off)
            off_obj = self.calculate_objective_value(off, off_c)

            # Add to population and keep best individuals
            pop = np.vstack([pop, off])
            pop_c = np.vstack([pop_c, off_c])
            pop_obj = np.concatenate([pop_obj, off_obj])
            top_idx = np.argsort(pop_obj)[:self.n_pop]
            pop = pop[top_idx]
            pop_c = pop_c[top_idx]
            pop_obj = pop_obj[top_idx]

            # Update best solution
            if np.amin(pop_obj) < self.best_obj:
                best_idx = np.argmin(pop_obj)
                self.best = pop[best_idx]
                self.best_c = pop_c[best_idx]
                self.best_obj = pop_obj[best_idx]
                count_term = 0
            else:
                count_term += 1

            print('{:<10}{:>10.4f}'.format(x + 1, self.best_obj))

        return self.best_obj

    def generate_solution(self, s):
        return self._generate_solution(self.prob.n, self.prob.m, self.prob.a, self.prob.b, self.prob.d, s)

    '''
    Greedy algorithm to calculate feasible start times for a given gate assignment.
    This method used numba's JIT compilation in nopython mode for faster for loops.
    '''
    @staticmethod
    @njit
    def _generate_solution(n, m, a, b, d, s):
        s_new = -np.ones((s.shape[0], n), dtype=np.int64)
        c = np.empty((s.shape[0], n))
        infeasible = np.empty(0, dtype=np.int64)
        for x in range(s.shape[0]):
            for i in a.argsort():
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
                    break
        delete = np.ones(s.shape[0], dtype=np.bool_)
        delete[infeasible] = 0
        s_new = s_new[delete]
        c = c[delete]
        return s_new, c

    '''
    Method to calculate the objective values for a whole population.
    Utilizes np.einsum for fast tensor multiplication.
    '''
    def calculate_objective_value(self, s, c):
        sum_delay_penalty = np.einsum('i,ni->n', self.prob.p, (c - self.prob.a))

        x = np.zeros([s.size, self.prob.m], dtype=np.int64)
        x[np.arange(s.size), s.ravel()] = 1
        x.shape = s.shape + (self.prob.m,)
        sum_walking_distance = np.einsum('nik,njl,ij,kl->n', x, x, self.prob.f, self.prob.w, optimize=True)

        return sum_delay_penalty + sum_walking_distance

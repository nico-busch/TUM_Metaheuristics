import numpy as np
from numba import njit
import timeit


class GeneticAlgorithm:

    def __init__(self,
                 prob,
                 n_iter=10**5,
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
        self.prob = prob

        # Output
        self.best = None
        self.best_c = None
        self.best_obj = None
        self.solutions = np.empty(0)
        self.runtimes = np.empty(0)

    def solve(self, show_print=True):

        start_time = timeit.default_timer()
        if show_print:
            print('Beginning Genetic Algorithm')

        # Create the initial population
        count_infeasible = 0
        pop = np.empty([0, self.prob.n], dtype=np.int64)
        pop_c = np.empty([0, self.prob.n])
        while pop.shape[0] < self.n_pop:
            if count_infeasible >= 10000:
                print("The algorithm cannot find enough feasible solutions")
                return None, None, None
            else:
                s, c = self.generate_solution(
                    np.random.randint(0, self.prob.m, [self.n_pop - pop.shape[0], self.prob.n]))
                pop = np.vstack([pop, s])
                pop_c = np.vstack([pop_c, c])
                count_infeasible += self.n_pop - pop.shape[0]
        pop_obj = self.calculate_objective_value(pop, pop_c)

        # Find the initial best solution
        best_idx = np.argmin(pop_obj)
        self.best = pop[best_idx]
        self.best_c = pop_c[best_idx]
        self.best_obj = pop_obj[best_idx]

        self.solutions = np.append(self.solutions, self.best_obj)
        self.runtimes = np.append(self.runtimes, timeit.default_timer() - start_time)
        if show_print:
            print('{:<10}{:>15}{:>10}'.format('Iter', 'Best Obj', 'Time'))
            print('{:<10}{:>15.4f}{:>9.0f}{}'.format('init',
                                                     self.best_obj,
                                                     timeit.default_timer() - start_time, 's'))

        count_term = 0
        for x in range(self.n_iter):

            # Determine terminal condition
            if count_term >= self.n_term:
                break

            # Perform crossover
            par1 = pop[np.random.randint(self.n_pop, size=self.n_crossover), :]
            par2 = pop[np.random.randint(self.n_pop, size=self.n_crossover), :]
            if self.crossover_type == '1p':
                cross = np.random.randint(self.prob.n, size=[self.n_crossover, 1])
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
            switch1 = np.random.randint(self.prob.n, size=self.n_crossover * 2)
            switch2 = np.random.randint(self.prob.n, size=self.n_crossover * 2)
            mut = np.copy(off)
            mut[np.arange(self.n_crossover * 2), switch1], mut[np.arange(self.n_crossover * 2), switch2] \
                = off[np.arange(self.n_crossover * 2), switch2], off[np.arange(self.n_crossover * 2), switch1]
            off = np.where(np.random.rand(self.n_crossover * 2, 1) <= self.p1, mut, off)

            # Calculate objective values for offspring
            off, off_c = self.generate_solution(off)
            off_obj = self.calculate_objective_value(off, off_c)

            # Carry top individuals to the next generation
            top_idx = np.argsort(off_obj)[:self.n_pop]
            pop = off[top_idx]
            pop_c = off_c[top_idx]
            pop_obj = off_obj[top_idx]

            import sys
            np.set_printoptions(threshold=sys.maxsize)

            # Update best solution
            if np.amin(pop_obj) < self.best_obj:
                best_idx = np.argmin(pop_obj)
                self.best = pop[best_idx]
                self.best_c = pop_c[best_idx]
                self.best_obj = pop_obj[best_idx]
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

    def generate_solution(self, s):
        return self._generate_solution(self.prob.m, self.prob.a, self.prob.b, self.prob.d, s)

    '''
    Greedy algorithm to calculate feasible start times for a given gate assignment.
    This method used numba's JIT compilation in nopython mode for faster for loops.
    '''
    @staticmethod
    @njit
    def _generate_solution(m, a, b, d, s):
        s_new = -np.ones(s.shape, dtype=np.int64)
        c = np.zeros(s.shape)
        infeasible = np.empty(0, dtype=np.int64)
        for x in range(s.shape[0]):
            for i in a.argsort():
                successful = False
                for k in np.roll(np.arange(m), -s[x, i]):
                    c_max = a[i] if s_new[x][s_new[x] == k].size == 0 \
                        else np.amax(np.maximum(c[x] + d, a[i])[s_new[x] == k])
                    if b[i] - c_max >= d[i]:
                        c[x, i] = c_max
                        s_new[x, i] = k
                        successful = True
                        break
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

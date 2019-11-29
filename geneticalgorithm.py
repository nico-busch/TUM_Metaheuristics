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

        # parameter
        self.n_iter = n_iter
        self.n_term = n_term
        self.n_pop = n_pop
        self.n_crossover = n_crossover
        self.crossover_type = crossover_type
        self.p1 = p1

        # input
        self.prob = problem

        # population
        self.pop = np.empty([0, self.prob.n], dtype=np.int64)
        self.pop_c = np.empty([0, self.prob.n])
        self.pop_obj = None

        # output
        self.best = None
        self.best_c = None
        self.best_obj = None

    def solve(self):

        count_infeas = 0
        while self.pop.shape[0] < 300:
            s, c = self.generate_solution(
                np.random.randint(0, self.prob.m, [self.n_pop - self.pop.shape[0], self.prob.n]))
            self.pop = np.vstack([self.pop, s])
            self.pop_c = np.vstack([self.pop_c, c])
            if count_infeas >= 1000:
                print("Greedy heuristic cannot find enough feasible solutions")
                return None
            else:
                count_infeas += 1

        self.pop_obj = self.calculate_objective_value(self.pop, self.pop_c)
        best_idx = np.argmin(self.pop_obj)
        self.best = self.pop[best_idx]
        self.best_c = self.pop_c[best_idx]
        self.best_obj = self.pop_obj[best_idx]

        print('{:<10}{:>10}'.format('Iter', 'Best Obj'))
        print('{:<10}{:>10.4f}'.format('init', self.best_obj))

        count_term = 0
        for x in range(self.n_iter):

            # terminal condition
            if count_term >= self.n_term:
                break

            # crossover
            par1 = self.pop[np.random.randint(0, self.n_pop, self.n_crossover), :]
            par2 = self.pop[np.random.randint(0, self.n_pop, self.n_crossover), :]
            if self.crossover_type == '1p':
                cross = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
                r = np.arange(self.prob.n)
                off1 = np.where(r < cross, par1, par2)
                off2 = np.where(r < cross, par2, par1)
            if self.crossover_type == '2p':
                cross1 = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
                cross2 = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
                r = np.arange(self.prob.n)
                off1 = np.where((r < cross1) & (r > cross2), par1, par2)
                off2 = np.where((r < cross1) & (r > cross2), par2, par1)
            off = np.vstack([off1, off2])

            # mutation
            switch1 = np.random.randint(0, self.prob.n, self.n_crossover * 2)
            switch2 = np.random.randint(0, self.prob.n, self.n_crossover * 2)
            mut = np.copy(off)
            mut[np.arange(len(switch1)), switch1], mut[np.arange(len(switch2)), switch2] \
                = off[np.arange(len(switch2)), switch2], off[np.arange(len(switch1)), switch1]
            off = np.where(np.random.rand(self.n_crossover * 2, 1) <= self.p1, mut, off)

            # calculate objective values
            off, off_c = self.generate_solution(off)
            off_obj = self.calculate_objective_value(off, off_c)

            # add feasible individuals to population
            self.pop = np.vstack([self.pop, off])
            self.pop_c = np.vstack([self.pop_c, off_c])
            self.pop_obj = np.concatenate([self.pop_obj, off_obj])

            # keep best individuals
            top_idx = np.argsort(self.pop_obj)[:self.n_pop]
            self.pop = self.pop[top_idx]
            self.pop_c = self.pop_c[top_idx]
            self.pop_obj = self.pop_obj[top_idx]

            # update best solution
            if np.amin(self.pop_obj) < self.best_obj:
                best_idx = np.argmin(self.pop_obj)
                self.best = self.pop[best_idx]
                self.best_c = self.pop_c[best_idx]
                self.best_obj = self.pop_obj[best_idx]
                count_term = 0
            else:
                count_term += 1
            print('{:<10}{:>10.4f}'.format(x + 1, self.best_obj))

    def generate_solution(self, s):
        return self._generate_solution(self.prob.n, self.prob.m, self.prob.a, self.prob.b, self.prob.d, s)

    @staticmethod
    @njit
    def _generate_solution(n, m, a, b, d, s):
        s_new = -np.ones((s.shape[0], n), dtype=np.int64)
        c = np.zeros((s.shape[0], n))
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

    def calculate_objective_value(self, s, c):
        sum_delay_penalty = np.einsum('i,ni->n', self.prob.p, (c - self.prob.a))

        x = np.zeros([s.size, self.prob.m], dtype=np.int64)
        x[np.arange(s.size), s.ravel()] = 1
        x.shape = s.shape + (self.prob.m,)
        sum_walking_distance = np.einsum('nik,njl,ij,kl->n', x, x, self.prob.f, self.prob.w, optimize=True)

        return sum_delay_penalty + sum_walking_distance

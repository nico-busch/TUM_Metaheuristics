import numpy as np


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
        self.pop = np.empty([self.n_pop, self.prob.n], dtype=int)
        self.pop_obj = np.empty(self.n_pop)
        self.best = None

    def solve(self):

        count_term = 0
        if not self.create_initial_population():
            print("Greedy heuristic cannot find feasible solution")
            return None

        print('{:<10}{:>10}'.format('Iter', 'Best Obj'))

        for x in range(self.n_iter):

            print('{:<10}{:>10.4f}'.format(x, self.best))

            # terminal condition
            if count_term >= self.n_term:
                break

            # crossover
            par1 = self.pop[np.random.randint(0, self.n_pop, self.n_crossover), :]
            par2 = self.pop[np.random.randint(0, self.n_pop, self.n_crossover), :]
            if self.crossover_type == '1p':
                off1, off2 = self.one_point_crossover(par1, par2)
            if self.crossover_type == '2p':
                off1, off2 = self.two_point_crossover(par1, par2)

            # mutation
            off = np.vstack([off1, off2])
            off = np.where(np.random.rand(self.n_crossover * 2, 1) <= self.p1, self.mutation(off), off)

            # calculate objective values
            off_obj = np.empty(self.n_crossover * 2)
            infeasible = []
            for y in range(self.n_crossover * 2):
                s, obj = self.generate_solution(off[y])
                if obj is None:
                    infeasible.append(y)
                else:
                    off[y] = s
                    off_obj[y] = obj

            # add feasible individuals to population
            off = np.delete(off, infeasible, 0)
            off_obj = np.delete(off_obj, infeasible, 0)
            self.pop = np.vstack([self.pop, off])
            self.pop_obj = np.concatenate([self.pop_obj, off_obj])

            # keep best individuals
            top = np.argsort(self.pop_obj)[:self.n_pop]
            self.pop = self.pop[top]
            self.pop_obj = self.pop_obj[top]

            # update best solution
            if np.amin(self.pop_obj) < self.best:
                self.best = np.amin(self.pop_obj)
                count_term = 0
            else:
                count_term += 1

        return self.best

    def create_initial_population(self):

        for n in range(self.n_pop):

            s = None
            obj = None
            count_feas = 0

            while obj is None:
                if count_feas >= self.n_pop * 50:
                    return False
                s = np.random.randint(0, self.prob.m, self.prob.n)
                s, obj = self.generate_solution(s)
                count_feas += 1

            self.pop[n] = s
            self.pop_obj[n] = obj

        self.best = np.amin(self.pop_obj)

        return True

    def generate_solution(self, s):

        s_new = -np.ones(self.prob.n, dtype=int)
        c = np.zeros(self.prob.n)
        for i in self.prob.a.argsort():
            k = s[i]
            successful = False
            x = 0
            while x < self.prob.m:
                c_max = self.prob.a[i] if s_new[s_new == k].size == 0 \
                    else np.amax(np.maximum(c + self.prob.d, self.prob.a[i])[s_new == k])
                if self.prob.b[i] - c_max >= self.prob.d[i]:
                    c[i] = c_max
                    s_new[i] = k
                    successful = True
                    break
                else:
                    k = k % (self.prob.m - 1) + 1
                    x += 1
            if not successful:
                return s, None

        sum_delay_penalty = np.sum(self.prob.p * (c - self.prob.a))
        x = np.zeros([self.prob.n, self.prob.m])
        x[np.arange(self.prob.n), s_new] = 1
        sum_walking_distance = np.sum(x[:, np.newaxis, :, np.newaxis]
                                      * x[:, np.newaxis, :]
                                      * self.prob.f[:, :, np.newaxis, np.newaxis]
                                      * self.prob.w)

        return s_new, sum_delay_penalty + sum_walking_distance

    def one_point_crossover(self, par1, par2):

        cross = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
        r = np.arange(self.prob.n)
        off1 = np.where(r < cross, par1, par2)
        off2 = np.where(r < cross, par2, par1)
        return off1, off2

    def two_point_crossover(self, par1, par2):

        cross1 = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
        cross2 = np.random.randint(0, self.prob.n, [self.n_crossover, 1])
        r = np.arange(self.prob.n)
        off1 = np.where((r < cross1) & (r > cross2), par1, par2)
        off2 = np.where((r < cross1) & (r > cross2), par2, par1)
        return off1, off2

    def mutation(self, off):

        switch1 = np.random.randint(0, self.prob.n, self.n_crossover * 2)
        switch2 = np.random.randint(0, self.prob.n, self.n_crossover * 2)
        mut = np.copy(off)
        mut[np.arange(len(switch1)), switch1], mut[np.arange(len(switch2)), switch2] \
            = off[np.arange(len(switch2)), switch2], off[np.arange(len(switch1)), switch1]
        return mut

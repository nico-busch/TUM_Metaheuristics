import numpy as np
import timeit

class GeneticAlgorithm:

    def __init__(self, problem, n_iter=None, n_term=None, n_pop=None, n_crossover=None, crossover_type=None, p1=None):

        # parameter
        self.n_iter = n_iter or 10000
        self.n_term = n_term or 500
        self.n_pop = n_pop or 300
        self.n_crossover = n_crossover or 500
        self.crossover_type = crossover_type or '2p'
        self.p1 = p1 or 0.2

        # input
        self.prob = problem

        # population
        self.pop = np.empty([self.n_pop, self.prob.n], dtype=int)
        self.pop_obj = np.empty(self.n_pop)
        self.best = None

    def solve(self):

        z = 0
        if not self.create_initial_population():
            print("No feasible solution achievable via greedy heuristic")
            return None

        for x in range(self.n_iter):

            print("iteration: ", x + 1, "best: ", self.best)

            # terminal condition
            if z >= self.n_term:
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
                s, c = self.generate_solution(off[y])
                if c is None:
                    infeasible.append(y)
                else:
                    off[y] = s
                    off_obj[y] = self.calculate_objective_value(s, c)



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
                z = 0
            else:
                z += 1

        return self.best

    def create_initial_population(self):

        for n in range(self.n_pop):

            s = None
            c = None
            z = 0

            while c is None:
                z += 1
                if z >= 10000:
                    return False
                s = np.random.randint(0, self.prob.m, self.prob.n)
                s, c = self.generate_solution(s)
            obj = self.calculate_objective_value(s, c)

            self.pop[n] = s
            self.pop_obj[n] = obj

        self.best = min(self.pop_obj)

        return True

    def generate_solution(self, s):
        s_new = -np.ones(self.prob.n, dtype=int)
        c = np.zeros(self.prob.n)
        for i in self.prob.a.argsort():
            k = s[i]
            successful = False
            x = 0
            while x < self.prob.m - 1:
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

        return s_new, c

    def calculate_objective_value(self, s, c):

        sum_delay_penalty = np.sum(self.prob.p * (c - self.prob.a))

        x = np.zeros([self.prob.n, self.prob.m])
        x[np.arange(self.prob.n), s] = 1

        sum_walking_distance = np.sum(x.reshape(self.prob.n, 1, self.prob.m, 1)
                                      * x.reshape(1, self.prob.n, 1, self.prob.m)
                                      * self.prob.f.reshape(self.prob.n, self.prob.n, 1, 1)
                                      * self.prob.w.reshape(1, 1, self.prob.m, self.prob.m))

        return sum_delay_penalty + sum_walking_distance

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

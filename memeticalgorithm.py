import numpy as np
import timeit
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm


class MemeticAlgorithm(GeneticAlgorithm):

    def __init__(self,
                 prob,
                 n_iter_ga=10**4,
                 n_term_ga=50,
                 n_pop=300,
                 n_crossover=500,
                 crossover_type='2p',
                 p1=0.2,
                 n_iter_ts=50,
                 n_neigh=10,
                 n_tabu_tenure=10,
                 n_term_ts=5,
                 p2=0.2):

        super().__init__(prob, n_iter_ga, n_term_ga, n_pop, n_crossover, crossover_type, p1)

        # Tabu parameters
        self.n_iter_ts = n_iter_ts
        self.n_neigh = n_neigh
        self.n_tabu_tenure = n_tabu_tenure
        self.n_term_ts = n_term_ts
        self.p2 = p2

    def solve(self):

        print('Beginning Memetic Algorithm')

        start_time = timeit.default_timer()

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

            # Call Tabu Search with probability p2
            for indv in range(off.shape[0]):
                if np.random.random() <= self.p2:
                    ts = TabuSearch(self.prob,
                                    n_iter=self.n_iter_ts,
                                    n_neigh=self.n_neigh,
                                    n_tabu_tenure=self.n_tabu_tenure,
                                    n_term=self.n_term_ts)
                    off[indv], off_c[indv], off_obj[indv] = \
                        ts.solve(off[indv], off_c[indv], off_obj[indv], show_print=False)

            # Carry top individuals to the next generation
            top_idx = np.argsort(off_obj)[:self.n_pop]
            pop = off[top_idx]
            pop_c = off_c[top_idx]
            pop_obj = off_obj[top_idx]

            # Update best solution
            if np.amin(pop_obj) < self.best_obj:
                best_idx = np.argmin(pop_obj)
                self.best = pop[best_idx]
                self.best_c = pop_c[best_idx]
                self.best_obj = pop_obj[best_idx]
                print('{:<10}{:>15.4f}{:>9.0f}{}'.format(x + 1,
                                                         self.best_obj,
                                                         timeit.default_timer() - start_time, 's'))
                count_term = 0
            else:
                count_term += 1

        print('Termination criterion reached')
        print('{}{}'.format('Best objective value is ', self.best_obj))
        print('{}{}'.format('Time is ', timeit.default_timer() - start_time))

        return self.best, self.best_c, self.best_obj

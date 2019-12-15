import pandas as pd
import numpy as np

from metaheuristics.problem import Problem
from metaheuristics.gurobi import Gurobi
from metaheuristics.tabusearch import TabuSearch
from metaheuristics.geneticalgorithm import GeneticAlgorithm
from metaheuristics.memeticalgorithm import MemeticAlgorithm
from metaheuristics.beecolony import BeeColony


instances = [(12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3)]
sizes = ['small'] * 10

results = pd.DataFrame(columns=['Instance', 'Algorithm', 'Size', '#Flights', '#Gates', 'Objective Values', 'Runtimes',
                                'Obj Difference'])
results.set_index(['Instance', 'Algorithm'], inplace=True)

for x in range(1):

    for index, instance in enumerate(instances, 1):

        while True:

            prob = Problem(*instance)

            if sizes[index - 1] != 'large':
                gu = Gurobi(prob)
                gu.solve()
                if gu.solutions.size == 0:
                    continue
                results.loc[(index, 'Gurobi'), :] = [sizes[index - 1], *instance, gu.solutions, gu.runtimes, 0]

            ts = TabuSearch(prob)
            ts.solve()
            if ts.solutions.size == 0:
                continue
            results.loc[(index, 'Tabu Search'), :] = [sizes[index - 1], *instance, ts.solutions, ts.runtimes,
                                                      gu.solutions[-1] - ts.solutions[-1]]

            ga = GeneticAlgorithm(prob)
            ga.solve()
            if ga.solutions.size == 0:
                continue
            results.loc[(index, 'Genetic Algorithm'), :] = [sizes[index - 1], *instance, ga.solutions, ga.runtimes,
                                                            gu.solutions[-1] - ga.solutions[-1]]

            ma = MemeticAlgorithm(prob)
            ma.solve()
            if ma.solutions.size == 0:
                continue
            results.loc[(index, 'Memetic Algorithm'), :] = [sizes[index - 1], *instance, ma.solutions, ma.runtimes,
                                                            gu.solutions[-1] - ma.solutions[-1]]

            bc = BeeColony(prob)
            bc.solve()
            if ma.solutions.size == 0:
                continue
            results.loc[(index, 'Bee Colony'), :] = [sizes[index - 1], *instance, bc.solutions, bc.runtimes,
                                                     gu.solutions[-1] - bc.solutions[-1]]

            break


results.to_csv('robustness_results.csv')

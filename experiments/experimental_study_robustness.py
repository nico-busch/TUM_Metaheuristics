import pandas as pd
import numpy as np

from metaheuristics.problem import Problem
from metaheuristics.gurobi import Gurobi
from metaheuristics.tabusearch import TabuSearch
from metaheuristics.geneticalgorithm import GeneticAlgorithm
from metaheuristics.memeticalgorithm import MemeticAlgorithm
from metaheuristics.beecolony import BeeColony


instances = [(12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3), (12, 3)]
sizes = ['medium'] * 10

results = pd.DataFrame(columns=['Instance', 'Algorithm', 'Size', '#Flights', '#Gates', 'Objective Values', 'Runtimes'])
results.set_index(['Instance', 'Algorithm'], inplace=True)
for x in range(1):

    np.random.seed(x)

    for index, instance in enumerate(instances, 1):

        while True:

            prob = Problem(*instance)

            ts = TabuSearch(prob)
            ts.solve()
            if ts.solutions.size == 0:
                continue
            results.loc[(index, 'Tabu Search'), :] = [sizes[index - 1], *instance, ts.solutions, ts.runtimes]

            ga = GeneticAlgorithm(prob)
            ga.solve()
            if ga.solutions.size == 0:
                continue
            results.loc[(index, 'Genetic Algorithm'), :] = [sizes[index - 1], *instance, ga.solutions, ga.runtimes]

            ma = MemeticAlgorithm(prob)
            ma.solve()
            if ma.solutions.size == 0:
                continue
            results.loc[(index, 'Memetic Algorithm'), :] = [sizes[index - 1], *instance, ma.solutions, ma.runtimes]

            bc = BeeColony(prob)
            bc.solve()
            if ma.solutions.size == 0:
                continue
            results.loc[(index, 'Bee Colony'), :] = [sizes[index - 1], *instance, bc.solutions, bc.runtimes]

            results.to_csv('robustness_results.csv')
            break

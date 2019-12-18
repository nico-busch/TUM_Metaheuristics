import pandas as pd

from metaheuristics.problem import Problem
from metaheuristics.gurobi import Gurobi
from metaheuristics.tabusearch import TabuSearch
from metaheuristics.geneticalgorithm import GeneticAlgorithm
from metaheuristics.memeticalgorithm import MemeticAlgorithm
from metaheuristics.beecolony import BeeColony

'''
    Running this file creates a new experiment. Note that the total runtime of the experiment is around 22 hours.
'''

instances = [(12, 3), (12, 3), (12, 4), (12, 4), (13, 4), (13, 4), (14, 4), (14, 4), (15, 4), (15, 4),
             (20, 5), (20, 5), (22, 5), (22, 5), (24, 6), (24, 6), (26, 6), (26, 6), (28, 6), (28, 6),
             (50, 10), (50, 10), (60, 10), (60, 12), (80, 12), (80, 15), (100, 15), (100, 20)]
sizes = ['small'] * 10 + ['medium'] * 10 + ['large'] * 8

results = pd.DataFrame(columns=['Instance', 'Algorithm', 'Size', '#Flights', '#Gates', 'Objective Values', 'Runtimes'])
results.set_index(['Instance', 'Algorithm'], inplace=True)

for index, instance in enumerate(instances, 1):

    while True:

        prob = Problem(*instance)

        if sizes[index - 1] != 'large':
            gu = Gurobi(prob)
            gu.solve()
            if gu.solutions.size == 0:
                continue
            results.loc[(index, 'Gurobi'), :] = [sizes[index - 1], *instance, gu.solutions, gu.runtimes]

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

        results.to_csv('experimental_results_2.csv')
        break

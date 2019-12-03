import pandas as pd

from problem import Problem
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm
from memeticalgorithm import MemeticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi

instances = [(12, 3), (12, 3), (12, 4), (12, 4), (13, 4), (13, 4), (14, 4), (14, 4), (15, 4), (15, 4),
             (20, 5), (20, 5), (22, 5), (22, 5), (24, 6), (24, 6), (26, 6), (26, 6), (28, 6), (28, 6),
             (50, 10), (50, 10), (60, 10), (60, 12), (80, 12), (80, 15), (100, 15), (100, 20)]
sizes = ['small'] * 10 + ['medium'] * 10 + ['large'] * 8

results = pd.DataFrame(columns=['Instance', 'Algorithm', 'Size', '#Flights', '#Gates', 'Objective Value', 'Runtime'])
results.set_index(['Instance', 'Algorithm'], inplace=True)

for index, instance in enumerate(instances, 1):

    feasible = False
    while not feasible:

        prob = Problem(*instance)

        gu = Gurobi(prob)
        gu.solve()
        if gu.solutions.size == 0:
            continue
        else:
            feasible = True
        results.loc[(index, 'Gurobi'), :] = [sizes[index - 1], *instance, gu.solutions[-1], gu.runtimes[-1]]

        ts = TabuSearch(prob)
        ts.solve()
        results.loc[(index, 'Tabu Search'), :] = [sizes[index - 1], *instance, ts.solutions[-1], ts.runtimes[-1]]

        ga = GeneticAlgorithm(prob)
        ga.solve()
        results.loc[(index, 'Genetic Algorithm'), :] = [sizes[index - 1], *instance, ga.solutions[-1], ga.runtimes[-1]]

        ma = MemeticAlgorithm(prob)
        ma.solve()
        results.loc[(index, 'Memetic Algorithm'), :] = [sizes[index - 1], *instance, ma.solutions[-1], ma.runtimes[-1]]

results.to_csv('results.csv')



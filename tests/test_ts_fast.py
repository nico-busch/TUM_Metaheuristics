import numpy as np

from problem import Problem
from ts_fast import TabuSearchFast
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm
from memeticalgorithm import MemeticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import gantt

np.random.seed(2)

prob = Problem(20, 5)

ts = TabuSearch(prob, n_term=10**3)
ts.solve()
gantt.create_gantt(prob, ts.best, ts.best_c)

ts = TabuSearchFast(prob, n_term=10**4)
ts.solve()
gantt.create_gantt(prob, ts.best, ts.best_c)



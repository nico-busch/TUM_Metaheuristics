import numpy as np

from problem import Problem
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

ga = GeneticAlgorithm(prob)
ga.solve()

ma = MemeticAlgorithm(prob)
ma.solve()

bc = BeeColony(prob)
bc.solve()

gu = Gurobi(prob)
gu.solve()

if ts.best is not None:
    gantt.create_gantt(prob, ts.best, ts.best_c)

if ga.best is not None:
    gantt.create_gantt(prob, ga.best, ga.best_c)

if ma.best is not None:
    gantt.create_gantt(prob, ma.best, ma.best_c)

if bc.best is not None:
    gantt.create_gantt(prob, bc.best, bc.best_c)



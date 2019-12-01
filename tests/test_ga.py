import timeit
import numpy as np

from problem import Problem
from tabu_neu import TabuSearch
from geneticalgorithm import GeneticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import gantt

np.random.seed(1)
prob = Problem(12, 3)

start_time = timeit.default_timer()
ts = TabuSearch(prob)
sol = ts.solve()
print("time: ", timeit.default_timer() - start_time)

# start_time = timeit.default_timer()
# ga = GeneticAlgorithm(prob)
# sol = ga.solve()
# print("time: ", timeit.default_timer() - start_time)
#
# if sol is not None:
#     gantt.create_gantt(prob, ga.best, ga.best_c)
#
# start_time = timeit.default_timer()
# bc = BeeColony(prob)
# sol = bc.solve()
# print("time: ", timeit.default_timer() - start_time)
#
# if sol is not None:
#     print(bc.best_obj)
#     gantt.create_gantt(prob, bc.best, bc.best_c)

start_time = timeit.default_timer()
gu = Gurobi(prob)
sol = gu.solve()
print("time: ", timeit.default_timer() - start_time)


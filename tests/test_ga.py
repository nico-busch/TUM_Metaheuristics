import timeit

from problem import Problem
from geneticalgorithm import GeneticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import gantt


prob = Problem(12, 3)

start_time = timeit.default_timer()
bc = BeeColony(prob)
bc.solve()
print("time: ", timeit.default_timer() - start_time)

if bc.best_obj is not None:
    gantt.create_gantt(prob, bc.best, bc.best_c)

# start_time = timeit.default_timer()
# ga = GeneticAlgorithm(prob)
# ga.solve()
# print("time: ", timeit.default_timer() - start_time)
# #
# if ga.best_obj is not None:
#     gantt.create_gantt(prob, ga.best, ga.best_c)


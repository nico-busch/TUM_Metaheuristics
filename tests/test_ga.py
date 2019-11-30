import timeit

from problem import Problem
from geneticalgorithm import GeneticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import gantt

prob = Problem(20, 5)

start_time = timeit.default_timer()
ga = GeneticAlgorithm(prob)
sol = ga.solve()
print("time: ", timeit.default_timer() - start_time)

if sol is not None:
    gantt.create_gantt(prob, ga.best, ga.best_c)

start_time = timeit.default_timer()
bc = BeeColony(prob)
sol = bc.solve()
print("time: ", timeit.default_timer() - start_time)

if sol is not None:
    print(bc.best_obj)
    gantt.create_gantt(prob, bc.best, bc.best_c)

start_time = timeit.default_timer()
gu = Gurobi(prob)
sol = gu.solve()
print("time: ", timeit.default_timer() - start_time)


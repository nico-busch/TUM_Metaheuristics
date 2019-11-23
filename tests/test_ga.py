import timeit

from problem import Problem
from geneticalgorithm import GeneticAlgorithm
from gurobi import Gurobi

prob = Problem(20, 5)

start_time = timeit.default_timer()
sol = Gurobi(prob)
sol.solve()
print("time: ", timeit.default_timer() - start_time)

start_time = timeit.default_timer()
ga = GeneticAlgorithm(prob)
ga.solve()
print("time: ", timeit.default_timer() - start_time)


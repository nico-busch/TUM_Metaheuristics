import timeit

from problem import Problem
from geneticalgorithm import GeneticAlgorithm

# create problem object

prob = Problem(50, 10)
# create solve object

start_time = timeit.default_timer()
ga = GeneticAlgorithm(prob)
ga.solve()
print("time: ", (timeit.default_timer() - start_time))

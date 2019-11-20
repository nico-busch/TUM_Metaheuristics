import timeit

from problem import Problem
from geneticalgorithm import GeneticAlgorithm

# create problem object

prob = Problem(12, 3)
# create solve object

start_time = timeit.default_timer()
sol = GeneticAlgorithm(prob)
sol.solve()
#sol.generate_solution(1)
print("time: ", (timeit.default_timer() - start_time))

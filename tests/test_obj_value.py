import timeit

from problem import Problem
from tabusearch import TabuSearch

# create problem object
prob = Problem(20, 5)
# create solve object
sol = TabuSearch(prob)


start_time = timeit.default_timer()
print("loop:")
print("value: ", sol.calculate_objective_value_old())
print("time: ", (timeit.default_timer() - start_time))

start_time = timeit.default_timer()
print("vectorized:")
print("value: ", sol.calculate_objective_value())
print("time: ", (timeit.default_timer() - start_time))
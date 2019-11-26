import timeit

from problem import Problem
from tabu_neu import TabuSearch
from gurobi import Gurobi

prob = Problem(12, 3)

start_time = timeit.default_timer()
ts = TabuSearch(prob)
ts.solve()
print("time: ", timeit.default_timer() - start_time)

# start_time = timeit.default_timer()
# sol = Gurobi(prob)
# sol.solve()
# print("time: ", timeit.default_timer() - start_time)


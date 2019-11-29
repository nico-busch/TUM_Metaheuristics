import timeit
from problem import Problem
from tabu_neu import TabuSearch
import numpy as np
#from gurobi import Gurobi

prob = Problem(12, 3)
#print("Problem Data")
#print("a: " + str(prob.a))
#print("b: " + str(prob.b))
#print("d: " + str(prob.d))
#print("p: " + str(prob.p))

start_time = timeit.default_timer()
ts = TabuSearch(prob)
sol = ts.solve()
while(sol == None):
    prob = Problem(12, 3)
    start_time = timeit.default_timer()
    ts = TabuSearch(prob)
    sol = ts.solve()
print("time: ", timeit.default_timer() - start_time)

# start_time = timeit.default_timer()
# sol = Gurobi(prob)
# sol.solve()
# print("time: ", timeit.default_timer() - start_time)


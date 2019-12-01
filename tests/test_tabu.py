import timeit
from problem import Problem
from tabusearch import TabuSearch
import numpy as np
from gurobi import Gurobi

n = 12
m = 3

prob = Problem(n, m)


# Gurobi
print("___________________________________________________________________________________")
print ("Gurobi")
start_time = timeit.default_timer()
sol = Gurobi(prob)
sol.solve()
print("time: ", timeit.default_timer() - start_time)
print("___________________________________________________________________________________")



print("___________________________________________________________________________________")
print("Tabu Search")
start_time = timeit.default_timer()
ts = TabuSearch(prob)
sol = ts.solve()
while (sol == None):
    prob = Problem(n, m)
    start_time = timeit.default_timer()
    ts = TabuSearch(prob)
    sol = ts.solve()
print("time: ", timeit.default_timer() - start_time)
print("___________________________________________________________________________________")
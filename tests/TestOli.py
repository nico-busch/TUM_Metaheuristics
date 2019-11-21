from problem import Problem
import numpy as np
import pandas as pd
from tabusearch import TabuSearch
import tabusearch

# test
test_prob = Problem(6,2)
test_sol = TabuSearch(test_prob)
print("whole schedule: ")
print(test_sol.get_schedule())
suc = test_sol.insert()
print("Successful: + " + str(suc))





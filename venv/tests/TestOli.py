from problem import Problem
import numpy as np
import pandas as pd
from tabusearch import TabuSearch
import tabusearch

# test
test_prob = Problem(4,2)
test_sol = TabuSearch(test_prob)
print(test_sol.x_ik)
tmp1 = pd.DataFrame()
tmp1 = test_sol.get_gate_schedule(1)
print(test_sol.get_gate_schedule(1))
print(test_sol.get_gate_schedule(1)['c'])
tmp = pd.DataFrame()
tmp = test_sol.get_gate_schedule(1)['c']
idx = test_sol.get_gate_schedule(1)['c'].idxmin(axis=0)
print("index: " + str(idx))
idx_pos = tmp.index.get_loc(idx)
print(tmp.index.get_loc(idx))
print(tmp.iloc[idx_pos+1].index[0])
#print("whole schedule: ")
#print(test_sol.get_schedule())
#suc = test_sol.insert()
#print("Successful: " + str(suc))





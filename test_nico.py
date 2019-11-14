import numpy as np
import pandas as pd
import itertools

from problem import Problem
from solve import Solve

# create problem
prob = Problem(50, 10)
# create solve object
sol = Solve(prob)
# find a flight which is assigned to a gate
temp = sol.x_ik.loc[sol.x_ik['x'] == 1].iloc[33].name
# move the starting time of that flight a bit
sol.c_i.loc[temp[0]] += 5
# test shift left
print('before:')
print(sol.get_gate_schedule(temp[1]))
sol.shift_left(temp[0], temp[1])
print('after:')
print(sol.get_gate_schedule(temp[1]))
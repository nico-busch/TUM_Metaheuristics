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
temp = sol.x_ik.loc[sol.x_ik['x'] == 1].iloc[15].name
# test shift right and left
print('initial:')
print(sol.get_gate_schedule(temp[1]))
sol.shift_right(temp[0], temp[1], sol.c_i.loc[temp[0]] + 5)
print('shift right:')
print(sol.get_gate_schedule(temp[1]))
sol.shift_left(temp[0], temp[1])
print('shift left')
print(sol.get_gate_schedule(temp[1]))
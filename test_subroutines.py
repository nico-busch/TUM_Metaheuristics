import numpy as np
import pandas as pd
import itertools

from problem import Problem
from solve import Solve

# create problem object
prob = Problem(50, 10)
# create solve object
sol = Solve(prob)

# find two flights assigned to one gate
temp = sol.x_ik.loc[sol.x_ik['x'] == 1].iloc[15].name
f1 = temp[0]
f2 = sol.get_gate_schedule(temp[1]).iloc[sol.get_gate_schedule(temp[1]).index.get_loc(f1) + 1].name
g = temp[1]

# test subroutine
print(sol.get_gate_schedule(temp[1]))

t = sol.attempt_shift_right(f1, g)
print(t)

sol.shift_right(f1, g, t)
print(sol.get_gate_schedule(temp[1]))

sol.shift_left(f1, g)
print(sol.get_gate_schedule(temp[1]))

t = sol.attempt_shift_interval_right(f1, f2, g)
print(t)

sol.shift_interval(f1, f2, g, t)
print(sol.get_gate_schedule(temp[1]))

c = sol.attempt_shift_interval(f1, f2, g, t)
print(c)

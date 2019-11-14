import numpy as np
import pandas as pd
import itertools

from problem import Problem

test = Problem(50, 10)

temp = test.x_ik.loc[test.x_ik['x'] == 1].iloc[33].name
test.c_i.loc[temp[0]] += 5
print(test.get_gate_schedule(temp[1]))
test.shift_left(temp[0], temp[1])
print(test.get_gate_schedule(temp[1]))
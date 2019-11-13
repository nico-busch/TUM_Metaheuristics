import numpy as np
import pandas as pd
import itertools
from problem_new import Problem

test = Problem(3,
               1,
               [1, 1, 1],
               [4, 4, 4],
               [1, 1, 1],
               [1, 1, 1],
               [1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1])

test.c_i = pd.DataFrame(data={'c': [2, 3, 4], 'i': test.i}).set_index('i')
test.x_ik = pd.DataFrame(data={'x': [1, 1, 1]},
                         index=pd.MultiIndex.from_product([test.i, test.k],
                                                          names=['i', 'k']))

test.create_sched()
print(test.sched)
test.shift_left(1, 1)
print(test.sched)
#print(test.c_i)
#test.shift_right(1, 1, 2)
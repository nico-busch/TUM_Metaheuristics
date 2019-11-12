import numpy as np
from problem import Problem

test = Problem(3,
               1,
               [1, 1, 1],
               [4, 4, 4],
               [1, 1, 1],
               [1, 1, 1],
               [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
               [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

test.x_ik = np.array([[1], [1], [1]])
test.c_i = np.array([1, 2, 4])

test.shift_left(0, 0)
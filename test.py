import random
import numpy as np


import timeit
from problem import Problem


prob = Problem(12, 3)

curr_sol = np.random.randint(0, prob.m, prob.n)
i = np.random.randint(0, prob.n)
k = np.random.choice(np.setdiff1d(range(prob.m), curr_sol[i]))
from problem import Problem
import numpy as np
import pandas as pd
from tabusearch import TabuSearch
import tabusearch
import random


"""
# eigenes Testbeispiel
n = 6
m = 2
i = range(1, n + 1)
k = range(1, m + 1)
a_i = pd.DataFrame(data={'a': [18, 26, 10, 11, 20, 26], 'i': i}).set_index('i')
b_i = pd.DataFrame(data={'b': [26, 35, 20, 20, 28, 35], 'i': i}).set_index('i')
d_i = pd.DataFrame(data={'d': [5, 5, 5, 5, 5, 5], 'i': i}).set_index('i')
f_ij = pd.DataFrame(data={'f': 0},
                         index=pd.MultiIndex.from_product([i, i],
                                                          names=['i', 'j']))
p_i = pd.DataFrame(data={'d': [3, 3, 3, 3, 3, 3], 'i': i}).set_index('i')

print(f_ij)
f_ij.loc[(3,1), 'f'] = 30
f_ij.loc[(3,2), 'f'] = 40
f_ij.loc[(1,2), 'f'] = 10
f_ij.loc[(4,5), 'f'] = 30
f_ij.loc[(5,6), 'f'] = 10
f_ij.loc[(4,6), 'f'] = 40
f_ij.loc[(3,1), 'f'] = 30
f_ij.loc[(3,5), 'f'] = 20
f_ij.loc[(3,6), 'f'] = 20
f_ij.loc[(1,6), 'f'] = 20
f_ij.loc[(4,1), 'f'] = 20
f_ij.loc[(4,2), 'f'] = 20
f_ij.loc[(5,2), 'f'] = 20

print(f_ij)

prob = Problem(n,m,a_i,b_i,d_i, f_ij, p_i)

x_ik = pd.DataFrame(data={'x': 0},
                         index=pd.MultiIndex.from_product([i, k],
                                                          names=['i', 'k']))
c_i = pd.DataFrame(data={'c': [18, 26, 10, 11, 20, 26], 'i': i}).set_index('i')

x_ik.loc[(1,1), 'x'] = 1
x_ik.loc[(2,1), 'x'] = 1
x_ik.loc[(3,1), 'x'] = 1
x_ik.loc[(4,1), 'x'] = 0
x_ik.loc[(5,1), 'x'] = 0
x_ik.loc[(6,1), 'x'] = 0
x_ik.loc[(1,2), 'x'] = 0
x_ik.loc[(2,2), 'x'] = 0
x_ik.loc[(3,2), 'x'] = 0
x_ik.loc[(4,2), 'x'] = 1
x_ik.loc[(5,2), 'x'] = 1
x_ik.loc[(6,2), 'x'] = 1

sol = TabuSearch(prob, c_i, x_ik)
sched = sol.get_schedule()

"""
prob = Problem(12, 3)
sol = TabuSearch(prob)
sched = sol.get_schedule()

print(sched)

print("objective value before: ")
print(sol.calculate_objective_value())

sol.tabu_search()

print("objective value after: ")
print(sol.calculate_objective_value())







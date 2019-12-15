import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from metaheuristics.problem import Problem
from metaheuristics.gurobi import Gurobi
from metaheuristics.tabusearch import TabuSearch
from metaheuristics.geneticalgorithm import GeneticAlgorithm
from metaheuristics.memeticalgorithm import MemeticAlgorithm
from metaheuristics.beecolony import BeeColony


ts_obj_diff = np.arange(10)
ga_obj_diff = np.arange(10)
ma_obj_diff = np.arange(10)
bc_obj_diff = np.arange(10)
x = 0

prob = Problem(12, 4)

while x < 10:

    gu = Gurobi(prob)
    gu.solve()
    if gu.solutions.size == 0:
        continue

    ts = TabuSearch(prob)
    ts.solve()
    if ts.solutions.size == 0:
        continue
    ts_obj_diff[x] = gu.solutions[-1] - ts.solutions[-1]

    ga = GeneticAlgorithm(prob)
    ga.solve()
    if ga.solutions.size == 0:
        continue
    ga_obj_diff[x] = gu.solutions[-1] - ga.solutions[-1]

    ma = MemeticAlgorithm(prob)
    ma.solve()
    if ma.solutions.size == 0:
        continue
    ma_obj_diff[x] = gu.solutions[-1] - ma.solutions[-1]

    bc = BeeColony(prob)
    bc.solve()
    if ma.solutions.size == 0:
        continue
    bc_obj_diff[x] = gu.solutions[-1] - bc.solutions[-1]

    x += 1

    print("TABUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU: " + str(ts_obj_diff))


print(ts_obj_diff)


my_dict = {'Tabu Search': ts_obj_diff, 'Genetic': ga_obj_diff, 'Memetic': ma_obj_diff, 'Bee Colony': bc_obj_diff}

fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.show()


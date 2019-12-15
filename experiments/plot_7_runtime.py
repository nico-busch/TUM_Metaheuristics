import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from metaheuristics.problem import Problem
from metaheuristics.gurobi import Gurobi
from metaheuristics.tabusearch import TabuSearch
from metaheuristics.geneticalgorithm import GeneticAlgorithm
from metaheuristics.memeticalgorithm import MemeticAlgorithm
from metaheuristics.beecolony import BeeColony


ts_run = np.arange(10)
ga_run = np.arange(10)
ma_run = np.arange(10)
ts_sol = np.arange(10)
ga_sol = np.arange(10)
ma_sol = np.arange(10)



x = 0

prob = Problem(50, 10)

while x < 10:

    ts = TabuSearch(prob)
    ts.solve()
    if ts.solutions.size == 0:
        continue
    ts_run[x] = ts.runtimes[-1]
    ts_sol[x] = ts.solutions[-1]

    ga = GeneticAlgorithm(prob)
    ga.solve()
    if ga.solutions.size == 0:
        continue
    ga_run[x] = ga.runtimes[-1]
    ga_sol[x] = ga.solutions[-1]

    ma = MemeticAlgorithm(prob)
    ma.solve()
    if ma.solutions.size == 0:
        continue
    ma_run[x] = ma.runtimes[-1]
    ma_sol[x] = ma.solutions[-1]

    x += 1

my_dict = {'Tabu Search': ts_run, 'Genetic': ga_run, 'Memetic': ma_run}
my_dict2 = {'Tabu Search': ts_sol, 'Genetic': ga_sol, 'Memetic': ma_sol}

pickle_out = open("runtime_variation.pickle", "wb")
pickle.dump(my_dict, pickle_out)
pickle_out.close()

pickle_out_2 = open("objective_variation.pickle", "wb")
pickle.dump(my_dict2, pickle_out_2)
pickle_out_2.close()

pickle_in = open("runtime_variation.pickle", "rb")
example_dict = pickle.load(pickle_in)

fig, ax = plt.subplots()
ax.boxplot(example_dict.values())
ax.set_xticklabels(example_dict.keys())
plt.show()

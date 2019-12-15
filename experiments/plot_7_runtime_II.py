import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pylab as P

from metaheuristics.problem import Problem
from metaheuristics.gurobi import Gurobi
from metaheuristics.tabusearch import TabuSearch
from metaheuristics.geneticalgorithm import GeneticAlgorithm
from metaheuristics.memeticalgorithm import MemeticAlgorithm
from metaheuristics.beecolony import BeeColony


pickle_in = open("runtime_variation.pickle", "rb")
example_dict = pickle.load(pickle_in)

fig, ax = plt.subplots()
ax.boxplot(example_dict.values())
ax.set_xticklabels(example_dict.keys())

names = ['Tabu Search', 'Genetic', 'Memetic']

for i in range(3):
    y = example_dict[names[i]]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'k.', alpha=0.5)


plt.show()
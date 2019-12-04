import numpy as np

from problem import Problem
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm
from memeticalgorithm import MemeticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import gantt
import matplotlib.pyplot as plt



# plot 2
fig, ax = plt.subplots()
ax.scatter(ts.runtimes, ts.solutions/div, color='r', label='Tabu Search', marker='.')
ax.scatter(ga.runtimes, ga.solutions/div, color='y', label='Genetic', marker='^')
ax.scatter(ma.runtimes, ma.solutions/div, color='b', label='Memetic', marker='s')
#ax.scatter(bc.runtimes/div, bc.solutions/div, color='g', label='Bee Colony')
ax.plot(ts.runtimes, ts.solutions/div, color='r')
ax.plot(ga.runtimes, ga.solutions/div, color='y')
ax.plot(ma.runtimes, ma.solutions/div, color='b')
#ax.plot(bc.runtimes/div, bc.solutions/div, color='g')
ax.set_xlabel("time [sec]")
ax.set_ylabel("objective value [K]")
ax.legend()
ax.set_xlim(xmin=0)
plt.show()
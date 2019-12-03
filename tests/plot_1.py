import numpy as np

from problem import Problem
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm
from memeticalgorithm import MemeticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import gantt
import matplotlib.pyplot as plt

# dummy cache run
prob = Problem(5, 2)
ts = TabuSearch(prob)
ts.solve()
ga = GeneticAlgorithm(prob)
ga.solve()

# algorithm execution
np.random.seed(1)

prob = Problem(12, 3)

ts = TabuSearch(prob)
ts.solve()

ga = GeneticAlgorithm(prob)
ga.solve()

#ma = MemeticAlgorithm(prob)
#ma.solve()

#bc = BeeColony(prob)
#bc.solve()

# gu = Gurobi(prob)
# gu.solve()

# plot 1
div = 1000
fig, ax = plt.subplots()
ax.scatter(ts.runtimes, ts.solutions/div, color='r', label='Tabu Search')
ax.scatter(ga.runtimes, ga.solutions/div, color='y', label='Genetic')
ax.scatter(ma.runtimes, ma.solutions/div, color='b', label='Memetic')
#ax.scatter(bc.runtimes/div, bc.solutions/div, color='g', label='Bee Colony')
ax.plot(ts.runtimes, ts.solutions/div, color='r')
ax.plot(ga.runtimes, ga.solutions/div, color='y')
ax.plot(ma.runtimes, ma.solutions/div, color='b')
#ax.plot(bc.runtimes/div, bc.solutions/div, color='g')
min_value = np.concatenate((ts.solutions, ga.solutions, ma.solutions), axis=0)
ax.axhline(y=np.amin(min_value), label='Best Solution', color='k')
ax.set_xlabel("time [sec]")
ax.set_ylabel("objective value [K]")
ax.legend()
ax.set_xlim(xmin=0)
plt.show()
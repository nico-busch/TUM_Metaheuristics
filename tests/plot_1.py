import numpy as np

from problem import Problem
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm
from memeticalgorithm import MemeticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import matplotlib.pyplot as plt

# dummy cache run
prob = Problem(5, 3)
ts = TabuSearch(prob)
ts.solve()
ga = GeneticAlgorithm(prob)
ga.solve()

# 1. SMALL / MEDIUM / LARGE INSTANCE
# algorithm execution
np.random.seed(1)

prob = Problem(50, 10)

ts = TabuSearch(prob)
ts.solve()

ga = GeneticAlgorithm(prob)
ga.solve()

ma = MemeticAlgorithm(prob)
ma.solve()

#bc = BeeColony(prob)
#bc.solve()

# gu = Gurobi(prob)
# gu.solve()

# read data - delete ending point
ts_sol = np.delete(ts.solutions, -1)
ts_run = np.delete(ts.runtimes, -1)
ga_sol = np.delete(ga.solutions, -1)
ga_run = np.delete(ga.runtimes, -1)
ma_sol = np.delete(ma.solutions, -1)
ma_run = np.delete(ma.runtimes, -1)
#bc_sol = np.delete(bc.solutions, -1)
#bc_run = np.delete(bc.runtimes, -1)
#gu_sol = np.delete(gu.solutions, -1)
#gu_run = np.delete(gu.runtimes, -1)

# plot 1 - SMALL
div = 1000
fig, ax = plt.subplots()
ax.scatter(ts_run, ts_sol/div, color='r', label='Tabu Search')
ax.scatter(ga_run, ga_sol/div, color='y', label='Genetic')
ax.scatter(ma_run, ma_sol/div, color='b', label='Memetic')
#ax.scatter(bc_run/div, bc_sol/div, color='g', label='Bee Colony')
ax.plot(ts_run, ts_sol/div, color='r')
ax.plot(ga_run, ga_sol/div, color='y')
ax.plot(ma_run, ma_sol/div, color='b')
#ax.plot(bc_run/div, bc_sol/div, color='g')
min_value = np.concatenate((ts_sol, ga_sol, ma_sol), axis=0)
ax.axhline(y=np.amin(min_value)/div, label='Best Solution', color='k')
ax.set_xlabel("time [sec]")
ax.set_ylabel("objective value [K]")
ax.legend()
ax.set_xlim(xmin=0)
plt.show()
#plt.savefig('plot_1_small')

# 2. MEDIUM INSTANCE

# 3. LARGE INSTANCE

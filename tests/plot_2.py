import numpy as np
import pandas as pd
from problem import Problem
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm
from memeticalgorithm import MemeticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# import data
data = pd.read_csv('results.csv')
print(data)
data = data.set_index(['Algorithm', 'Size'])
print(data)

# read data
ts_run_small = data.loc[('Tabu Search', 'small'), 'Runtime']
ts_sol_small = data.loc[('Tabu Search', 'small'), 'Objective Value']
ga_run_small = data.loc[('Genetic Algorithm', 'small'), 'Runtime']
ga_sol_small = data.loc[('Genetic Algorithm', 'small'), 'Objective Value']
ma_run_small = data.loc[('Memetic Algorithm', 'small'), 'Runtime']
ma_sol_small = data.loc[('Memetic Algorithm', 'small'), 'Objective Value']
bc_run_small = data.loc[('Bee Colony', 'small'), 'Runtime']
bc_sol_small = data.loc[('Bee Colony', 'small'), 'Objective Value']

ts_run_medium = data.loc[('Tabu Search', 'medium'), 'Runtime']
ts_sol_medium = data.loc[('Tabu Search', 'medium'), 'Objective Value']
ga_run_medium = data.loc[('Genetic Algorithm', 'medium'), 'Runtime']
ga_sol_medium = data.loc[('Genetic Algorithm', 'medium'), 'Objective Value']
ma_run_medium = data.loc[('Memetic Algorithm', 'medium'), 'Runtime']
ma_sol_medium = data.loc[('Memetic Algorithm', 'medium'), 'Objective Value']
bc_run_medium = data.loc[('Bee Colony', 'medium'), 'Runtime']
bc_sol_medium = data.loc[('Bee Colony', 'medium'), 'Objective Value']

ts_run_large = data.loc[('Tabu Search', 'large'), 'Runtime']
ts_sol_large = data.loc[('Tabu Search', 'large'), 'Objective Value']
ga_run_large = data.loc[('Genetic Algorithm', 'large'), 'Runtime']
ga_sol_large = data.loc[('Genetic Algorithm', 'large'), 'Objective Value']
ma_run_large = data.loc[('Memetic Algorithm', 'large'), 'Runtime']
ma_sol_large = data.loc[('Memetic Algorithm', 'large'), 'Objective Value']
bc_run_large = data.loc[('Bee Colony', 'large'), 'Runtime']
bc_sol_large = data.loc[('Bee Colony', 'large'), 'Objective Value']

# plot 2
div = 1000

fig, ax = plt.subplots()

ax.scatter(ts_run_small, ts_sol_small/div, color='c', label='Tabu Search, small', marker='o')
ax.scatter(ga_run_small, ga_sol_small/div, color='c', label='Genetic, small', marker='^')
ax.scatter(ma_run_small, ma_sol_small/div, color='c', label='Memetic, small', marker='s')
ax.scatter(bc_run_small, bc_sol_small/div, color='c', label='Bee Colony, small', marker='*')

ax.scatter(ts_run_medium, ts_sol_medium/div, color='g', label='Tabu Search, medium', marker='o')
ax.scatter(ga_run_medium, ga_sol_medium/div, color='g', label='Genetic, medium', marker='^')
ax.scatter(ma_run_medium, ma_sol_medium/div, color='g', label='Memetic, medium', marker='s')
ax.scatter(bc_run_medium, bc_sol_medium/div, color='g', label='Bee Colony, medium', marker='*')

ax.scatter(ts_run_large, ts_sol_large/div, color='b', label='Tabu Search, large', marker='o')
ax.scatter(ga_run_large, ga_sol_large/div, color='b', label='Genetic, large', marker='^')
ax.scatter(ma_run_large, ma_sol_large/div, color='b', label='Memetic, large', marker='s')
ax.scatter(bc_run_large, bc_sol_large/div, color='b', label='Bee Colony, large', marker='*')

ax.set_xlabel("time [sec]")
ax.set_ylabel("objective value [K]")

legend_elements = [Line2D([0], [0], color='c', lw=4, label='Small instance'),
                   Line2D([0], [0], color='g', lw=4, label='Medium instance'),
                   Line2D([0], [0], color='b', lw=4, label='Large instance'),
                   Line2D([0], [0], color='w', marker='o', markerfacecolor='k', label='Tabu Search'),
                   Line2D([0], [0], color='w', marker='^', markerfacecolor='k', label='Genetic'),
                   Line2D([0], [0], color='w', marker='s', markerfacecolor='k', label='Memetic'),
                   Line2D([0], [0], color='w', marker='*', markerfacecolor='k', markersize=11, label='Bee Colony')]
ax.legend(handles=legend_elements)

#ax.grid()
ax.set_xlim(xmin=0)
plt.show()
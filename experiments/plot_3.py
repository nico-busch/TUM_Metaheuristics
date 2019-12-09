import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# import data
data = pd.read_csv('results.csv')
print(data)
data = data.set_index(['Algorithm', 'Size'])
print(data)

# read data
ts_run_small = data.loc[('Tabu Search', 'small'), 'Runtime'].mean()
ga_run_small = data.loc[('Genetic Algorithm', 'small'), 'Runtime'].mean()
ma_run_small = data.loc[('Memetic Algorithm', 'small'), 'Runtime'].mean()
bc_run_small = data.loc[('Bee Colony', 'small'), 'Runtime'].mean()

ts_run_medium = data.loc[('Tabu Search', 'medium'), 'Runtime'].mean()
ga_run_medium = data.loc[('Genetic Algorithm', 'medium'), 'Runtime'].mean()
ma_run_medium = data.loc[('Memetic Algorithm', 'medium'), 'Runtime'].mean()
bc_run_medium = data.loc[('Bee Colony', 'medium'), 'Runtime'].mean()

ts_run_large = data.loc[('Tabu Search', 'large'), 'Runtime'].mean()
ga_run_large = data.loc[('Genetic Algorithm', 'large'), 'Runtime'].mean()
ma_run_large = data.loc[('Memetic Algorithm', 'large'), 'Runtime'].mean()
bc_run_large = data.loc[('Bee Colony', 'large'), 'Runtime'].mean()

# data from paper
paper_cplex_runtime_small = np.average(np.array([24, 28, 782, 367, 871, 438, 1679, 1454, 2110, 1792]))
paper_ts_runtime_small = np.average(np.array([20, 15, 24, 14, 29, 40, 16, 29, 24, 28]))
paper_ma1_runtime_small = np.average(np.array([41, 17, 25, 16, 32, 32, 18, 28, 26, 30]))
paper_ma2_runtime_small = np.average(np.array([41, 18, 25, 17, 34, 36, 18, 28, 26, 31]))
paper_ga1_runtime_small = np.average(np.array([15, 21, 21, 14, 23, 18, 18, 21, 14, 27]))
paper_ga2_runtime_small = np.average(np.array([15, 20, 22, 12, 23, 19, 18, 22, 15, 27]))

paper_cplex_runtime_medium = np.average(np.array([1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800]))
paper_ts_runtime_medium = np.average(np.array([23, 37, 26, 19, 30, 38, 24, 31, 39, 23]))
paper_ma1_runtime_medium = np.average(np.array([30, 41, 27, 22, 36, 33, 36, 36, 39, 37]))
paper_ma2_runtime_medium = np.average(np.array([29, 45, 29, 24, 74, 42, 36, 42, 53, 41]))
paper_ga1_runtime_medium = np.average(np.array([32, 47, 36, 22, 26, 30, 41, 23, 78, 37]))
paper_ga2_runtime_medium = np.average(np.array([32, 45, 38, 23, 47, 27, 38, 28, 87, 43]))

paper_ts_runtime_large = np.average(np.array([98, 86, 117, 84, 188, 115, 163, 162]))
paper_ma1_runtime_large = np.average(np.array([189, 247, 263, 334, 402, 619, 849, 1028]))
paper_ma2_runtime_large = np.average(np.array([221, 208, 341, 379, 439, 431, 539, 762]))
paper_ga1_runtime_large = np.average(np.array([126, 143, 168, 266, 637, 649, 649, 1323]))
paper_ga2_runtime_large = np.average(np.array([147, 141, 294, 227, 568, 863, 1780, 2178]))

# plot 3
fig, ax = plt.subplots()

ax.scatter(ts_run_small, paper_ts_runtime_small, color='c', label='Tabu Search, small', marker='o')
ax.scatter(ga_run_small, paper_ga2_runtime_small, color='c', label='Genetic, small', marker='^')
ax.scatter(ma_run_small, paper_ma2_runtime_small, color='c', label='Memetic, small', marker='s')
# bc is compared to best algorithm of paper TS
ax.scatter(bc_run_small, paper_ts_runtime_small, color='c', label='Bee Colony, small', marker='*')

ax.scatter(ts_run_medium, paper_ts_runtime_medium, color='g', label='Tabu Search, medium', marker='o')
ax.scatter(ga_run_medium, paper_ga2_runtime_medium, color='g', label='Genetic, medium', marker='^')
ax.scatter(ma_run_medium, paper_ma2_runtime_medium, color='g', label='Memetic, medium', marker='s')
# bc is compared to best algorithm of paper TS
ax.scatter(bc_run_medium, paper_ts_runtime_medium, color='g', label='Bee Colony, medium', marker='*')

ax.scatter(ts_run_large, paper_ts_runtime_large, color='b', label='Tabu Search, large', marker='o')
ax.scatter(ga_run_large, paper_ga1_runtime_large, color='b', label='Genetic, large', marker='^')
ax.scatter(ma_run_large, paper_ma2_runtime_large, color='b', label='Memetic, large', marker='s')
# bc is compared to best algorithm of paper TS
ax.scatter(bc_run_large, paper_ts_runtime_large, color='b', label='Bee Colony, large', marker='*')

legend_elements = [Line2D([0], [0], color='c', lw=4, label='Small instance'),
                   Line2D([0], [0], color='g', lw=4, label='Medium instance'),
                   Line2D([0], [0], color='b', lw=4, label='Large instance'),
                   Line2D([0], [0], color='w', marker='o', markerfacecolor='k', label='Tabu Search'),
                   Line2D([0], [0], color='w', marker='^', markerfacecolor='k', label='Genetic'),
                   Line2D([0], [0], color='w', marker='s', markerfacecolor='k', label='Memetic'),
                   Line2D([0], [0], color='w', marker='*', markerfacecolor='k', markersize=11, label='Bee Colony')]
ax.legend(handles=legend_elements)

ax.set_xlabel("Our runtime [sec]")
ax.set_ylabel("Paper runtime [sec]")
plt.axis('square')
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
x = np.linspace(*ax.get_xlim())
ax.plot(x, x, '--k')
plt.show()
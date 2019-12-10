import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull


# import data
data = pd.read_csv('experimental_results.csv',
                   converters={'Objective Values': lambda x: np.fromstring(x[1:-1], sep=' '),
                               'Runtimes': lambda x: np.fromstring(x[1:-1], sep=' ')},
                   index_col=['Instance', 'Algorithm'])

print(data.head())

def tmp(slice):
    return slice[-1]

#ts_run_small = data[np.in1d(data.index.get_level_values(1), ['Tabu Search'])][data['Size']=='small'].loc[:,'Runtimes'].apply(tmp).mean()
#print(ts_run_small)
#exit()

# data from paper
paper_cplex_runtime_small = np.array([24, 28, 782, 367, 871, 438, 1679, 1454, 2110, 1792])
paper_ts_runtime_small = np.array([20, 15, 24, 14, 29, 40, 16, 29, 24, 28])
paper_ma1_runtime_small = np.array([41, 17, 25, 16, 32, 32, 18, 28, 26, 30])
paper_ma2_runtime_small = np.array([41, 18, 25, 17, 34, 36, 18, 28, 26, 31])
paper_ga1_runtime_small = np.array([15, 21, 21, 14, 23, 18, 18, 21, 14, 27])
paper_ga2_runtime_small = np.array([15, 20, 22, 12, 23, 19, 18, 22, 15, 27])

paper_cplex_runtime_medium = np.array([1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800])
paper_ts_runtime_medium = np.array([23, 37, 26, 19, 30, 38, 24, 31, 39, 23])
paper_ma1_runtime_medium = np.array([30, 41, 27, 22, 36, 33, 36, 36, 39, 37])
paper_ma2_runtime_medium = np.array([29, 45, 29, 24, 74, 42, 36, 42, 53, 41])
paper_ga1_runtime_medium = np.array([32, 47, 36, 22, 26, 30, 41, 23, 78, 37])
paper_ga2_runtime_medium = np.array([32, 45, 38, 23, 47, 27, 38, 28, 87, 43])

paper_ts_runtime_large = np.array([98, 86, 117, 84, 188, 115, 163, 162])
paper_ma1_runtime_large = np.array([189, 247, 263, 334, 402, 619, 849, 1028])
paper_ma2_runtime_large = np.array([221, 208, 341, 379, 439, 431, 539, 762])
paper_ga1_runtime_large = np.array([126, 143, 168, 266, 637, 649, 649, 1323])
paper_ga2_runtime_large = np.array([147, 141, 294, 227, 568, 863, 1780, 2178])

# read data
'''
ts_run_small = data[np.in1d(data.index.get_level_values(1), ['Tabu Search'])][data['Size']=='small'].loc[:,'Runtimes'].apply(tmp)
ga_run_small = data[np.in1d(data.index.get_level_values(1), ['Genetic Algorithm'])][data['Size']=='small'].loc[:,'Runtimes'].apply(tmp)
ma_run_small = data[np.in1d(data.index.get_level_values(1), ['Memetic Algorithm'])][data['Size']=='small'].loc[:,'Runtimes'].apply(tmp)
bc_run_small = data[np.in1d(data.index.get_level_values(1), ['Bee Colony'])][data['Size']=='small'].loc[:,'Runtimes'].apply(tmp)

ts_run_medium = data[np.in1d(data.index.get_level_values(1), ['Tabu Search'])][data['Size']=='medium'].loc[:,'Runtimes'].apply(tmp)
ga_run_medium = data[np.in1d(data.index.get_level_values(1), ['Genetic Algorithm'])][data['Size']=='medium'].loc[:,'Runtimes'].apply(tmp)
ma_run_medium = data[np.in1d(data.index.get_level_values(1), ['Memetic Algorithm'])][data['Size']=='medium'].loc[:,'Runtimes'].apply(tmp)
bc_run_medium = data[np.in1d(data.index.get_level_values(1), ['Bee Colony'])][data['Size']=='medium'].loc[:,'Runtimes'].apply(tmp)

ts_run_large = data[np.in1d(data.index.get_level_values(1), ['Tabu Search'])][data['Size']=='large'].loc[:,'Runtimes'].apply(tmp)
ga_run_large = data[np.in1d(data.index.get_level_values(1), ['Genetic Algorithm'])][data['Size']=='large'].loc[:,'Runtimes'].apply(tmp)
ma_run_large = data[np.in1d(data.index.get_level_values(1), ['Memetic Algorithm'])][data['Size']=='large'].loc[:,'Runtimes'].apply(tmp)
bc_run_large = data[np.in1d(data.index.get_level_values(1), ['Bee Colony'])][data['Size']=='large'].loc[:,'Runtimes'].apply(tmp)
'''

# Version two with objective values
ts_run_small = data[np.in1d(data.index.get_level_values(1), ['Tabu Search'])][data['Size']=='small'].loc[:,'Objective Values'].apply(tmp)
ga_run_small = data[np.in1d(data.index.get_level_values(1), ['Genetic Algorithm'])][data['Size']=='small'].loc[:,'Objective Values'].apply(tmp)
ma_run_small = data[np.in1d(data.index.get_level_values(1), ['Memetic Algorithm'])][data['Size']=='small'].loc[:,'Objective Values'].apply(tmp)
bc_run_small = data[np.in1d(data.index.get_level_values(1), ['Bee Colony'])][data['Size']=='small'].loc[:,'Objective Values'].apply(tmp)

ts_run_medium = data[np.in1d(data.index.get_level_values(1), ['Tabu Search'])][data['Size']=='medium'].loc[:,'Objective Values'].apply(tmp)
ga_run_medium = data[np.in1d(data.index.get_level_values(1), ['Genetic Algorithm'])][data['Size']=='medium'].loc[:,'Objective Values'].apply(tmp)
ma_run_medium = data[np.in1d(data.index.get_level_values(1), ['Memetic Algorithm'])][data['Size']=='medium'].loc[:,'Objective Values'].apply(tmp)
bc_run_medium = data[np.in1d(data.index.get_level_values(1), ['Bee Colony'])][data['Size']=='medium'].loc[:,'Objective Values'].apply(tmp)

ts_run_large = data[np.in1d(data.index.get_level_values(1), ['Tabu Search'])][data['Size']=='large'].loc[:,'Objective Values'].apply(tmp)
ga_run_large = data[np.in1d(data.index.get_level_values(1), ['Genetic Algorithm'])][data['Size']=='large'].loc[:,'Objective Values'].apply(tmp)
ma_run_large = data[np.in1d(data.index.get_level_values(1), ['Memetic Algorithm'])][data['Size']=='large'].loc[:,'Objective Values'].apply(tmp)
bc_run_large = data[np.in1d(data.index.get_level_values(1), ['Bee Colony'])][data['Size']=='large'].loc[:,'Objective Values'].apply(tmp)



'''
# plot 3
fig, ax = plt.subplots()

col = {'Gurobi': 'tab:red',
       'Tabu Search': 'tab:green',
       'Genetic Algorithm': 'tab:blue',
       'Memetic Algorithm': 'tab:purple',
       'Bee Colony': 'tab:orange'}

ax.scatter(ts_run_small, paper_ts_runtime_small, color=col['Tabu Search'], label='Tabu Search, small', marker='*')
ax.scatter(ga_run_small, paper_ga2_runtime_small, color=col['Genetic Algorithm'], label='Genetic, small', marker='*')
ax.scatter(ma_run_small, paper_ma2_runtime_small, color=col['Memetic Algorithm'], label='Memetic, small', marker='*')
# bc is compared to best algorithm of paper TS
ax.scatter(bc_run_small, paper_ts_runtime_small, color=col['Bee Colony'], label='Bee Colony, small', marker='*')

ax.scatter(ts_run_medium, paper_ts_runtime_medium, color=col['Tabu Search'], label='Tabu Search, medium', marker='s')
ax.scatter(ga_run_medium, paper_ga2_runtime_medium, color=col['Genetic Algorithm'], label='Genetic, medium', marker='s')
ax.scatter(ma_run_medium, paper_ma2_runtime_medium, color=col['Memetic Algorithm'], label='Memetic, medium', marker='s')
# bc is compared to best algorithm of paper TS
ax.scatter(bc_run_medium, paper_ts_runtime_medium, color=col['Bee Colony'], label='Bee Colony, medium', marker='s')

ax.scatter(ts_run_large, paper_ts_runtime_large, color=col['Tabu Search'], label='Tabu Search, large', marker='^')
ax.scatter(ga_run_large, paper_ga1_runtime_large, color=col['Genetic Algorithm'], label='Genetic, large', marker='^')
ax.scatter(ma_run_large, paper_ma2_runtime_large, color=col['Memetic Algorithm'], label='Memetic, large', marker='^')
# bc is compared to best algorithm of paper TS
ax.scatter(bc_run_large, paper_ts_runtime_large, color=col['Bee Colony'], label='Bee Colony, large', marker='^')

legend_elements = [Line2D([0], [0], color=col['Tabu Search'], lw=4, label='Tabu Search'),
                   Line2D([0], [0], color=col['Genetic Algorithm'], lw=4, label='Genetic'),
                   Line2D([0], [0], color=col['Memetic Algorithm'], lw=4, label='Memetic'),
                   Line2D([0], [0], color=col['Bee Colony'], lw=4, label='Bee Colony'),
                   Line2D([0], [0], color='w', marker='^', markerfacecolor='k', label='Large instance'),
                   Line2D([0], [0], color='w', marker='s', markerfacecolor='k', label='Medium instance'),
                   Line2D([0], [0], color='w', marker='*', markerfacecolor='k', markersize=11, label='Small instance')]
ax.legend(handles=legend_elements)

ax.set_xlabel("Our runtime [sec]")
ax.set_ylabel("Paper runtime [sec]")
plt.axis('square')
ax.set_yscale('log')
ax.set_xscale('log')
x = np.linspace(*ax.get_xlim())
ax.plot(x, x, '--k')
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
plt.show()
'''

fig, axs = plt.subplots(2, 2)

col = {'Gurobi': 'r',
       'Tabu Search': 'y',
       'Genetic Algorithm': 'g',
       'Memetic Algorithm': 'c',
       'Bee Colony': 'b'}

axs[0, 0].set_title('Tabu Search')
axs[0, 0].scatter(ga_run_small, ts_run_small, color=col['Genetic Algorithm'])
axs[0, 0].scatter(ma_run_small, ts_run_small, color=col['Memetic Algorithm'])
axs[0, 0].scatter(bc_run_small, ts_run_small, color=col['Bee Colony'])
axs[0, 0].scatter(ga_run_medium, ts_run_medium, color=col['Genetic Algorithm'])
axs[0, 0].scatter(ma_run_medium, ts_run_medium, color=col['Memetic Algorithm'])
axs[0, 0].scatter(bc_run_medium, ts_run_medium, color=col['Bee Colony'])
axs[0, 0].scatter(ga_run_large, ts_run_large, color=col['Genetic Algorithm'])
axs[0, 0].scatter(ma_run_large, ts_run_large, color=col['Memetic Algorithm'])
axs[0, 0].scatter(bc_run_large, ts_run_large, color=col['Bee Colony'])

axs[0, 1].set_title('Genetic')
axs[0, 1].scatter(ts_run_small, ga_run_small, color=col['Tabu Search'])
axs[0, 1].scatter(ma_run_small, ga_run_small, color=col['Memetic Algorithm'])
axs[0, 1].scatter(bc_run_small, ga_run_small, color=col['Bee Colony'])
axs[0, 1].scatter(ts_run_medium, ga_run_medium, color=col['Tabu Search'])
axs[0, 1].scatter(ma_run_medium, ga_run_medium, color=col['Memetic Algorithm'])
axs[0, 1].scatter(bc_run_medium, ga_run_medium, color=col['Bee Colony'])
axs[0, 1].scatter(ts_run_large, ga_run_large, color=col['Tabu Search'])
axs[0, 1].scatter(ma_run_large, ga_run_large, color=col['Memetic Algorithm'])
axs[0, 1].scatter(bc_run_large, ga_run_large, color=col['Bee Colony'])

axs[1, 0].set_title('Memetic')
axs[1, 0].scatter(ts_run_small, ma_run_small, color=col['Tabu Search'])
axs[1, 0].scatter(ga_run_small, ma_run_small, color=col['Genetic Algorithm'])
axs[1, 0].scatter(bc_run_small, ma_run_small, color=col['Bee Colony'])
axs[1, 0].scatter(ts_run_medium, ma_run_medium, color=col['Tabu Search'])
axs[1, 0].scatter(ga_run_medium, ma_run_medium, color=col['Genetic Algorithm'])
axs[1, 0].scatter(bc_run_medium, ma_run_medium, color=col['Bee Colony'])
axs[1, 0].scatter(ts_run_large, ma_run_large, color=col['Tabu Search'])
axs[1, 0].scatter(ga_run_large, ma_run_large, color=col['Genetic Algorithm'])
axs[1, 0].scatter(bc_run_large, ma_run_large, color=col['Bee Colony'])

axs[1, 1].set_title('Bee Colony')
axs[1, 1].scatter(ts_run_small, bc_run_small, color=col['Tabu Search'])
axs[1, 1].scatter(ma_run_small, bc_run_small, color=col['Memetic Algorithm'])
axs[1, 1].scatter(ga_run_small, bc_run_small, color=col['Genetic Algorithm'])
axs[1, 1].scatter(ts_run_medium, bc_run_medium, color=col['Tabu Search'])
axs[1, 1].scatter(ma_run_medium, bc_run_medium, color=col['Memetic Algorithm'])
axs[1, 1].scatter(ga_run_medium, bc_run_medium, color=col['Genetic Algorithm'])
axs[1, 1].scatter(ts_run_large, bc_run_large, color=col['Tabu Search'])
axs[1, 1].scatter(ma_run_large, bc_run_large, color=col['Memetic Algorithm'])
axs[1, 1].scatter(ga_run_large, bc_run_large, color=col['Genetic Algorithm'])



for i in range(2):
    for j in range(2):
        #axs[i, j].set_yscale('log')
        #axs[i, j].set_xscale('log')
        x = np.linspace(*axs[i, j].get_xlim())
        axs[i, j].plot(x, x, '--k')
        #plt.axis('square')

        #x = np.linspace(*axs[i, j].get_xlim())
        #axs[i, j].plot(axs[i, j].get_xlim(), axs[i, j].get_ylim(), '--k')
        #axs[i, j].set_aspect('equal', adjustable='box')
        #plt.axis('square')

plt.show()

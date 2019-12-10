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
data = data.loc[~data.index.get_level_values(1).isin(['Gurobi'])]

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

legend = set()
col = {'Gurobi': 'tab:red',
       'Tabu Search': 'tab:green',
       'Genetic Algorithm': 'tab:blue',
       'Memetic Algorithm': 'tab:purple',
       'Bee Colony': 'tab:orange'}

algorithms1 = data.groupby('Algorithm')

for (x, algorithm1), ax in zip(algorithms1, axs.flatten()):
    tim1 = [z[-1]/1000 for z in algorithm1['Objective Values']]
    algorithms2 = data.loc[~data.index.get_level_values(1).isin([x])].groupby('Algorithm')
    for (y, algorithm2) in algorithms2:
        tim2 = [z[-1]/1000 for z in algorithm2['Objective Values']]
        ax.scatter(tim2, tim1, color=col[y], label=y if y not in legend else None)
        legend.add(y)
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.set_aspect('equal')
        ax.set_xlim(0, 10 ** 3)
        ax.set_ylim(0, 10 ** 3)
        s = np.linspace(*ax.get_xlim())
        ax.plot(s, s, color='grey', zorder=0)
        ax.set_ylabel(x, labelpad=10)

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(x, []) for x in zip(*lines_labels)]

fig.legend(lines, labels, loc='right')

plt.show()

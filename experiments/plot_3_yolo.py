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

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

legend = set()
col = {'Gurobi': 'r',
       'Tabu Search': 'y',
       'Genetic Algorithm': 'g',
       'Memetic Algorithm': 'c',
       'Bee Colony': 'b'}

algorithms1 = data.groupby('Algorithm')

for (x, algorithm1), ax in zip(algorithms1, axs.flatten()):
    tim1 = [z[-1] for z in algorithm1['Runtimes']]
    algorithms2 = data.loc[~data.index.get_level_values(1).isin([x])].groupby('Algorithm')
    for (y, algorithm2) in algorithms2:
        tim2 = [z[-1] for z in algorithm2['Runtimes']]
        ax.axis('square')
        ax.scatter(tim2, tim1, color=col[y], label=y if y not in legend else None)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(1)
        ax.set_ylim(1)
        s = np.linspace(*ax.get_xlim())
        ax.plot(s, s, color='grey', zorder=0)
        ax.set_ylabel(x)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

plt.show()

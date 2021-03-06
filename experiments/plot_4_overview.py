import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

'''
    This plot shows an overview of all instances and all implemented heuristic.
'''

data = pd.read_csv('experimental_results.csv',
                   converters={'Objective Values': lambda x: np.fromstring(x[1:-1], sep=' '),
                               'Runtimes': lambda x: np.fromstring(x[1:-1], sep=' ')},
                   index_col=['Instance', 'Algorithm'])

fig, ax = plt.subplots()

legend = set()
col = {'Gurobi': 'tab:red',
       'Tabu Search': 'tab:green',
       'Genetic Algorithm': 'tab:blue',
       'Memetic Algorithm': 'tab:purple',
       'Bee Colony': 'tab:orange'}

for x, size in data.groupby('Size'):

    points_x = [z[-1] for z in size['Runtimes']]
    points_y = [z[-1] for z in size['Objective Values']]
    points = np.array(list(zip(points_x, points_y)))
    hull = ConvexHull(points)
    poly = plt.Polygon(points[hull.vertices, :], color='lightgray', zorder=0)
    ax.add_patch(poly)

    for y, algorithm in size.groupby('Algorithm'):

        tim = [z[-1] for z in algorithm['Runtimes']]
        sol = [z[-1] for z in algorithm['Objective Values']]
        ax.scatter(tim, sol, color=col[y], label=y if y not in legend else None)
        legend.add(y)

# logarithmic scale
ax.set_yscale('log')

ax.set_xscale('log')
ax.set_xlabel("time [sec]")
ax.set_ylabel("objective value")
ax.legend()
plt.show()
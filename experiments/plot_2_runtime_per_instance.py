import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
    This plot shows the runtime of each metaheuristic for each instance (number of flights x number of gates)
'''

# import data
data = pd.read_csv('experimental_results.csv',
                   converters={'Objective Values': lambda x: np.fromstring(x[1:-1], sep=' '),
                               'Runtimes': lambda x: np.fromstring(x[1:-1], sep=' ')},
                   index_col=['Instance', 'Algorithm'])

col = {'Gurobi': 'tab:red',
       'Tabu Search': 'tab:green',
       'Genetic Algorithm': 'tab:blue',
       'Memetic Algorithm': 'tab:purple',
       'Bee Colony': 'tab:orange'}

fig, ax = plt.subplots()

for x, algorithm in data.groupby('Algorithm'):
    instances = [str(a) + "x" + str(b) for a, b in zip(algorithm['#Flights'], algorithm['#Gates'])]
    tim = [z[-1] for z in algorithm['Runtimes']]
    ax.plot(instances, tim, color=col[x], marker='x', label=x)

# logarithmic scale
ax.set_yscale('log')

ax.set_xlabel("instance")
ax.set_ylabel("runtime [sec]")
ax.legend()
plt.show()
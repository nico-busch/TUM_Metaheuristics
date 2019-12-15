import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('experimental_results.csv',
                   converters={'Objective Values': lambda x: np.fromstring(x[1:-1], sep=' '),
                               'Runtimes': lambda x: np.fromstring(x[1:-1], sep=' ')},
                   index_col=['Instance', 'Algorithm'])
instance = 21
data = data.loc[instance]

col = {'Gurobi': 'tab:red',
       'Tabu Search': 'tab:green',
       'Genetic Algorithm': 'tab:blue',
       'Memetic Algorithm': 'tab:purple',
       'Bee Colony': 'tab:orange'}

div = 1000

fig, ax = plt.subplots()

for x, algorithm in data.groupby('Algorithm'):
    tim = algorithm['Runtimes'].iloc[0]
    sol = algorithm['Objective Values'].iloc[0]
    ax.plot(tim, sol / div, color=col[x], marker='o', label=x)

#ax.set_xscale('log')
ax.set_xlabel("time [sec]")
ax.set_ylabel("objective value [k]")
ax.legend()
plt.show()
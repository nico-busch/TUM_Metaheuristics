import numpy as np
import pandas as pd
import seaborn as sns
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
ts_run = data.loc[('Tabu Search', ), 'Runtime'].to_numpy()
ga_run = data.loc[('Genetic Algorithm', ), 'Runtime'].to_numpy()
ma_run = data.loc[('Memetic Algorithm', ), 'Runtime'].to_numpy()
bc_run = data.loc[('Bee Colony', ), 'Runtime'].to_numpy()

# data from paper
paper_ts_runtime = np.average(np.array([20, 15, 23, 37, 98, 96]))
paper_ma2_runtime = np.average(np.array([41, 18, 30, 41, 189, 247]))
paper_ga2_runtime = np.average(np.array([15, 20, 32, 45, 147, 141]))

# calculate the difference
ts_diff = paper_ts_runtime - ts_run
ga_diff = paper_ga2_runtime - ga_run
ma_diff = paper_ma2_runtime - ma_run
bc_diff = paper_ts_runtime - bc_run

# create representative dataframe
df = pd.DataFrame()
df['Tabu Search'] = ts_diff
df['Genetic'] = ga_diff
df['Memetic'] = ma_diff
df['Bee Colony'] = bc_diff

df = df.melt(var_name='Algorithm', value_name='Runtime difference (paper runtime - our runtime)')

# plot 3 - alternative
ax = sns.violinplot(x="Algorithm", y="Runtime difference (paper runtime - our runtime)", data=df)
plt.show()


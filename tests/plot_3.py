import numpy as np

from problem import Problem
from tabusearch import TabuSearch
from geneticalgorithm import GeneticAlgorithm
from memeticalgorithm import MemeticAlgorithm
from beecolony import BeeColony
from gurobi import Gurobi
import gantt
import matplotlib.pyplot as plt

# plot 3
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

numpy.concatenate((ts.solutions, ga.solutions, ma.solutions), axis=0)

fig, ax = plt.subplots()
ax.set_xlabel("Our runtime [sec]")
ax.set_ylabel("Paper runtime [sec]")
ax.scatter(np.amin(ts.runtimes), paper_ts_runtime_small, color='r', label='Tabu Search', marker='.')
ax.scatter(np.amin(ga.runtimes), paper_ga2_runtime_small, color='y', label='Genetic', marker='^')
ax.scatter(np.amin(ma.runtimes), paper_ma2_runtime_small, color='b', label='Memetic', marker='s')
ax.legend()
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=0)
plt.plot(x, x + 0, '--k')
plt.show()
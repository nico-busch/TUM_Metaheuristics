from problem import Problem
import numpy as np

# general parameter
max_num_iter = 10**6
num_neigh = 100
tabu_tenure = 10
max_num_without_impr = 10**4

def solveTabuSearch(Problem):

    # initial solution
    x_init = Problem.x_ik.copy()
    # neighborhood storage
    X_neigh = np.arange(num_neigh*Problem.m*Problem.n).reshape(num_neigh, Problem.m, Problem.n)
    # short term storage
    X_tabu_tenure = np.arange(tabu_tenure*Problem.m*Problem.n).reshape(tabu_tenure, Problem.m, Problem.n)

    # create neighborhood



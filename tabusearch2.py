from problem import Problem
import numpy as np
import pandas as pd
import random
from tabusearch import TabuSearch


def tabu_search(s):
    start = pd.DataFrame()
    start = TabuSearch.x_ik
    max_num_iter = 10**6
    neigh = 100
    tabu_tenure = 10
    max_num_without_impr = 10**4
    # create A illustration
    A = pd.DataFrame(data={'k': 0},
                                 index=pd.MultiIndex.from_product([s.prob.i],
                                                                  names=['i']))
    create_A(A,s.x_ik)

    # STEPS
    # 1. generate set of neighborhood solutions (insert = true / interval exchange move = false)
    method = bool(random.getrandbits(1))
    N = pd.DataFrame(data={'A': 0},
                                 index=pd.MultiIndex.from_product([s.prob.i],
                                                                  names=['N']))
    count = 1
    while(count <= 100):
        if(method == True):
            flight = random.randint(1, s.prob.n)
            #todo OLI: implement attempt insert
            if(s.attemptInsert(flight, A[flight])):
                N.loc[1,'A'] = s.insert(flight, A[flight])
                count += 1
        else:
            flight_i = random.randint(1, s.prob.n)
            flight_j = random.randint(1, s.prob.n)
            # todo OLI: implement attempt insert
            if (s.attemptIntervalExchange(flight_i, flight_j, A[flight_i])):
                N.loc[1, 'A'] = s.intervalExchange(flight_i, flight_j, A[flight_i])
                count += 1
    # 2. choose best solution (two criterion)
    #todo OLI: write objective calculation with input A
    #todo OLI: apply function on all rows of N (lambda)
    #todo OLI: select best value

    # 3. tests if termination condition applies
    return "d"

# todo OLI: make more efficient if possible
def create_A(A,x):
    tmp = pd.DataFrame()
    tmp = x[x['x'] == 1]
    for idx, row in tmp.iterrows():
        A.loc[(idx[0], 'k')] = idx[1]
    return A

# old stuff
def create_A_old(x, s):
    tmp = pd.DataFrame()
    tmp = x[x['x'] == 1]
    print(tmp)
    A = pd.DataFrame(data={'k': 0},
                                 index=pd.MultiIndex.from_product([s.prob.i],
                                                                  names=['i']))
    for idx, row in tmp.iterrows():
        A.loc[(idx[0], 'k')] = idx[1]
    return A


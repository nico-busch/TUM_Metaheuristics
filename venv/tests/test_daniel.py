import numpy as np
import pandas as pd
import itertools
from gurobipy import *

from problem import Problem
from gurobi import Gurobi

# create problem object
prob = Problem(12, 3)
# create solve object
gur = Gurobi(prob)

print(gur.solve())



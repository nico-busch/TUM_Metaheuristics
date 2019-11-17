from problem import Problem
import numpy as np
from solve import Solve

tmp = Problem(4,2)
print(tmp.w_kl)
print(tmp.f_ij)
tmp_sol = Solve(tmp)
print(tmp_sol.x_ik)
print(tmp_sol.calculate_objective_value())




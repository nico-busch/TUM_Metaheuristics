from problem import Problem
import numpy as np
import pandas as pd
from solve import Solve

tmp = Problem(4,2)
df_temp = pd.DataFrame()
df_temp = pd.concat([tmp.a_i,tmp.b_i, tmp.d_i], axis=1)
print(df_temp)
tmp_sol = Solve(tmp)
print(tmp_sol.x_ik)
df_temp = pd.concat([tmp.a_i,tmp.b_i, tmp.d_i, tmp_sol.c_i], axis=1)
print(df_temp)
print(tmp_sol.calculate_objective_value())
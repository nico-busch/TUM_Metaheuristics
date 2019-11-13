from problem import Problem
from problem import Solution
import numpy as np

tmp = Problem(4, 2, 0, 0, 0, 0, 0, 0)
tmp.create_data()
sol = Solution(tmp.x_ik, tmp.c_i)

print("delay time value: ")
print((tmp.c_i - tmp.a) * tmp.p)
print("w_kl: ")
print(tmp.w)
print("f_ij: ")
print(tmp.f)
print("x_ik: ")
print(tmp.x_ik)
print("Solution: ")
print(sol.calculateObjectiveValue(tmp))


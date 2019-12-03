from gurobipy import *
import numpy as np


class Gurobi:

    def __init__(self, prob):
        self.prob = prob
        self.M = np.amax(prob.b)

    def solve(self):
        model = Model()

        # Creation of decision variables
        x_ik = {}
        for i in range(self.prob.n):
            for k in range(self.prob.m):
                x_ik[i, k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'x_i[{i},{k}]')

        y_ij = {}
        for i in range(self.prob.n):
            for j in range(self.prob.n):
                y_ij[i, j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'y_i[{i},{j}]')

        z_ijkl = {}
        for i in range(self.prob.n):
            for j in range(self.prob.n):
                for k in range(self.prob.m):
                    for l in range(self.prob.m):
                        z_ijkl[i, j, k, l] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                                          name=f'z_ijkl[{i},{j},{k},{l}]')

        c_i = {}
        for i in range(self.prob.n):
            c_i[i] = model.addVar(lb=0.000001, vtype=GRB.CONTINUOUS, name=f'c_i[{i}]')

        model.update()

        objective = quicksum(self.prob.f[i, j] * self.prob.w[k, l] * z_ijkl[i, j, k, l]
                             for i in range(self.prob.n)
                             for j in range(self.prob.n)
                             for k in range(self.prob.m)
                             for l in range(self.prob.m)) + \
                    quicksum(self.prob.p[i] * (c_i[i] - self.prob.a[i])
                             for i in range(self.prob.n))
        # minimize objective function
        model.setObjective(objective, GRB.MINIMIZE)

        # constraints

        # c1
        model.addConstrs(quicksum(x_ik[i, k] for k in range(self.prob.m)) == 1 for i in range(self.prob.n))

        # c2
        model.addConstrs(z_ijkl[i, j, k, l] <= x_ik[i, k]
                         for i in range(self.prob.n)
                         for j in range(self.prob.n)
                         for k in range(self.prob.m)
                         for l in range(self.prob.m))

        # c3
        model.addConstrs(z_ijkl[i, j, k, l] <= x_ik[j, l]
                         for i in range(self.prob.n)
                         for j in range(self.prob.n)
                         for k in range(self.prob.m)
                         for l in range(self.prob.m))

        # c4
        model.addConstrs(x_ik[i, k] + x_ik[j, l] - 1 <= z_ijkl[i, j, k, l]
                         for i in range(self.prob.n)
                         for j in range(self.prob.n)
                         for k in range(self.prob.m)
                         for l in range(self.prob.m))

        # c5
        model.addConstrs(c_i[i] >= self.prob.a[i] for i in range(self.prob.n))

        # c6
        model.addConstrs(c_i[i] <= self.prob.b[i] - self.prob.d[i] for i in range(self.prob.n))

        # c7 - BIG M 1
        model.addConstrs((c_i[i] + self.prob.d[i]) - c_i[j] + y_ij[i, j] * self.M >= 0
                         for i in range(self.prob.n)
                         for j in range(self.prob.n))

        # c8 - BIG M 2
        model.addConstrs((c_i[i] + self.prob.d[i]) - c_i[j] - (1 - y_ij[i, j]) * self.M <= 0
                         for i in range(self.prob.n)
                         for j in range(self.prob.n))

        # c9
        model.addConstrs(y_ij[i, j] + y_ij[j, i] >= z_ijkl[i, j, k, k]
                         for k in range(self.prob.m)
                         for i in range(self.prob.n)
                         for j in range(self.prob.n)
                         if i != j)


        def mycallback(model, where):
            if where == GRB.Callback.MIPSOL:
                print(model.cbGet(GRB.Callback.MIPSOL_OBJ))
                print(round(model.cbGet(GRB.Callback.RUNTIME)))

        #model._vars = model.getVars()
        model.update()
        model.setParam('TimeLimit', 1800)
        model.optimize(mycallback)
        # objective_value = model.objVal
        # run_time = model.Runtime
        # print("Objective Value: ", round(objective_value))
        # print("Runtime: ", round(run_time))
        # model.computeIIS()

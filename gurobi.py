from gurobipy import *


class Gurobi:

    def __init__(self, prob):
        self.prob = prob
        self.M = prob.b_i.max()['b']

    def solve(self):
        model = Model()
        # model.Params.M = self.M

        # Creation of decision variables
        x_ik = {}
        for i in self.prob.i:
            for k in self.prob.k:
                x_ik[i, k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'x_i[{i},{k}]')

        y_ij = {}
        for i in self.prob.i:
            for j in self.prob.i:
                y_ij[i, j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'y_i[{i},{j}]')

        z_ijkl = {}
        for i in self.prob.i:
            for j in self.prob.i:
                for k in self.prob.k:
                    for l in self.prob.k:
                        z_ijkl[i, j, k, l] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY,
                                                          name=f'z_ijkl[{i},{j},{k},{l}]')

        c_i = {}
        for i in self.prob.i:
            c_i[i] = model.addVar(lb=0.000001, vtype=GRB.CONTINUOUS, name=f'c_i[{i}]')

        model.update()

        # objective
        objective = LinExpr()

        objective = quicksum(
            self.prob.f_ij.loc[(i, j), 'f'] * self.prob.w_kl.loc[(k, l), 'w'] * z_ijkl[i, j, k, l] for i in self.prob.i
            for j in self.prob.i for k in self.prob.k for l in self.prob.k) + quicksum(
            self.prob.p_i.loc[i, 'p'] * (c_i[i] - self.prob.a_i.loc[i, 'a']) for i in self.prob.i)
        # minimize objective function
        model.setObjective(objective, GRB.MINIMIZE)

        # constraints

        # c1

        model.addConstrs(quicksum(x_ik[i, k] for k in self.prob.k) == 1 for i in self.prob.i)

        # c2

        model.addConstrs(
            z_ijkl[i, j, k, l] <= x_ik[i, k] for i in self.prob.i for j in self.prob.i for k in self.prob.k for l in
            self.prob.k)

        # c3

        model.addConstrs(
            z_ijkl[i, j, k, l] <= x_ik[j, l] for i in self.prob.i for j in self.prob.i for k in self.prob.k for l in
            self.prob.k)

        # c4

        model.addConstrs(
            x_ik[i, k] + x_ik[j, l] - 1 <= z_ijkl[i, j, k, l] for i in self.prob.i for j in self.prob.i for k in self.prob.k
            for l in self.prob.k)

        # c5

        model.addConstrs(c_i[i] >= self.prob.a_i.loc[i, 'a'] for i in self.prob.i)

        # c6

        model.addConstrs(c_i[i] <= self.prob.b_i.loc[i, 'b'] - self.prob.d_i.loc[i, 'd'] for i in self.prob.i)

        # c7 - BIG M 1
        model.addConstrs(
            (c_i[i] + self.prob.d_i.loc[i, 'd']) - c_i[j] + y_ij[i, j] * self.M >= 0 for i in self.prob.i for j in
            self.prob.i)

        # c8 - BIG M 2

        model.addConstrs(
            (c_i[i] + self.prob.d_i.loc[i, 'd']) - c_i[j] - (1 - y_ij[i, j]) * self.M <= 0 for i in self.prob.i for j in
            self.prob.i)

        # c9

        model.addConstrs(
            y_ij[i, j] + y_ij[j, i] >= z_ijkl[i, j, k, k] for k in self.prob.k for i in self.prob.i for j in self.prob.i if
            i != j)

        model.update()
        model.setParam('TimeLimit', 1800)
        model.optimize()
        objective_value = model.objVal
        run_time = model.Runtime
        print("Objective Value: ", round(objective_value))
        print("Runtime: ", round(run_time))

        # model.computeIIS()

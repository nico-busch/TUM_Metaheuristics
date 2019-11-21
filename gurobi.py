from gurobipy import *


class Gurobi:

    def __init__(self, prob):

        self.prob = prob
        self.M = prob.b_i.max()['b']
        print(self.M)

    def solve(self):

        model = Model()
        #model.Params.M = self.M

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
        for i in self.prob.i:
            model.addConstr(quicksum(x_ik[i, k] for k in self.prob.k), GRB.EQUAL, 1)

        # c2
        for i in self.prob.i:
            for j in self.prob.i:
                for k in self.prob.k:
                    for l in self.prob.k:
                        model.addConstr(z_ijkl[i, j, k, l] <= x_ik[i, k])

        # c3
        for i in self.prob.i:
            for j in self.prob.i:
                for k in self.prob.k:
                    for l in self.prob.k:
                        model.addConstr(z_ijkl[i, j, k, l] <= x_ik[j, l])

        # c4
        for i in self.prob.i:
            for j in self.prob.i:
                for k in self.prob.k:
                    for l in self.prob.k:
                        model.addConstr(x_ik[i, k] + x_ik[j, l] - 1 <= z_ijkl[i, j, k, l])
        # c5
        for i in self.prob.i:
            model.addConstr(c_i[i] >= self.prob.a_i.loc[i, 'a'])

        # c6
        for i in self.prob.i:
            model.addConstr(c_i[i] <= self.prob.b_i.loc[i, 'b'] - self.prob.d_i.loc[i, 'd'])

        # c7 - BIG M 1
        for i in self.prob.i:
            for j in self.prob.i:
                #if i != j:
                model.addConstr((c_i[i] + self.prob.d_i.loc[i, 'd']) - c_i[j] + y_ij[i, j] * self.M >= 0)

        # c8 - BIG M 2
        for i in self.prob.i:
            for j in self.prob.i:
                #if i != j:
                model.addConstr((c_i[i] + self.prob.d_i.loc[i, 'd']) - c_i[j] - (1 - y_ij[i, j]) * self.M <= 0)

        # c9
        for k in self.prob.k:
            for i in self.prob.i:
                for j in self.prob.i:
                    if i != j:
                        model.addConstr(y_ij[i, j] + y_ij[j, i] >= z_ijkl[i, j, k, k])

        model.update()
        model.optimize()

        #model.computeIIS()

import numpy as np


class TabuSearch:

    def __init__(self, prob):

        # parameters
        self.n_iter = 10**6
        self.n_neigh = 100
        self.n_tabu_tenure = 10
        self.n_term = 10**4

        # input
        self.prob = prob

        # search arrays
        self.neigh = np.empty([self.n_neigh, self.prob.n], dtype=int)
        self.tabu_tenure = np.empty([self.n_tabu_tenure, self.prob.n], dtype=int)

        # output
        self.best = np.empty(self.prob.n, dtype=int)
        self.best_obj = None
        self.best_c = np.empty(self.prob.n, dtype=float)

    def solve(self):

        # create initial solution
        # self.best creation

        # local best solution
        curr_sol = self.best
        curr_obj_val = self.best_obj
        working_sol = self.best

        # initialize tabu tenure
        for i in self.tabu_tenure:
            i = curr_sol

        # Generate set of neighborhood solutions
        count = 0
        count_iter = 0
        count_unchanged = 0

        while (count_iter < self.n_iter and count_unchanged < self.n_term):

            # update forbidden solutions
            self.tabu_tenure[count_iter % self.n_tabu_tenure] = curr_sol

            # generate neighbours
            while (count < self.n_neigh):
                # choose method randomly (insert = true / interval exchange move = false)
                method = bool(random.getrandbits(1))
                if (method == True):
                    working_sol = self.insert()
                    if (self.insert()):
                        self.neigh[count, :] =
                        N[count] = Solution(self.x_ik, self.c_i, self.prob.i)
                        count += 1
                else:
                    if (self.interval_exchange()):
                        N[count] = Solution(self.x_ik, self.c_i, self.prob.i)
                        count += 1
            count = 0

            # calculate all objective values of neighbours
            # todo vectorize
            for i in range(neigh):
                Obj_val.loc[i + 1, 'O'] = N[i].calculate_objective_value(self.prob)

            # Choose best solution
            idx_best = Obj_val['O'].idxmin(axis=0)

            # Check if best solution is forbidden
            # todo vectorize
            forbidden = False
            for i in range(tabu_tenure):
                if (N[idx_best[0] - 1].A.equals(Tabu[i].A)):
                    forbidden = True
                    break

                # Check if better than current solution or not forbidden
                if (Obj_val.loc[idx_best, 'O'] < curr_obj_val or forbidden == False):
                    curr_sol = Solution(N[idx_best[0] - 1].x_ik, N[idx_best[0] - 1].c_i, self.prob.i)
                    curr_obj_val = Obj_val.loc[idx_best, 'O']
                    if (curr_obj_val < self.sol_obj_val):
                        self.sol_obj_val = curr_obj_val
                        self.sol = curr_sol
                        count_unchanged = 0
                    else:
                        count_unchanged += 1
                else:
                    count_unchanged += 1

            count_iter += 1
            print(self.sol_obj_val)

        # save the final solution
        self.sol = curr_sol
        self.sol_obj_val = curr_obj_val
        print(self.sol_obj_val)






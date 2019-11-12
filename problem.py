import numpy as np


class Problem:

    def __init__(self,
                 n,
                 m,
                 a_i,
                 b_i,
                 d_i,
                 p_i,
                 w_kl,
                 f_ij):
        self.n = n
        self.m = m
        self.a_i = a_i
        self.b_i = b_i
        self.d_i = d_i
        self.p_i = p_i
        self.w_kl = w_kl
        self.f_ij = f_ij

        self.x_ik = []
        self.c_i = []
        self.y_ij = []
        self.z_ijkl = []



    def create_data(self):
        return 'hello world'

    def shift_left(self, k, i):
        foo = [x for _, x in sorted(zip(self.x_ik[i:, k], self.c_i[i:]))]
        for x in range(len(self.x_ik[i:, k])):
            if i + x == 0:
                self.c_i[i + x] = self.a_i[i + x]
            else:
                self.c_i[i + x] = max(self.a_i[i + x], self.c_i[i + x - 1] + self.d_i[i + x - 1])
        print(foo)

    def shift_right(self, k , i, t):
        temp = self.c_i
        violation = False
        for x in range(len(self.x_ik[i:, k])):
            if i + x == 0 & t + self.d_i[i + x] <= self.b_i[i + x]:
                temp[i + x] = t
            elif temp[i + x - 1] + self.d_i[i + x - 1] + self.d_i[i + x]:
                temp[i + x] = max(temp[i + x - 1], temp[i + x - 1] + self.d_i[i + x - 1])
            else:
                violation = True
        if not violation:
            self.c_i = temp

    def attempt_shift_right(self):
        return 'hello world'

    def shift_interval(self):
        return 'hello world'

    def attempt_shift_interval(self):
        return 'hello world'

    def attempt_shift_interval_right(self):
        return 'hello world'

    def insert(self):
        return 'hello world'

    def tabu_search(self):
        return 'hello world'

    def genetic_algorithm(self):
        return 'hello world'

    def memetic_algorithm(self):
        return 'hello world'

    def solve(self):
        return 'hello world'


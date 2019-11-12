import numpy

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

    def __init__(self,
                 n,
                 m):

        self.n = n
        self.m = m

    def create_data(self):

        # distance matrix
        self.w_kl = list()
        for k in range(self.m):
            self.w_kl[k] = list()
            for l in range(self.m):
                if (k % 2 == 0) == (l % 2 == 0):
                    self.w_kl[k][l] = math.sqrt((k - l) ** 2)
                else:
                    self.w_kl[k][l] = k - 2 + 3 + l - 1

        print(self.w_kl)


# subroutines

    def shift_left(self, gate, flight):
        return 'hello world'

    def shift_right(self):
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



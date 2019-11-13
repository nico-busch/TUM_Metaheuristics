import numpy as np
import math

class Problem:

    # todo recode init function with only n and m as input parameter and calling of create data function
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

    # creates data with regards to the number of flights n and number of gates m
    def create_data(self):
        # parameter
        self.a_i = np.empty([self.n])
        self.b_i = np.empty([self.n])
        self.d_i = np.empty([self.n])
        self.p_i = np.empty([self.n])
        self.f_ij = np.empty([self.n, self.n])
        des = 0.8
        # setting of random values
        for i in range(self.n):
            self.a_i[i] = np.random.uniform(1, self.n*70/self.m)
            self.b_i[i] = self.a_i[i] + np.random.uniform(45,74)
            self.d_i[i] = des*(self.b_i[i] - self.a_i[i])
            self.p_i[i] = np.random.uniform(10,14)
            for j in range(self.n):
                if (self.a_i[i] < self.a_i[j]):
                    self.f_ij[i][j] = np.random.random_integers(6, 60)
                else:
                    self.f_ij[i][j] = 0
        # gate distances
        self.createDistanceMatrix()
        # print(self.a_i) # Todo delete if not necessary anymore
        if (self.createInitialSolution()==False):
            # No feasibility secured. Try to another random data set again
            self.create_data()
        else:
            # todo delete if not necessary anymore
            """
            print("x_ik: ")
            print(self.x_ik) #todo delete if not necessary anymore
            print("c_i: ")
            print(self.c_i)
            print("d_i: ")
            print(self.d_i)
            print("c_i + d_i: ")
            print(self.d_i+self.c_i)
            print("b_i: ")
            print(self.b_i)
            """

    # distance matrix
    def createDistanceMatrix(self):
        self.w_kl = np.empty([self.m, self.m])
        for k in range(self.m):
            kk = k + 1
            for l in range(self.m):
                ll = l+1
                if (kk % 2 == 0) == (ll % 2 == 0):
                    self.w_kl[k][l] = math.sqrt(((kk - ll)*0.5) ** 2)
                elif (kk%2==0):
                    self.w_kl[k][l] = 3 + math.sqrt(((kk - 2)*0.5) ** 2) + math.sqrt(((ll - 1)*0.5) ** 2)
                elif (ll%2==0):
                    self.w_kl[k][l] = 3 + math.sqrt(((ll - 2)*0.5) ** 2) + math.sqrt(((kk - 1)*0.5) ** 2)
                else:
                    self.w_kl[k][l] = math.sqrt(((kk - ll) * 0.5) ** 2)
        # print(self.w_kl) Todo delete if not necessary anymore

    # creates a feasible initial solution
    def createInitialSolution(self):
        # Todo adapt to Nico changes
        #flights_ascending = np.empty([self.n])
        flights_ascending = np.arange(self.n)
        #self.x_ik = np.empty([self.m, self.n])
        self.x_ik = np.arange(self.n*self.m).reshape(self.m,self.n)
        a_i_copy = np.empty([self.n])
        # sorting flights ascending regarding a_i
        for i in range(self.n):
            a_i_copy[i] = self.a_i[i]
        for i in range(self.n):
            low = 0
            for j in range(self.n):
                if(a_i_copy[j] < a_i_copy[low]):
                    low = j
            flights_ascending[i] = low
            a_i_copy[low] = self.n*70 # is higher than every possible a_i number
        # filling into the x_ik-matrix
        count = 0
        for i in range(self.n):
            for k in range(self.m):
                if(count < self.n):
                    self.x_ik[k][i] = flights_ascending[count]
                    count += 1
                else:
                    self.x_ik[k][i] = -1 # values of x_ik are only valid if not -1
        # setting c_i and checking if solution creation is possible
        self.c_i = np.empty([self.n])
        for i in range(self.n):
            self.c_i[i] = -1 # initializing all values of c_i with -1
        for i in range(self.n):
            for k in range(self.m):
                if(i==0):
                    self.c_i[self.x_ik[k][0]] = self.a_i[self.x_ik[k][0]]
                elif(self.x_ik[k][i] < 0):
                    # nothing should happen as there is no flight number at this point
                    1
                elif( ( self.c_i[self.x_ik[k][i-1]] + self.d_i[self.x_ik[k][i-1]] ) <=
                      ( self.b_i[self.x_ik[k][i]] - self.d_i[self.x_ik[k][i]] ) ):
                    if(( self.c_i[self.x_ik[k][i-1]] + self.d_i[self.x_ik[k][i-1]] ) > self.a_i[self.x_ik[k][i]]):
                        self.c_i[self.x_ik[k][i]] = self.c_i[self.x_ik[k][i-1]] + self.d_i[self.x_ik[k][i-1]]
                    else:
                        self.c_i[self.x_ik[k][i]] = self.a_i[self.x_ik[k][i]]
                else:
                    # no initial solution could be found (feasibility not secured)
                    return False
        return True

    def getX(self):
        return self.x_ik

    # todo mit Nico abstimmen. Ich glaube die Methoden dürfen/sollten nicht zur Problemklasse gehören (Oli)
    def shift_left(self, k, i):
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

    def attempt_shift_right(self, k, i):


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

# Hilfsklasse für die Berechnung von Ergebnissewerten (objective Function)
# Generelle Trennung in Daten-Speicher-Klasse Problem (Input-Parameter) und Lösungsspeicher-Klasse Solution (Output decision variables)
# todo abstimmen mit Nico
class Solution:

    def __init__(self, x_ik, c_i):
        self.x_ik = x_ik
        self.c_i = c_i

    # todo abstimmen mit Nico
    def calculateObjectiveValue(self, Problem):
        sumDelayPenalty = 0
        for i in range(Problem.n):
            sumDelayPenalty += Problem.p_i[i]*(self.c_i[i] - Problem.a_i[i])
        sumWalkingDistance = 0
        for i in range(Problem.n):
            for j in range(Problem.n):
                for k in range(Problem.m):
                    for l in range(Problem.m):
                        if(self.x_ik[k][i] > -1 and self.x_ik[l][j] > -1):
                            sumWalkingDistance += Problem.w_kl[k][l]*Problem.f_ij[self.x_ik[k][i]][self.x_ik[l][j]]
        return sumWalkingDistance + sumDelayPenalty
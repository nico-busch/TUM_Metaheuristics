class Problem:
    def __init__(self,
                 number_of_flights,
                 number_of_gates,
                 flight_parameters,
                 walking_distance,
                 number_of_passengers):

        self.number_of_flights = number_of_flights
        self.number_of_gates = number_of_gates
        self.flight_parameters = flight_parameters,
        self.walking_distance = walking_distance,
        self.number_of_passengers = number_of_passengers


d = Problem(2, 2, [3, 2], 4, 5)

print(d.number_of_flights)

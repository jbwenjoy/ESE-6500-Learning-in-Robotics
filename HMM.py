import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):

        alpha = ...

        return alpha

    def backward(self):

        beta = ...

        return beta

    def gamma_comp(self, alpha, beta):

        gamma = ...

        return gamma

    def xi_comp(self, alpha, beta, gamma):

        xi = ...

        return xi

    def update(self, alpha, beta, gamma, xi):

        new_init_state = ...
        T_prime = ...
        M_prime = ...

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = ...
        P_prime = ...

        return P_original, P_prime

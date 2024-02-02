import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):
        '''
        Compute alpha_k(x_i) for all k and i
        :return: alpha, a matrix of size num_states x num_obs
        '''
        num_states = len(self.Transition)
        num_obs = len(self.Observations)
        alpha = np.zeros((num_states, num_obs))  # alpha_k are col vectors in alpha

        # Compute alpha_1
        alpha[:, 0] = self.Initial_distribution * self.Emission[:, self.Observations[0]]
        for i in range(1, num_obs):
            alpha[:, i] = self.Emission[:, self.Observations[i]] * np.dot(self.Transition.T, alpha[:, i-1])

        # for i in range(num_obs):
        #     if i == 0:  # alpha_k, k = 1
        #         alpha = self.Initial_distribution * self.Emission[:, self.Observations[i]]  # alpha_1 is a col vector
        #     else:  # alpha_k, k = 2, 3, ..., T
        #         # Compute alpha_kp1
        #         # Sum alpha_k(x_j) * T(x_j, x) for all j
        #         alpha_kp1 = np.zeros((num_states, 1))
        #         sum_alpha_k_T = 0
        #         for j in range(num_states):  # alpha_kp1(x_j)
        #             for q in range(num_states):  # alpha_k(x_q), x_q -> x_j
        #                 sum_alpha_k_T += alpha[q, i-1] * self.Transition[q, j]
        #             # x_j -> y_i
        #             alpha_kp1[j][0] = self.Emission[j, self.Observations[i]] * sum_alpha_k_T
        #         # Concatenate as [alpha, alpha_k]
        #         alpha = np.concatenate((alpha, alpha_kp1), axis=1)

        return alpha

    def backward(self):
        num_states = len(self.Transition)
        num_obs = len(self.Observations)
        beta = np.zeros((num_states, num_obs))  # beta_k are col vectors in beta

        # Compute beta_T
        beta[:, -1] = 1
        for i in range(num_obs-2, -1, -1):
            beta[:, i] = np.dot(self.Transition, self.Emission[:, self.Observations[i+1]] * beta[:, i+1])

        return beta

    def gamma_comp(self, alpha, beta):

        '''
        Compute gamma_k(x_i) for all k and i
        gamma is the probability of each state given all the observations
        gamma_k(x) = alpha_k(x) * beta_k(x) / sum(alpha_k(x)), for all x
        :param alpha: num_states x num_obs
        :param beta: num_states x num_obs
        :return: gamma, a matrix of size num_states x num_obs
        '''
        num_states = len(self.Transition)
        num_obs = len(self.Observations)
        gamma = np.zeros((num_states, num_obs))
        for i in range(num_obs):
            gamma[:, i] = alpha[:, i] * beta[:, i] / np.sum(alpha[:, i])

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


if __name__ == "__main__":
    # Define HMM
    LA = 0
    NY = 1
    null = 2
    Observations = [null, LA, LA, null, NY, null, NY, NY, NY, null, NY, NY, NY, NY, NY, null, null, LA, LA, NY]
    Transition = np.array([[0.5, 0.5], [0.5, 0.5]])
    Emission = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])
    Initial_distribution = np.array([0.5, 0.5])

    # Create HMM
    hmm = HMM(Observations, Transition, Emission, Initial_distribution)

    # Test
    alpha = hmm.forward()
    beta = hmm.backward()
    gamma = hmm.gamma_comp(alpha, beta)

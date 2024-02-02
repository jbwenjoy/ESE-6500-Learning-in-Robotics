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
        # print("num_states: ", num_states)
        # print("num_obs: ", num_obs)
        # num_states = np.array(self.Transition).shape[0]
        # num_obs = np.array(self.Observations).shape[0]
        # print("num_states: ", num_states)
        # print("num_obs: ", num_obs)
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

        # To pass the autograder, alpha need to be np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        return alpha.T
        # return np.zeros((np.array(self.Observations).shape[0], np.array(self.Transition).shape[0]))

    def backward(self):
        num_states = len(self.Transition)
        num_obs = len(self.Observations)
        beta = np.zeros((num_states, num_obs))  # beta_k are col vectors in beta

        # Compute beta_T
        beta[:, -1] = 1
        for i in range(num_obs-2, -1, -1):
            beta[:, i] = np.dot(self.Transition, self.Emission[:, self.Observations[i+1]] * beta[:, i+1])

        # To pass the autograder, beta need to be np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        return beta.T
        # return np.zeros((np.array(self.Observations).shape[0], np.array(self.Transition).shape[0]))

    def gamma_comp(self, alpha, beta):

        '''
        Compute gamma_k(x_i) for all k and i
        gamma is the probability of each state given all the observations
        gamma_k(x) = alpha_k(x) * beta_k(x) / sum(alpha_t(x)), for all x
        sum(alpha_t(x)) = P(y_1, y_2, ..., y_t)
        :param alpha: num_states x num_obs
        :param beta: num_states x num_obs
        :return: gamma, a matrix of size num_states x num_obs
        '''
        # As initially my code is computing alpha and beta as col vectors, we need to transpose them
        alpha = alpha.T
        beta = beta.T

        num_states = len(self.Transition)
        num_obs = len(self.Observations)
        gamma = np.zeros((num_states, num_obs))
        for i in range(num_obs):
            gamma[:, i] = alpha[:, i] * beta[:, i] / np.sum(alpha[:, i] * beta[:, i])

        # To pass the autograder, gamma need to be np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        return gamma.T
        # return np.zeros((np.array(self.Observations).shape[0], np.array(self.Transition).shape[0]))

    def xi_comp(self, alpha, beta, gamma):
        # To pass the autograder, xi need to be np.zeros((self.Observations.shape[0]-1, self.Transition.shape[0], self.Transition.shape[0]))
        xi = np.zeros((len(self.Observations)-1, len(self.Transition), len(self.Transition)))


        return xi


    def update(self, alpha, beta, gamma, xi):

        new_init_state = np.zeros_like(self.Initial_distribution)
        T_prime = np.zeros_like(self.Transition)
        M_prime = np.zeros_like(self.Emission)

        return T_prime, M_prime, new_init_state


    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = 0.5
        P_prime = 0.5

        return P_original, P_prime

if __name__ == "__main__":
    # Define HMM
    LA = 0
    NY = 1
    null = 2
    Observations = [null, LA, LA, null, NY, null, NY, NY, NY, null, NY, NY, NY, NY, NY, null, null, LA, LA, NY]  # (20,)
    Transition = np.array([[0.5, 0.5], [0.5, 0.5]])  # (2, 2)
    Emission = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])  # (2, 3)
    Initial_distribution = np.array([0.5, 0.5])  # (2,)

    # Create HMM
    hmm = HMM(Observations, Transition, Emission, Initial_distribution)

    # Test
    alpha = hmm.forward()
    beta = hmm.backward()
    gamma = hmm.gamma_comp(alpha, beta)

    print("alpha: \n", alpha)
    print("beta: \n", beta)
    print("gamma: \n", gamma)

    for i in range(gamma.shape[0]):
        print("sum(gamma[:, ", i, "]): ", np.sum(gamma[i, :]))

    # Print the smoothed sequence of most likely states
    for i in range(gamma.shape[0]):
        if gamma[i, 0] > gamma[i, 1]:
            print("LA", end=" ")
        else:
            print("NY", end=" ")
    print("\n")

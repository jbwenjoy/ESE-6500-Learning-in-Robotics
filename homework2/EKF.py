import numpy as np
import matplotlib.pyplot as plt


class system:
    def __init__(self, a):
        self.a = a  # dynamics parameter, to be estimated by EKF

    def dynamics(self, xk):
        epsilon_k = np.random.normal(0, 1)
        x_kp1 = self.a * xk + epsilon_k
        return x_kp1

    def observation(self, xk):
        nu_k = np.random.normal(0, np.sqrt(0.5))
        yk = np.sqrt(xk ** 2 + 1) + nu_k
        return yk

    def simulate(self, x0, N=100):
        x = np.zeros(N)
        y = np.zeros(N)
        x[0] = x0
        y[0] = self.observation(x0)
        for k in range(1, N):
            x[k] = self.dynamics(x[k - 1])
            y[k] = self.observation(x[k])
        return x, y


class EKF:
    def __init__(self, x_hat, a_hat, P, Q, R):
        self.x_hat = x_hat
        self.P = P
        self.Q = Q
        self.R = R
        self.a_hat = a_hat
        self.a_hat_history = []

    def prediction(self, xk_hat):
        # predict the state and cov(state) at k+1 given k
        xkp1_hat = self.a_hat * xk_hat
        P = (self.a_hat ** 2) * self.P + self.Q
        return xkp1_hat, P

    def update_step(self, yk, x_hat_kp1):
        # Jacobian of the measurement function wrt the state
        H_k = x_hat_kp1 / np.sqrt(x_hat_kp1 ** 2 + 1)
        # Measurement prediction
        y_hat_k = np.sqrt(x_hat_kp1 ** 2 + 1)
        # Kalman gain
        K_k = self.P * H_k / (H_k ** 2 * self.P + self.R)
        # State update
        self.a_hat += K_k * (yk - y_hat_k)
        # Covariance update
        self.P = (1 - K_k * H_k) * self.P
        # Store the estimate
        self.a_hat_history.append(self.a_hat)

    def run_ekf(self, D):
        # Initialize state estimate with first measurement assuming a=1
        x_hat_k = np.sqrt(D[0] ** 2 - 1)
        for yk in D:
            # Prediction step
            x_hat_kp1, P_kp1 = self.prediction(x_hat_k)
            # Update step
            self.update_step(yk, x_hat_kp1)
            # The new estimate becomes the current estimate for the next iteration
            x_hat_k = x_hat_kp1
            self.P = P_kp1

        return self.a_hat_history, self.P


if __name__ == "__main__":
    a = -1  # true value of a
    sys = system(a)
    x0 = np.random.normal(1, np.sqrt(2))
    _, D = sys.simulate(x0)  # we only know the observation

    # EKF
    x_hat = 0
    a_hat = 1
    P = 1
    Q = 1
    R = 0.5
    ekf = EKF(x_hat, a_hat, P, Q, R)
    a_hat_history, P = ekf.run_ekf(D)

    # Plot
    plt.plot(a_hat_history)
    plt.axhline(a, color='r', linestyle='--')
    plt.xlabel('iteration')
    plt.ylabel('a_hat')
    plt.title('EKF estimate of a')
    plt.show()
    print('Final estimate of a:', a_hat_history[-1])
    print('Final covariance:', P)


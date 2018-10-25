import numpy as np
import sys

if __name__ == '__main__':
#function to get data from console
    def get_data():
        my_data = np.genfromtxt(sys.argv[1], delimiter=',')
        X = my_data[:, 0:-1]
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate((ones, X), 1)
        y = my_data[:, -1].reshape(-1, 1)
        return X, y

# cost function
    def compute_cost(X, y, theta):
        summation = ((X @ theta.T)-y) ** 2
        return np.sum(summation)/(2 * len(X))

#gradient descent
    def gradient_descent(X, y, theta_1, alpha, threshold):
        i = 0
        cost = np.zeros(100000)
        error = np.sum(X @ theta_1.T - y, axis=0)
        print(error)
        while True:
            theta_1 = theta_1 - alpha / len(X) * np.sum((X @ theta_1.T - y) * X, axis=0)
            cost[i] = compute_cost(X, y, theta_1)
            if np.sum((X @ theta_1.T - y), axis=0) - error <= threshold:
                break
            i += 1
        print(i)
        return theta_1, cost
    X, y = get_data()
    alpha = float(sys.argv[2])
    threshold = float(sys.argv[3])
    theta = np.zeros([1, X.shape[1]])
    g, cost = gradient_descent(X, y, theta, alpha, threshold)
    print(g)
    final_cost = compute_cost(X,y,g)
    print(final_cost)
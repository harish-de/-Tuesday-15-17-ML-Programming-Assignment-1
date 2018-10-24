import numpy as np
import sys

if __name__ == '__main__':
    def get_data():
        my_data = np.genfromtxt(sys.argv[1], delimiter=',')
        X = my_data[:, 0:-1]
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate((ones, X), 1)
        y = my_data[:, -1].reshape(-1, 1)
        return X, y

    def compute_cost(X, y, theta):
        summation = ((X @ theta.T)-y) ** 2
        return np.sum(summation)/(2 * len(X))


    X, y = get_data()
    alpha = sys.argv[2]
    threshold = sys.argv[3]
    print(compute_cost(X, y, np.zeros([1,X.shape[1]])))


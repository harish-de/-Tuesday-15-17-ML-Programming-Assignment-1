import numpy as np
import sys

if __name__ == '__main__':
    # function to get data from console
    def get_data():
        # my_data = np.genfromtxt(sys.argv[1], delimiter=',')
        my_data = np.genfromtxt('yacht.csv', delimiter=',')
        X = my_data[:, 0:-1]
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate((ones, X), 1)
        y = my_data[:, -1].reshape(-1, 1)
        return X, y


    # cost function
    def compute_cost(X, y, theta):
        summation = ((X @ theta.T) - y) ** 2
        return np.sum(summation) / (2 * len(X))


    def get_error(X,y,theta):
        return np.sum(X @ theta.T - y)


    # gradient descent
    def gradient_descent(X, y, theta, alpha, threshold):
        i = 0
        cost = np.zeros(1000000)
        print(get_error(X, y, theta))
        while True:
            temp = theta
            theta = theta - (alpha / len(X)) * (np.sum((X @ theta.T - y) * X, axis=0))
            cost = compute_cost(X, y, theta)
            x = abs(abs(get_error(X, y, theta)) - abs(get_error(X, y, temp)))
            #if i % 10 == 0:  # just look at cost every ten loops for debugging
                #print(cost)
            i += 1
            if abs(get_error(X, y, theta)) - abs(get_error(X, y, temp)) <= threshold:
                break
        print(get_error(X, y, theta) - get_error(X, y, temp))
        print()
        print(i)
        return theta, cost


    X, y = get_data()
    alpha = 0.0001  # float(sys.argv[2]) learning rate
    threshold = 0.0001  # float(sys.argv[3])
    theta = np.zeros([1, X.shape[1]])
    g, cost = gradient_descent(X, y, theta, alpha, threshold)
    print(g)
    final_cost = compute_cost(X, y, g)
    print(final_cost)
